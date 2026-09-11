# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.systems.system import System, SystemParams
import glob
import os
import re
from typing import NamedTuple, Optional

import logging

logger = logging.getLogger(__name__)


def _parse_rocm_version(text: str) -> Optional[tuple[int, int, int]]:
    """
    Extract a ROCm version triple from strings such as ``rocm-6.4.2``,
    ``rocm-7.1``, or a ``torch.version.hip`` string like
    ``7.2.24191-cf58cf3856``. A missing patch component is treated as 0.

    :return: ``(major, minor, patch)``, or ``None`` when no
             ``major.minor`` version is present at all.
    """
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def _version_str(version: tuple[int, int, int]) -> str:
    return ".".join(str(v) for v in version)


def _rocm_path_version() -> Optional[tuple[int, int, int]]:
    """
    The ROCm version encoded in ``$ROCM_PATH``. The path is resolved
    through ``os.path.realpath`` first so the conventional unversioned
    ``/opt/rocm`` symlink still yields a version when it points at a
    ``rocm-X.Y.Z`` tree.
    """
    rocm_path = os.getenv("ROCM_PATH")
    if not rocm_path:
        return None
    return _parse_rocm_version(os.path.basename(os.path.realpath(rocm_path)))


def _torch_hip_version() -> Optional[tuple[int, int, int]]:
    """
    The ROCm version bundled with the installed torch wheel, from
    ``torch.version.hip``. The import is deliberately lazy and guarded:
    CLI startup and non-torch users must never require torch.
    """
    try:
        import torch
    except Exception:
        return None
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if not hip:
        return None
    return _parse_rocm_version(str(hip))


class _RocmRuntime(NamedTuple):
    """Resolution of the ROCm runtime version a launched job will use."""

    # The resolved version, used for all version-dependent configuration
    version: Optional[tuple[int, int, int]]
    # Where the version came from: "torch" or "ROCM_PATH" (None if unknown)
    source: Optional[str]
    # True when torch and ROCM_PATH both report a version and they
    # disagree at the major.minor level
    mismatch: bool


def _rocm_runtime_version() -> _RocmRuntime:
    """
    Resolve the ROCm runtime version torch-first: a torch wheel bundles
    its own ROCm runtime -- the one RCCL actually links against -- so
    ``torch.version.hip`` wins over the version encoded in ``ROCM_PATH``.
    Logs a prominent warning when the two disagree.
    """
    torch_version = _torch_hip_version()
    path_version = _rocm_path_version()
    mismatch = (
        torch_version is not None
        and path_version is not None
        and torch_version[:2] != path_version[:2]
    )
    if mismatch:
        logger.warning(
            "ROCm version mismatch: torch's bundled ROCm runtime is "
            f"{_version_str(torch_version)} (torch.version.hip) but "
            f"ROCM_PATH={os.getenv('ROCM_PATH')} is ROCm "
            f"{_version_str(path_version)}. The torch wheel's bundled ROCm "
            "wins: the RCCL/NCCL configuration is derived from ROCm "
            f"{torch_version[0]}.{torch_version[1]}."
        )
    if torch_version is not None:
        return _RocmRuntime(torch_version, "torch", mismatch)
    if path_version is not None:
        return _RocmRuntime(path_version, "ROCM_PATH", mismatch)
    return _RocmRuntime(None, None, False)


# Root of the LC-provided aws-ofi-rccl plugin installs; the probe looks in
# "{root}/{SYS_TYPE}/rocm-X.Y.Z/install/lib". Module-level so tests can
# point it at a scratch tree.
_AWS_OFI_RCCL_ROOT = "/collab/usr/global/tools/rccl"


def _find_aws_ofi_plugin_dir(version: tuple[int, int, int]) -> Optional[str]:
    """
    Locate the aws-ofi-rccl plugin lib directory for a ROCm version.

    Tries the exact ``rocm-X.Y.Z`` tree first, then falls back to any
    ``rocm-X.Y.*`` sibling with the same major.minor: the plugin trees are
    not installed for every patch level, and a torch wheel's HIP build
    number (e.g. 7.2.24191) never names one exactly.

    :return: The plugin lib directory, or ``None`` when no tree matches.
    """
    major, minor, patch = version
    base = f'{_AWS_OFI_RCCL_ROOT}/{os.getenv("SYS_TYPE")}'
    exact = os.path.join(base, f"rocm-{major}.{minor}.{patch}", "install", "lib")
    if os.path.isdir(exact):
        return exact
    candidates = [
        path
        for path in glob.glob(
            os.path.join(base, f"rocm-{major}.{minor}.*", "install", "lib")
        )
        if os.path.isdir(path)
    ]
    if not candidates:
        return None

    def _tree_version(path: str) -> tuple[int, int, int]:
        tree = os.path.basename(os.path.dirname(os.path.dirname(path)))
        return _parse_rocm_version(tree) or (0, 0, 0)

    # Prefer the highest available patch level for determinism.
    return max(candidates, key=_tree_version)

# Known LC systems
_mi250x_node = SystemParams(64, 8, "gfx90a", 64.0, 4, "flux")
# APUs can run into a snarl where they OOM if too much GPU memory is allocated
_mi300a_node = SystemParams(96, 4, "gfx942", 128.0, 4, "flux", 0.8)
_system_params = {
    "tioga": (
        "pdebug",
        {
            "pdebug": _mi250x_node,
            "mi300a": _mi300a_node,
        },
    ),
    "tuolumne": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "elcap": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "rzadams": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
    "rzvernal": (
        "pdebug",
        {
            "pdebug": _mi250x_node,
            "pllm":   _mi250x_node,
        },
    ),
    "tenaya": (
        "pbatch",
        {
            "pbatch": _mi300a_node,
            "pdebug": _mi300a_node,
        },
    ),
}


class ElCapitan(System):
    """
    LLNL LC Systems based on the El Capitan MI300a architecture.
    """

    def __init__(self, system_name):
        super().__init__(system_name, _system_params)

    def environment_variables(self) -> list[tuple[str, str]]:
        env_list = []
        env_list.append(("NCCL_NET_GDR_LEVEL", "3"))  # From HPE to avoid hangs
        env_list.append(
            ("NCCL_MIN_NCHANNELS", "24")
        )  # From AMD to improve collective performance
        env_list.append(("MIOPEN_DEBUG_DISABLE_FIND_DB", "0"))
        env_list.append(("MIOPEN_DISABLE_CACHE", "0"))
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            env_list.append(("MIOPEN_USER_DB_PATH", f"{tmpdir}/MIOpen_user_db"))
            env_list.append(("MIOPEN_CUSTOM_CACHE_DIR", f"{tmpdir}/MIOpen_custom_cache"))

        if os.getenv("CRAY_LD_LIBRARY_PATH") is not None:
            env_list.append(
                (
                    "LD_LIBRARY_PATH",
                    os.getenv("CRAY_LD_LIBRARY_PATH") + ":${LD_LIBRARY_PATH}",
                )
            )

        optimize_rccl_protocol = False
        optimize_comm_protocol = ""
        if self.job_comm_protocol:
            optimize_comm_protocol = self.job_comm_protocol
        if optimize_comm_protocol.upper() == "RCCL" or optimize_comm_protocol.upper() == "*CCL":
            optimize_rccl_protocol = True

        aws_ofi_plugin = None
        different_ofi_plugin = os.getenv("LBANN_USE_THIS_OFI_PLUGIN")
        if different_ofi_plugin is not None:
            if os.path.isdir(different_ofi_plugin):
                env_list.append(
                    ("LD_LIBRARY_PATH", different_ofi_plugin + ":${LD_LIBRARY_PATH}")
                )
                aws_ofi_plugin = different_ofi_plugin
            else:
                logger.warn(f"WARNING: invalid path provided in LBANN_USE_THIS_OFI_PLUGIN: {different_ofi_plugin}. Ensure one is loaded or performance will be degraded.")

        # Resolve the ROCm runtime version torch-first:
        # a torch wheel bundles its own ROCm runtime -- the one RCCL
        # actually runs against -- so it takes precedence over the version
        # of whatever environment module set ROCM_PATH. Note that a torch
        # wheel may also require ROCM_PATH to be unset entirely, so nothing
        # below except the llvm/lib prepend may depend on ROCM_PATH.
        rocm = _rocm_runtime_version()
        rocm_path = os.getenv("ROCM_PATH")

        if rocm_path is not None:
            llvm_lib_path = os.path.join(f"{rocm_path}", "llvm", "lib")
            if rocm.mismatch:
                # Mixing another ROCm's llvm/lib into a process that runs
                # the torch wheel's bundled ROCm is an ABI hazard.
                logger.warning(
                    f"Not prepending {llvm_lib_path} to LD_LIBRARY_PATH: the "
                    "ROCm version in ROCM_PATH differs from the torch wheel's "
                    "bundled ROCm runtime."
                )
            else:
                env_list.append(
                    (
                        "LD_LIBRARY_PATH",
                        llvm_lib_path + ":${LD_LIBRARY_PATH}",
                    )
                )

        if rocm.version is None:
            # Never crash on an undeterminable ROCm version: skip the
            # version-dependent configuration. Stay quiet when there is no
            # sign of ROCm use at all.
            if rocm_path is not None or optimize_rccl_protocol:
                logger.warning(
                    "Could not determine the ROCm runtime version (torch "
                    f"reports no HIP runtime and ROCM_PATH={rocm_path} does "
                    "not resolve to a rocm-X.Y.Z tree); skipping the "
                    "ROCm-version-dependent RCCL/NCCL configuration."
                )
        else:
            if optimize_rccl_protocol and not aws_ofi_plugin:
                # Check for and include the AWS_OFI_PLUGIN if it exists
                aws_ofi_plugin = _find_aws_ofi_plugin_dir(rocm.version)
                if aws_ofi_plugin is not None:
                    logger.info(f"Setting path to default AWS_OFI_RCCL plugin {aws_ofi_plugin} to accelerate RCCL communication protocol.")
                    env_list.append(
                        (
                            "LD_LIBRARY_PATH",
                            aws_ofi_plugin
                            + ":${LD_LIBRARY_PATH}",
                        )
                    )
                else:
                    checked = f'{_AWS_OFI_RCCL_ROOT}/{os.getenv("SYS_TYPE")}/rocm-{rocm.version[0]}.{rocm.version[1]}.*/install/lib'
                    logger.warning(
                        "No AWS OFI RCCL (libfabric) plugin was found for ROCm "
                        f"{_version_str(rocm.version)} (checked {checked}). "
                        "NCCL_NET is left unset, so RCCL will fall back to its "
                        "built-in transports: multi-node jobs will not use the "
                        "Slingshot fabric plugin and may underperform or fail "
                        "to scale. Install the plugin or point "
                        "LBANN_USE_THIS_OFI_PLUGIN at an existing plugin lib "
                        "directory."
                    )

            # Only force the libfabric NET plugin when one was actually
            # found (probe hit or explicit override): forcing NCCL_NET
            # without the plugin present hard-crashes RCCL initialization
            # with "Failed to initialize any NET plugin", even for
            # single-node jobs.
            if aws_ofi_plugin is not None:
                # Unless overriden by an external env variable set the NCCL_NET to ensure that the libfabric interface is used, e.g.: libfabric, IB, Socket
                msg = "HPC-launcher forces slingshot systems to use the detected libfabric NCCL/RCCL plugin.  This behavior can be overridden by setting NCCL_NET=Socket in the calling environment."
                if rocm.version[:2] >= (7, 1):
                    # Add AWS_OFI_NCCL for ROCm 7.1 - Ensure that it pick up the correct library object
                    if not os.getenv("NCCL_NET_PLUGIN"):
                        env_list.append(("NCCL_NET_PLUGIN", "librccl-net.so"))
                    if not os.getenv("NCCL_NET"):
                        env_list.append(("NCCL_NET", "libfabric", msg))
                else:
                    if not os.getenv("NCCL_NET"):
                        env_list.append(("NCCL_NET", '\"AWS Libfabric\"', msg))

        if optimize_rccl_protocol:
            # Performance tuning for HPE Slingshot Cassini NIC (Audited on 3/31/25) - Only use with RCCL
            msg = "Performance tuning for RCCL + HPE Slingshot Cassini NIC (Audited on 3/31/25)"
            env_list.append((f"\n# {msg}",))
            env_list.append(("FI_CXI_RDZV_PROTO", "alt_read", msg))
            env_list.append(("FI_CXI_RDZV_THRESHOLD", "0", msg))
            env_list.append(("FI_CXI_RDZV_GET_MIN", "0", msg))
            env_list.append(("FI_CXI_RDZV_EAGER_SIZE", "0", msg))

        # Known issue with memhooks and RCCL hangs (Audited on 3/31/25)
        # https://support.hpe.com/hpesc/public/docDisplay?docId=dp00004854en_us&docLocale=en_US
        # env_list.append(("FI_MR_CACHE_MAX_COUNT", "0")) # MPI has a significant performance hit
        # kdreg2 will be the future
        env_list.append(("\n# Known issue with memhooks and RCCL hangs (Audited on 3/31/25)",))
        env_list.append(("# https://support.hpe.com/hpesc/public/docDisplay?docId=dp00004854en_us&docLocale=en_US",))
        msg = "Known issue with memhooks and RCCL hang (Audited on 3/31/25)"
        env_list.append(("FI_MR_CACHE_MONITOR", "userfaultfd", msg)) # This should work and be safe and performant
        msg = "Performance tuning for HPE Slingshot Cassini NIC (Audited on 3/31/25)"
        env_list.append(("FI_CXI_DEFAULT_TX_SIZE", "1024", msg))
        env_list.append(("FI_CXI_DISABLE_HOST_REGISTER", "1", msg))
        env_list.append(("FI_CXI_DEFAULT_CQ_SIZE", "131072", msg))
        # Run in hardware until the HW queues are exhausted, then fallback to SW
        env_list.append(("FI_CXI_RX_MATCH_MODE", "hybrid", msg)) # set to software instead when setting up the alt_read

        env_list.append(("\n# General tuning knobs (Audited on 3/31/25)",))
        # =2 may be a future performance improvement (Removes rails configuration)
        env_list.append(("NCCL_CROSS_NIC", "1"))
        # Improve the performance of large scale RCCL initialization - should only be used on wire-up
        env_list.append(("NCCL_SOCKET_IFNAME", "hsi0"))

        # Ensure that PyTorch respects channel's last for MIOpen (Audited on 1/13/2026)
        env_list.append(("PYTORCH_MIOPEN_SUGGEST_NHWC", "1"))
        env_list.append(("PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM", "1"))
        # Ensure that MIOpen uses Stream-k for PyTorch backwards operations in 7.x (Audited on 3/17/2026)
        env_list.append(("TENSILE_SOLUTION_SELECTION_METHOD", "2"))

        for i in self._aux_env_list:
            env_list.append(i)

        return env_list

    def customize_scheduler(self, scheduler):
        use_this_rccl = os.getenv("LBANN_USE_THIS_RCCL")
        if type(scheduler) is FluxScheduler:
            # Not when launching nested jobs inside an existing allocation:
            # --exclusive makes every nested `flux run` demand exclusive use
            # of the node, so concurrent launches in one allocation
            # serialize instead of packing side by side.
            if not scheduler.in_allocation():
                scheduler.common_launch_args["--exclusive"] = None # This is an alloc only on slurm and alloc or run on flux
            # Note that options cannot have a space after the -o flag, e.g. -o<option>
            # Performance tuning for HPE Slingshot Cassini NIC
            scheduler.common_launch_args["-ofastload"] = "on"
            scheduler.common_launch_args["--setattr=rdzv_get_en"] = "0"
            if scheduler.in_allocation() or scheduler.gpus_per_proc > 1:
                # mpibind computes a task's GPUs from the NUMA locality of
                # its cores, not from the GPU set Flux granted the job.
                # That breaks two cases:
                # - A nested job sharing a node: mpibind can export a GPU
                #   that Flux granted to a *different* concurrent job
                #   (whichever one owns that domain's cores), silently
                #   double-booking it.
                # - gpus_per_proc > 1: cores in one MI300A domain can never
                #   see more than that domain's single GPU, whatever
                #   --gpus-per-task requested.
                # Use Flux's own affinity plugins for those; they bind CPUs
                # and export exactly the granted, per-task GPU set.
                scheduler.common_launch_args["-ompibind"] = "off"
                scheduler.common_launch_args["-ogpu-affinity"] = "per-task"
                scheduler.common_launch_args["-ocpu-affinity"] = "per-task"
            else:
                # Avoid bug in OMP that ruins the CPU_SET
                scheduler.common_launch_args["-ompibind"] = "omp_proc_bind,omp_places"

        # if type(scheduler) is SlurmScheduler:
        #     scheduler.submit_args["--exclusive"] = None # This is an alloc only on slurm and alloc or run on flux
            
        if use_this_rccl is not None:
            scheduler.ld_preloads = [f"{use_this_rccl}"]

        return

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return FluxScheduler
