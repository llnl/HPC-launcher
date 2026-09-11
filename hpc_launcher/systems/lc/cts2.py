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
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems.system import System, SystemParams
import os

_h100_node = SystemParams(112, 4, "sm_90", 80.0, 8, "slurm")

# Known LC systems
_system_params = {
    "ipa": (
        "a100",
        {
            "a100": SystemParams(32, 2, "sm_80", 40.0, 1, "slurm"),
            "aa100": SystemParams(16, 2, "sm_80", 40.0, 2, "slurm"),
            "av100": SystemParams(32, 2, "sm_70", 32.0, 2, "slurm"),
            "v100": SystemParams(16, 2, "sm_70", 32.0, 2, "slurm"),
        },
    ),
    "matrix": (
        "pbatch",
        {
            "pbatch": _h100_node,
            "pdebug": _h100_node,
            "erl": _h100_node,
        },
    ),
    "rzvector": (
        "pbatch",
        {
            "pbatch": _h100_node,
            "pdebug": _h100_node,
        },
    ),
}


class CTS2(System):
    """
    LLNL LC Systems based on the Commodity Technology System platform.
    """

    def __init__(self, system_name):
        super().__init__(system_name, _system_params)

    def environment_variables(self) -> list[tuple[str, str]]:
        env_list = []
        env_list.append(("MPICH_OFI_NIC_POLICY", "GPU"))

        for i in self._aux_env_list:
            env_list.append(i)

        return env_list

    def customize_scheduler(self, scheduler):
        if self.system_name == "matrix" and type(scheduler) is SlurmScheduler:
            # Not for a nested job step (srun inside an existing
            # allocation): these two flags disable exactly the two
            # mechanisms that export per-task CUDA_VISIBLE_DEVICES into a
            # step -- the mpibind SPANK plugin and Slurm's own GPU binding
            # -- so with them a step's GPU-visibility variable is silently
            # left unset (the cgroup still confines the devices, but CUDA
            # code that reads the variable sees nothing).
            if not scheduler.in_allocation():
                scheduler.run_only_args["--mpibind"] = "off"
                scheduler.run_only_args["--gpu-bind"] = "none"

        return

    @property
    def preferred_scheduler(self) -> type[Scheduler]:
        return SlurmScheduler
