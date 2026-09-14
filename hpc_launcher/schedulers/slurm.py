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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from io import StringIO
import os
import re
import subprocess

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems.system import System

from hpc_launcher.systems.lc.sierra_family import Sierra

from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers import parse_env_list

import logging

logger = logging.getLogger(__name__)


def _time_string(minutes):
    """Time D-hh:mm:ss format."""
    minutes = max(minutes, 0)
    seconds = int(round((minutes % 1) * 60))
    hours, minutes = divmod(int(minutes), 60)
    days, hours = divmod(hours, 24)
    return f"{days}-{hours:02}:{minutes:02}:{seconds:02}"


@dataclass
class SlurmScheduler(Scheduler):

    @classmethod
    def in_allocation(cls) -> bool:
        # SLURM_JOB_ID is set by salloc and sbatch. When it is present srun
        # creates a nested job step within that allocation rather than
        # requesting a new one -- provided its options stay compatible with
        # the enclosing job (see build_scheduler_specific_arguments).
        return os.getenv("SLURM_JOB_ID") is not None

    def build_scheduler_specific_arguments(
        self, system: "System", blocking: bool = True
    ):
        # A blocking launch inside an existing allocation is a nested job
        # step: srun inherits the allocation from SLURM_JOB_ID, and any
        # allocation-level option that contradicts the enclosing job
        # (partition, account, reservation, exclusivity, or a node count the
        # allocation cannot cover) makes srun try to satisfy the request as a
        # *new* allocation instead of running the step in the current one.
        # Non-blocking (sbatch) submissions are left untouched: submitting a
        # new batch job from inside an allocation is a deliberate new job.
        nested_job_step = blocking and self.in_allocation()

        if self.out_log_file and not blocking:
            self.submit_only_args["--output"] = f"{self.out_log_file}"
        if self.err_log_file and not blocking:
            self.submit_only_args["--error"] = f"{self.err_log_file}"

        # Unbuffered output - Only pass to srun
        if blocking and not isinstance(system, Sierra):
            # On Sierra family systems srun is a proxy to lrun and lacks this flag
            self.run_only_args["-u"] = None

        # Since Slurm 20.11 a job step is given exclusive use of all CPUs on
        # the nodes it runs on, so a second concurrent step inside the same
        # allocation blocks on "Job <id> step creation temporarily disabled,
        # retrying (Requested nodes are busy)" until the first finishes.
        # Concurrent steps are enabled in one of two ways, depending on
        # whether the step's resource footprint was stated:
        #
        # - A footprint was given (--cpus-per-task and/or GPUs per task):
        #   --exact confines the step to exactly the resources it asked for,
        #   so disjoint concurrent steps pack side by side without sharing.
        # - No footprint: --overlap lets steps share resources outright
        #   (sizing them so they do not contend is up to the user; remove
        #   the flag with `-x ~--overlap` to restore exclusive steps).
        #
        # --exclusive means the user asked for dedicated resources, which
        # both flags would undermine.
        if nested_job_step and not self.exclusive:
            if self.cpus_per_task or self.gpus_per_proc > 0:
                self.run_only_args["--exact"] = None
                # --exact only confines the *CPUs*; the two implicitly
                # granted resources still make concurrent steps serialize:
                #
                # - Memory: a step with no --mem consumes ALL of the job's
                #   memory, queueing every later step behind it. --mem=0 is
                #   the documented remedy: the step may use up to the job's
                #   memory but does not remove any of it from availability
                #   to other steps.
                # - GRES: a step with no GRES request implicitly holds all
                #   of the job's GPUs, so a CPU-only step would block every
                #   GPU step. --gres=none makes it hold none.
                self.run_only_args["--mem"] = "0"
                if self.gpus_per_proc == 0:
                    self.run_only_args["--gres"] = "none"
            else:
                self.run_only_args["--overlap"] = None

        # Number of Nodes
        if nested_job_step:
            # Within the allocation, -N <= the allocation's node count runs
            # as a job step on the nodes already held; asking for more is
            # what silently turns the srun into a request for a brand-new
            # allocation (which then pends behind the one the user is
            # sitting in). Fail fast with an explanation instead.
            alloc_nodes = self.num_nodes_in_allocation()
            if alloc_nodes is not None and self.nodes > alloc_nodes:
                raise ValueError(
                    f"Requested {self.nodes} nodes inside an allocation of "
                    f"{alloc_nodes} node(s): srun would treat this as a new "
                    f"allocation request rather than a job step. Request at "
                    f"most {alloc_nodes} node(s) (or omit the job-size flags "
                    f"to inherit the allocation's size)."
                )
        self.common_launch_args["--nodes"] = f"{self.nodes}"

        # Total number of Tasks / Processes
        self.common_launch_args["--ntasks"] = f"{self.nodes * self.procs_per_node}"

        # Number of Tasks per node
        self.common_launch_args["--ntasks-per-node"] = f"{self.procs_per_node}"

        # CPUs per task. On srun --cpus-per-task implies --exact, giving a
        # nested job step a precise CPU footprint so concurrent steps pack
        # side by side instead of one step claiming every CPU on its nodes.
        if self.cpus_per_task:
            self.common_launch_args["--cpus-per-task"] = f"{self.cpus_per_task}"

        # Set the Number of GPUs per task
        if self.gpus_per_proc > 0:
            self.common_launch_args["--gpus-per-task"] = f"{self.gpus_per_proc}"

        # Request for node exclusivity
        if self.exclusive:
            self.submit_only_args["--exclusive"] = ""

        if self.work_dir:
            self.submit_only_args["--chdir"] = f"{os.path.abspath(self.work_dir)}"

        if self.ld_preloads:
            self._merge_export([f'LD_PRELOAD={",".join(self.ld_preloads)}'])

        if self.time_limit is not None:
            self.common_launch_args["--time"] = f"{_time_string(self.time_limit)}"

        if self.dependency is not None:
            self.common_launch_args["--dependency"] = f"{self.dependency}"
        dependency = self.common_launch_args.get('--dependency', None)
        if self.override_launch_args and self.override_launch_args.get('--dependency', None):
            dependency = self.override_launch_args['--dependency']
        if dependency and not blocking:
            try:
                del self.common_launch_args['--dependency']
            except KeyError:
                pass
            try:
                if self.override_launch_args:
                    del self.override_launch_args['--dependency']
            except KeyError:
                pass
            self.submit_only_args["--dependency"] = dependency

        if self.job_name:
            self.common_launch_args["--job-name"] = f"{self.job_name}"

        # Allocation-selection options. For a nested job step these are
        # already fixed by the enclosing allocation; passing a conflicting
        # value makes srun request a new allocation instead of running a
        # step, so drop them (loudly) rather than forward them.
        for flag, value in (
            ("--partition", self.queue),
            ("--account", self.account),
            ("--reservation", self.reservation),
        ):
            if not value:
                continue
            if nested_job_step:
                logger.warning(
                    f"WARNING: Dropping {flag}={value}: it selects an "
                    f"allocation, and this launch runs as a job step inside "
                    f"the existing allocation {os.getenv('SLURM_JOB_ID')}"
                )
            else:
                self.submit_only_args[flag] = f"{value}"

        return

    def batch_script_prefix(self) -> str:
        return "#SBATCH"

    def blocking_launch_command(self) -> list[str]:
        return ["srun"]

    def nonblocking_launch_command(self) -> list[str]:
        return ["sbatch"]

    def _merge_export(self, entries: list[str]) -> None:
        """
        Fold ``NAME=value`` entries into Slurm's single ``--export`` token.

        srun and sbatch accept only one ``--export``; a second occurrence
        replaces the first rather than adding to it. Every producer of
        exported variables therefore has to merge into the same argument
        instead of introducing its own -- which is how LD_PRELOAD used to be
        lost on the ephemeral path, where it was emitted as a separate
        ``--export=ALL,LD_PRELOAD`` alongside the environment's ``--export``.

        :param entries: ``NAME=value`` strings to add.
        """
        if "--export" in self.submit_only_args:
            self.submit_only_args["--export"] += "," + ",".join(entries)
        else:
            self.submit_only_args["--export"] = "ALL," + ",".join(entries)

    def cli_env_arg(self, env_list) -> None:
        # Expand ${VAR} references, merge duplicate keys, and dequote values
        # like the shell-script path would before folding them into Slurm's
        # single --export=ALL,k=v,... token.
        env_vars = [f"{k}={v}" for k, v in self.expand_cli_env(env_list).items()]
        self._merge_export(env_vars)
        return

    def export_hostlist(self) -> str:
        return "export HPC_LAUNCHER_HOSTLIST=${SLURM_JOB_NODELIST}\n"

    def internal_script_run_command(self) -> str:
        return "srun -u "

    def ephemeral_identity_wrapper(self) -> list[str]:
        """Have rank zero announce the Slurm job-step ID before user code."""
        return [
            "/bin/sh",
            "-c",
            (
                'if [ "${SLURM_PROCID:-}" = 0 ] '
                '&& [ -n "${SLURM_JOB_ID:-}" ] '
                '&& [ -n "${SLURM_STEP_ID:-}" ]; then '
                'printf "HPC_LAUNCHER_STEP_ID=%s.%s\\n" '
                '"$SLURM_JOB_ID" "$SLURM_STEP_ID" >&2; '
                'fi; exec "$@"'
            ),
            "hpc-launcher",
        ]

    @classmethod
    def cancel_command(cls, identifier: str) -> list[str]:
        """Return the Slurm command that cancels a job or job step."""
        return ["scancel", identifier]

    def get_job_id(self, output: str) -> Optional[str]:
        # The job ID is the last number in the printout
        last_line = output.strip().split("\n")[-1].strip()
        if last_line.startswith("Submitted batch job"):
            return last_line.split(" ")[-1]
        return None

    @classmethod
    def num_nodes_in_allocation(cls) -> Optional[int]:
        # Slurm only. salloc and sbatch export SLURM_JOB_NUM_NODES alongside
        # SLURM_JOB_ID, but a shell that only had SLURM_JOB_ID exported into
        # it (e.g. after ssh-ing to an allocated node) still runs srun as a
        # nested step, so ask the controller for the job's size in that case.
        if not cls.in_allocation():
            return None
        nodes = os.getenv("SLURM_JOB_NUM_NODES")
        if nodes:
            return int(nodes)
        try:
            proc = subprocess.run(
                ["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%D"],
                universal_newlines=True,
                capture_output=True,
            )
        except FileNotFoundError:
            return None
        m = re.match(r"\s*(\d+)\s*$", proc.stdout)
        return int(m.group(1)) if m else None

    @classmethod
    def get_parallel_rank_env_variable(self) -> str:
        return "${SLURM_PROCID}"

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        # Interesting but unused variables SLURM_JOB_NUM_NODES, SLURM_NPROCS, SLURM_DISTRIBUTION
        # Skipping 'SLURM_TASKS_PER_NODE' because this field has a weird format e.g. 2(x2)
        env_vars = ["SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NNODES"]
        env = {}
        for e in env_vars:
            if not os.getenv(e):
                msg = f"Unable to launch torchrun_hpc on SLURM scheduler - {e} not defined"
                raise Exception(msg)
            else:
                env[e] = int(os.getenv(e))

        world_size = env["SLURM_NTASKS"]
        rank = env["SLURM_PROCID"]
        local_rank = env["SLURM_LOCALID"]
        nodes_per_job = env["SLURM_NNODES"]
        local_world_size = world_size // nodes_per_job
        # local_world_size = env['SLURM_TASKS_PER_NODE']
        return (world_size, rank, local_world_size, local_rank)

    # Instance method (not a classmethod): it reads the per-instance
    # rendezvous port so all env entries of one launch agree.
    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        # No RANK entry: this list becomes ``export`` lines in the generated
        # script, which for a --bg submission is the *batch* script running
        # once at allocation scope, so ``export RANK=${SLURM_PROCID}`` froze
        # the batch step's 0 into every task. On the ephemeral CLI path it is
        # expanded on the launch host, where SLURM_PROCID is unset at all.
        # The trampoline publishes RANK from the rank it already computes.
        env_list = []
        if protocol.lower() == "tcp":
            env_list.append(
                (
                    "TORCHRUN_HPC_MASTER_ADDR",
                    "`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`",
                )
            )
            env_list.append(
                ("TORCHRUN_HPC_MASTER_PORT", str(self.rendezvous_port()))
            )
            return env_list
        elif protocol.lower() == "mpi":
            # To use MPI, pass `init_method="mpi://"` - no special work here.
            return env_list
        else:
            msg = f"Unsupported rendezvous protocol {protocol} for scheduler {type(self).__name__}"
            raise Exception(msg)
