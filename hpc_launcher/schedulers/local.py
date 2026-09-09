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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import os
import shlex
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems.system import System


@dataclass
class LocalScheduler(Scheduler):
    """
    A class that runs the job without any underlying batch scheduler. Used
    in ``--local`` jobs.

    A local job is the degenerate case of a scheduled one -- a single
    process, started directly, with no submit command in front of it -- not a
    different kind of job. Everything else the base class puts in a launch
    script (the system's environment block, ``PYTHONPATH``,
    ``HPC_LAUNCHER_MAX_GPU_MEM``, the ``--save-hostlist`` block, argument
    quoting) applies here exactly as it does everywhere else, so this class
    *extends* :meth:`Scheduler.launcher_script` by filling in the handful of
    scheduler-shaped hooks below rather than reimplementing it. Every
    guarantee the base class grew was silently missing under ``--local``
    while that method was reimplemented, and ``--local`` is the backend users
    reach for first when something is already going wrong.

    What genuinely differs is confined to those hooks: there is no submit
    command (:meth:`launch_command`), no directive syntax
    (:meth:`batch_script_prefix`), no parallel launcher
    (:meth:`require_parallel_internal_run_command`), no rank election to
    perform (:meth:`script_runs_once_per_task`), and no ``--chdir``-style
    option, so the script performs its own ``cd``
    (:meth:`build_command_string_and_batch_script`).
    """

    def launch_command(self, system: "System", blocking: bool = True, cli_env_only: bool = False) -> list[str]:
        """
        A local job is started directly, so there is no submit command to
        prefix it with.

        This is also why anything the base class would hand to a command
        line has nowhere to go: the launch script
        (:meth:`build_command_string_and_batch_script`) carries it instead,
        or -- when there is no launch script either --
        :meth:`ephemeral_environment`.
        """
        return []

    def ephemeral_environment(self, system: "System") -> Optional[dict[str, str]]:
        """
        Build the child's environment directly, since a local job has neither
        of the channels the base class relies on.

        An ephemeral run writes no launch script, so the ``export`` lines
        :meth:`Scheduler.launcher_script` would emit have nowhere to go; and
        the fallback the base class uses in that case -- putting the block on
        the submit command line via ``cli_env_arg`` -- needs a command line,
        which :meth:`launch_command` does not produce. So an ephemeral
        ``launch --local`` used to run the user's command with none of the
        system's environment block: no ``NCCL_SOCKET_IFNAME``, no
        ``FI_CXI_*``, no ``MIOPEN_*``. Nothing about the run showed it --
        correct output, correct exit code, untuned (or non-functional)
        communication.

        The values are expanded with :meth:`Scheduler.expand_cli_env`, for
        the same reason the CLI channel does: they are authored as shell
        ``export`` right-hand sides, so a ``${LD_LIBRARY_PATH}`` reference or
        a repeated key only means what it says once a shell -- or this
        stand-in for one -- has processed the list in order. They go into a
        copy of the launcher's own environment, matching the script path,
        where the ``export`` lines likewise amend an inherited environment
        rather than replace it.

        :param system: The system to take the environment from.
        :return: The complete environment for the child.
        """
        env = dict(os.environ)
        # One pass over both lists so that a passthrough value referring to a
        # variable the system block just set resolves against it.
        env.update(
            Scheduler.expand_cli_env(
                list(system.environment_variables())
                + list(system.passthrough_environment_variables())
            )
        )
        # Emitted by launcher_script for every other mode; it is read by the
        # torchrun-hpc trampoline to cap per-process GPU memory.
        if system.active_system_params:
            fraction = system.active_system_params.fraction_max_gpu_mem
            if fraction and fraction != 1.0:
                env["HPC_LAUNCHER_MAX_GPU_MEM"] = str(fraction)
        return env

    def batch_script_prefix(self) -> str:
        """
        There is no scheduler to read directive lines, so anything the base
        class writes as a directive can only be a comment here. Emitting it
        as one -- rather than dropping it -- keeps the generated script an
        honest record of the request: the only arguments that reach this path
        locally are ``-x`` overrides, which have no scheduler argv to
        override, and the comment says as much.
        """
        return "# no scheduler, ignored:"

    def require_parallel_internal_run_command(self, blocking: bool) -> bool:
        """
        There is no parallel launcher (``srun``/``flux run``/``jsrun``) to
        place in front of the command: the script runs the command itself.
        """
        return False

    def script_runs_once_per_task(self, blocking: bool) -> bool:
        """
        A local job is exactly one process, so the two cases the base class
        distinguishes -- "the script is the per-task program" and "the script
        runs once for the whole allocation" -- coincide, and the rank-0 guard
        the base emits for the first case has no election left to perform.

        Report the second so the ``--save-hostlist`` write is emitted
        unguarded: this script is already the single writer, and there is no
        scheduler-provided per-task rank variable for a guard to test. (The
        old hand-rolled version guarded on a ``RANK`` it had exported itself
        one line earlier, which is the shell-snapshot pattern that rank
        identity was moved out of the launch scripts to eliminate.)
        """
        return False

    def export_hostlist(self) -> str:
        """The one host of a local job is the host we are running on."""
        return "export HPC_LAUNCHER_HOSTLIST=$(hostname)\n"

    def build_command_string_and_batch_script(
        self,
        system: "System",
        blocking: bool = True,
        cli_env_only: bool = False,
        for_launch_cmd: bool = True,
    ) -> (str, list[str]):
        """
        Build the launch-script header, routing everything into the script.

        The base class uses ``blocking``/``cli_env_only`` to decide *where*
        the environment travels: on a blocking run it hands the passthrough
        variables to :meth:`cli_env_arg`, because for a real scheduler the
        blocking launch command (``srun``, ``flux run``) is the parallel
        launcher and carries them. A local job has no launch command at all
        (:meth:`launch_command` returns ``[]``), so that channel does not
        exist and the answer is always "write it into the script" -- which is
        precisely the branch the base class takes for a non-blocking,
        non-``cli_env_only`` call. Ask it for that branch instead of
        reimplementing the method; skipping this call entirely is what used
        to drop the ``-x`` override pass (applied here) under ``--local``.

        :param system: The system to use.
        :param blocking: Ignored -- see above; a local job answers the
                         question this parameter asks the same way either
                         way.
        :param cli_env_only: Ignored, for the same reason: there is no
                             command line to put the environment on.
        :param for_launch_cmd: Passed through unchanged.
        :return: A tuple of (shell script header as a string, list of
                 command-line arguments).
        """
        (header, cmd_args) = super().build_command_string_and_batch_script(
            system, blocking=False, cli_env_only=False, for_launch_cmd=for_launch_cmd
        )

        if self.work_dir:
            # No scheduler is available to place the job in its working
            # directory (there is no --chdir/-D to set), so the script does
            # it. The working directory can carry a user-controlled job name
            # (it is embedded in an auto-generated folder name), so quote it
            # before it is interpreted by /bin/sh in the cd.
            header += f"\ncd {shlex.quote(os.path.abspath(self.work_dir))}\n"

        return (header, cmd_args)

    def get_job_id(self, output: str) -> Optional[str]:
        return None

    @classmethod
    def cancel_command(cls, identifier: str) -> list[str]:
        """Return the command that terminates a local launch by PID."""
        if not identifier.isdecimal():
            raise ValueError("a local cancellation identifier must be a process ID")
        return ["kill", "-TERM", identifier]

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        """
        A local job is always exactly one process: ``--local`` does not spawn
        the ``-N``/``-n``/``-g`` job size it is given, so this is the true
        configuration rather than a misreport of a larger one.
        ``common_args.validate_scheduler_arguments`` warns when a larger size
        was requested.
        """
        return 1, 0, 1, 0

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        env_list = []
        if protocol.lower() == "tcp":
            env_list.append(("TORCHRUN_HPC_MASTER_ADDR", "localhost"))
            env_list.append(
                ("TORCHRUN_HPC_MASTER_PORT", str(self.rendezvous_port()))
            )
            return env_list
        else:
            msg = f"Unsupported rendezvous protocol {protocol} for scheduler {type(self).__name__}"
            raise Exception(msg)
