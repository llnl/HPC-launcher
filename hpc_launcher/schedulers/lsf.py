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
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union
from io import StringIO
import os
import re
import shlex

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems import System

from hpc_launcher.schedulers.scheduler import Scheduler
from hpc_launcher.schedulers import parse_env_list


@dataclass
class LSFScheduler(Scheduler):

    @classmethod
    def in_allocation(cls) -> bool:
        # LSB_HOSTS is exported inside an lalloc / bsub -Is shell, where the
        # blocking launch command is jsrun rather than bsub.
        return os.getenv("LSB_HOSTS") is not None

    def build_scheduler_specific_arguments(
        self, system: "System", blocking: bool = True
    ):
        # Number of Nodes
        self.run_only_args["--nrs"] = f"{self.nodes}"
        # bsub-only flag (submit-only): jsrun has no notion of "-nnodes", it
        # only ever sees "--nrs" (above). Keep this out of common_launch_args
        # so it doesn't leak into the internal jsrun run command written into
        # the batch script.
        self.submit_only_args["-nnodes"] = f"{self.nodes}"

        # bsub-only flag: not a jsrun option.
        self.submit_only_args["--shared-launch"] = None

        # jsrun options (do we need to guard this with something like if os.getenv("LSB_HOSTS"):
        self.run_only_args["--rs_per_host"] = "1"
        self.run_only_args["--tasks_per_rs"] = f"{self.procs_per_node}"
        self.run_only_args["--launch_distribution"] = "packed"
        self.run_only_args["--cpu_per_rs"] = "ALL_CPUS"
        self.run_only_args["--gpu_per_rs"] = "ALL_GPUS"

        if self.out_log_file and not blocking:
            self.submit_only_args["-o"] = f"{self.out_log_file}"
        if self.err_log_file and not blocking:
            self.submit_only_args["-e"] = f"{self.err_log_file}"

        # Configure header with LSF job options
        if self.time_limit:
            minutes = int(round(max(self.time_limit, 0)))
            hours, minutes = divmod(minutes, 60)
            self.submit_only_args["-W"] = f"{hours}:{minutes:02}"
        if self.job_name:
            # Flag and value are now separate dict entries, so the job name
            # flows through as a normal value and is quoted by the
            # scheduler's central serialization (format_submit_arg /
            # build_command_string_and_batch_script) wherever it is emitted.
            # No source-level quoting needed here.
            self.submit_only_args["-J"] = self.job_name
        if self.queue:
            # bsub-only flag: not a jsrun option.
            self.submit_only_args["-q"] = f"{self.queue}"
        if self.account:
            # bsub-only flag: not a jsrun option.
            self.submit_only_args["-G"] = f"{self.account}"
        if self.reservation:
            self.submit_only_args["-U"] = f"{self.reservation}"

        if self.work_dir:
            if blocking and self.launch_command_is_run_command():
                # Inside an existing allocation the blocking launch command is
                # jsrun, not bsub, and jsrun's working-directory option is
                # --chdir. This is the branch the original --chdir assignment
                # was written for; it was mis-guarded on `blocking` alone, so
                # it also reached the `bsub -Is` command, which has no such
                # option.
                self.run_only_args["--chdir"] = f"{self.work_dir}"
            else:
                # bsub spells it -cwd, both as a command-line option on the
                # interactive `bsub -Is` command and as a #BSUB directive in
                # the batch script.
                self.submit_only_args["-cwd"] = f"{self.work_dir}"

        return

    def format_submit_arg(self, k: str, v: Optional[str], quote_value: bool = False) -> list[str]:
        # Both launch paths execute argv lists (or an internal run-command
        # line built from them) without a shell, so a bsub flag and its
        # value must be separate tokens like ``["-nnodes", "2"]`` rather than
        # one token containing a literal space, unlike Slurm's/
        # Flux's GNU-style "--flag=value" long options. Every entry in
        # submit_only_args is a genuine bsub-only flag, so this formatting
        # applies unconditionally here; common_launch_args keeps the base class's
        # default "=" formatting for LSF (it only holds generic/overridden
        # flags, since LSF has none genuinely shared between bsub and jsrun).
        if not v:
            return [k]
        val = shlex.quote(v) if quote_value else v
        return [k, val]

    def format_run_arg(
        self, k: str, v: Union[None, str, list[str]], quote_value: bool = False
    ) -> list[str]:
        # jsrun's long options take "--flag=value" (the base class default),
        # but a repeatable short option -- currently only "-E", which takes one
        # NAME=value per occurrence -- is stored with a *list* value so that
        # several entries can share a single dict key. Emit one flag/value
        # token pair per element; an attached "-E=NAME=value" would make the
        # first '=' part of the value.
        if isinstance(v, list):
            tokens = []
            for item in v:
                tokens += [k, shlex.quote(item) if quote_value else item]
            return tokens
        return self._kv_arg_tokens(k, v, quote_value)

    def batch_script_prefix(self) -> str:
        return "#BSUB"

    def blocking_launch_command(self) -> list[str]:
        if self.in_allocation():
            return ["jsrun"]
        else:
            return ["bsub", "-Is"]

    def nonblocking_launch_command(self) -> list[str]:
        return ["bsub"]

    @staticmethod
    def _parse_env_pairs(entries: list[str]) -> "OrderedDict[str, str]":
        """
        Parse ``NAME=value`` entries back into a mapping, in order. Entries
        without a ``=`` (e.g. the leading ``ALL`` of a bsub ``-env`` list) are
        not variables and are dropped.
        """
        pairs: "OrderedDict[str, str]" = OrderedDict()
        for entry in entries:
            if "=" not in entry:
                continue
            name, value = entry.split("=", 1)
            pairs[name] = value
        return pairs

    def cli_env_arg(self, env_list) -> None:
        # Expand ${VAR} references, merge duplicate keys, and dequote values
        # like the shell-script path would before folding them onto the
        # command line.
        env_vars = self.expand_cli_env(env_list)

        # This method is called up to twice per launch (once for the system's
        # environment variables, once for the passthrough ones), so each call
        # merges into whatever the previous one left behind rather than adding
        # a second flag. Re-parsing the stored value keeps the accumulated
        # state in the argument dict itself, with no extra scheduler field.
        if self.launch_command_is_run_command():
            # jsrun: -E takes a single NAME=value per occurrence and is
            # repeated. The bsub-only args are not emitted on this command
            # (see launch_command), so the environment has to travel with the
            # run args instead.
            merged = self._parse_env_pairs(self.run_only_args.get("-E") or [])
            merged.update(env_vars)
            self.run_only_args["-E"] = [f"{k}={v}" for k, v in merged.items()]
        else:
            # bsub: -env takes one comma-separated list -- "ALL" to inherit
            # the submitting environment, then the additions. Flag and value
            # are separate argv elements; the double quotes usually written
            # around the list belong to the shell, and there is no shell on
            # this path. Joined without spaces so that no argv element carries
            # an embedded space (the separator LSF parses is the comma).
            existing = self.submit_only_args.get("-env") or ""
            merged = self._parse_env_pairs(existing.split(","))
            merged.update(env_vars)
            self.submit_only_args["-env"] = ",".join(
                ["ALL"] + [f"{k}={v}" for k, v in merged.items()]
            )
        return

    def export_hostlist(self) -> str:
        return "export HPC_LAUNCHER_HOSTLIST=$(echo $LSB_HOSTS | tr ' ' '\\n' | sort -u)\n"

    def enable_run_args_on_launch_command(self) -> bool:
        return self.in_allocation()

    def launch_command_is_run_command(self) -> bool:
        """
        Is the *blocking* launch command the run command (jsrun) rather than
        the submit command (bsub)? True exactly when we are already inside an
        allocation (:meth:`in_allocation`), which is the standard Lassen
        workflow of running from an ``lalloc``/``bsub -Is`` shell. Same
        condition :meth:`enable_run_args_on_launch_command` keys off, named
        for what the callers below actually ask about.

        :return: True if a blocking launch runs jsrun directly.
        """
        return self.enable_run_args_on_launch_command()

    def emit_submit_args_on_launch_command(self, blocking: bool) -> bool:
        """
        Inside an existing allocation the blocking launch command is ``jsrun``,
        which shares no options with ``bsub`` -- handing it ``-nnodes``,
        ``--shared-launch``, ``-W``, ``-J``, ``-q`` or ``-G`` is an immediate
        parse error. Emit the submit-only arguments only when the command
        being built really is ``bsub``.

        :param blocking: Whether the launch command waits for the job.
        :return: True if submit-only arguments belong on this command line.
        """
        return not (blocking and self.launch_command_is_run_command())

    def require_parallel_internal_run_command(self, blocking: bool) -> bool:
        # Only a blocking launch from inside an allocation runs jsrun
        # directly; every other path submits a script that must carry it.
        return not (blocking and self.in_allocation())

    def internal_script_run_command(self) -> str:
        return "jsrun "

    @classmethod
    def cancel_command(cls, identifier: str) -> list[str]:
        """Return the LSF command that cancels a job."""
        return ["bkill", identifier]

    def get_job_id(self, output: str) -> Optional[str]:
        # bsub prints e.g. "Job <123> is submitted to queue <pbatch>." on a
        # successful non-blocking submission; return None (per the base
        # class's contract) when the output can't be parsed instead of
        # raising.
        match = re.search(r"Job <(\d+)>", output)
        if match:
            return match.group(1)
        return None

    @classmethod
    def num_nodes_in_allocation(cls) -> Optional[int]:
        if not cls.in_allocation():
            return None
        # LC's LSF prolog exports the node count directly; elsewhere derive
        # it from LSB_HOSTS, which lists each host once per allocated slot
        # (the same dedup export_hostlist does with `sort -u`).
        nodes = os.getenv("LLNL_NUM_COMPUTE_NODES")
        if nodes:
            return int(nodes)
        hosts = set(os.getenv("LSB_HOSTS", "").split())
        return len(hosts) if hosts else None

    @classmethod
    def get_parallel_rank_env_variable(self) -> str:
        return "${OMPI_COMM_WORLD_RANK}"

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        env_vars = [
            "OMPI_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
        ]
        env = {}
        for e in env_vars:
            if not os.getenv(e):
                msg = (
                    f"Unable to launch torchrun_hpc on LSF scheduler - {e} not defined"
                )
                raise Exception(msg)
            else:
                env[e] = int(os.getenv(e))

        world_size = env["OMPI_COMM_WORLD_SIZE"]
        rank = env["OMPI_COMM_WORLD_RANK"]
        local_rank = env["OMPI_COMM_WORLD_LOCAL_RANK"]
        local_world_size = env["OMPI_COMM_WORLD_LOCAL_SIZE"]
        return (world_size, rank, local_world_size, local_rank)

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        # No RANK entry: OMPI_COMM_WORLD_RANK is set by jsrun at task launch
        # and never in a bsub script, so ``export RANK=${OMPI_COMM_WORLD_RANK}``
        # in the generated script published an empty rank to every task. The
        # trampoline publishes RANK per task instead. See SlurmScheduler for
        # the full rationale.
        env_list = []
        if protocol.lower() == "tcp":
            if self.in_allocation():
                # When running under an allocation use the current node as the coordinator
                env_list.append(("TORCHRUN_HPC_MASTER_ADDR", os.getenv("HOSTNAME")))
            else:
                env_list.append(
                    ("TORCHRUN_HPC_MASTER_ADDR", "`jsrun --nrs 1 -r 1 /bin/hostname`")
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
