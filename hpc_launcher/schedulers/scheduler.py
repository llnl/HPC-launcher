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
from collections import OrderedDict, ChainMap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from io import StringIO
import os
import re
import sys
import time
import tempfile
import subprocess
import shlex
import shutil
import socket
import uuid
from hpc_launcher.cli.console_pipe import run_process_with_live_output
from hpc_launcher.schedulers import parse_env_list

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rendezvous port selection
# ---------------------------------------------------------------------------

# Range for the UUID-hash fallback used when an ephemeral bind is not
# possible (e.g. a sandbox that forbids ``bind``). High, unprivileged ports.
_RENDEZVOUS_PORT_FALLBACK_MIN = 20000
_RENDEZVOUS_PORT_FALLBACK_MAX = 65000


def pick_rendezvous_port() -> int:
    """
    Choose a TCP port for the distributed rendezvous, once per launch.

    Ask the OS for a free ephemeral port by binding a throwaway socket to
    port 0, reading back the port it assigned, and closing it. Because each
    launch asks independently, two concurrent jobs almost never receive the
    same port -- which is what fixes the cross-job rendezvous-store
    collisions of the old hardcoded ``23456``.

    On any failure (for instance a sandbox that forbids ``bind``), fall back
    to hashing a fresh UUID into a high, unprivileged port range so that two
    launches still almost certainly differ.

    Known limitation (accepted in the plan): the port is chosen on the
    *launch* host, so a port free here may already be in use on the rank-0
    compute node. This narrows -- but does not fully eliminate -- the
    collision window; the previous fixed port collided for *every* pair of
    coincident jobs.

    :return: A TCP port number in ``[1024, 65535]``.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]
    except OSError:
        span = _RENDEZVOUS_PORT_FALLBACK_MAX - _RENDEZVOUS_PORT_FALLBACK_MIN + 1
        return _RENDEZVOUS_PORT_FALLBACK_MIN + (uuid.uuid4().int % span)


# ---------------------------------------------------------------------------
# Ephemeral CLI environment expansion
# ---------------------------------------------------------------------------

# A ``$NAME`` or ``${NAME}`` parameter reference (POSIX variable name).
_ENV_VAR_REF = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}"
    r"|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)


def _expand_env_refs(text: str, lookup) -> str:
    """
    Expand ``$NAME`` / ``${NAME}`` references in ``text`` against
    ``lookup``. Unknown names expand to the empty string, as ``sh`` does.
    """
    def _replace(match: "re.Match") -> str:
        name = match.group("braced") or match.group("bare")
        return lookup.get(name, "")

    return _ENV_VAR_REF.sub(_replace, text)


def shell_expand_assignment(value: str, lookup) -> str:
    """
    Emulate what a POSIX shell produces for the right-hand side of an
    ``export NAME=<value>`` statement: parameter expansion followed by quote
    removal, *without* invoking a shell. Field splitting and pathname
    expansion do not apply to an assignment's RHS, so the result is always a
    single value -- exactly what a single ``--env=NAME=...`` argv element
    needs.

    - Text inside single quotes is literal (no ``$`` expansion); the quotes
      are removed.
    - Text inside double quotes has ``$NAME`` / ``${NAME}`` expanded; the
      quotes are removed.
    - Unquoted text has ``$NAME`` / ``${NAME}`` expanded.

    Simplification (documented): backslash escaping is not interpreted --
    the env values this handles (paths, plugin names, ``${VAR}``-built
    library paths) never rely on it. This is a deliberately small emulation,
    not a full shell word-expansion engine.

    :param value: The raw value as authored for the shell-script path.
    :param lookup: A mapping consulted for parameter expansion (typically
                   the accumulated overlay chained over ``os.environ``).
    :return: The expanded, dequoted value.
    """
    out = []
    i, n = 0, len(value)
    while i < n:
        c = value[i]
        if c == "'":
            j = value.find("'", i + 1)
            if j == -1:  # unbalanced quote: take the remainder literally
                out.append(value[i + 1:])
                break
            out.append(value[i + 1:j])
            i = j + 1
        elif c == '"':
            j = value.find('"', i + 1)
            if j == -1:  # unbalanced quote: expand the remainder
                out.append(_expand_env_refs(value[i + 1:], lookup))
                break
            out.append(_expand_env_refs(value[i + 1:j], lookup))
            i = j + 1
        else:
            j = i
            while j < n and value[j] not in "'\"":
                j += 1
            out.append(_expand_env_refs(value[i:j], lookup))
            i = j
    return "".join(out)

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems.system import System


@dataclass
class LaunchResult:
    """
    The result of a :meth:`Scheduler.launch` call.

    :ivar job_id: The scheduler job ID for a non-blocking (background)
                  submission, or ``None`` when there is no job ID (blocking
                  runs, interactive runs, dry/setup-only runs, or a failed
                  submission).
    :ivar returncode: The exit status the launcher should propagate. This is
                      the child's exit code for blocking/interactive runs,
                      ``0`` for setup-only/dry-run, ``None`` for a successful
                      non-blocking submission whose job is still running, and
                      non-zero when a submission fails.
    """

    job_id: Optional[str] = None
    returncode: Optional[int] = None


@dataclass
class Scheduler:
    """
    An instance of a batch job scheduler that can launch jobs on a given
    system. Produces command line arguments and scripts to support job
    launching, and provides functionality to interactively or asynchronously
    launch a job.
    """

    # Number of nodes to use
    nodes: int
    # Processes per node
    procs_per_node: int
    # GPUs per Process (or task) if any
    gpus_per_proc: int
    # CPUs per task, if explicitly requested (scheduler default otherwise).
    # Inside an allocation this also gives a job step an exact CPU footprint,
    # letting several steps run concurrently without overlapping.
    cpus_per_task: Optional[int] = None
    # Request exclusive access to the resources
    exclusive: Optional[bool] = None
    # Job name
    job_name: Optional[str] = None
    # Working directory (by default, uses current working directory)
    work_dir: Optional[str] = None
    # File for logging output stream (stdout)
    out_log_file: Optional[str] = None
    # File for logging error stream (stderr)
    err_log_file: Optional[str] = None
    # Time limit (in minutes), default is no limit
    time_limit: Optional[int] = None
    # The partition or queue to use with the scheduler
    queue: Optional[str] = None
    # The account to use for the scheduler
    account: Optional[str] = None
    # The reservation to use for the scheduler
    reservation: Optional[str] = None
    # Dependency str
    dependency: Optional[str] = None
    # Hijack preload commands into a scheduler
    ld_preloads: Optional[list[str]] = None
    # Capture the original command so that it can be added to the launch script
    command_line: Optional[list[str]] = None

    # Command line flags given to a batch or interactive submit command
    submit_only_args: OrderedDict = field(default_factory=OrderedDict)
    # Commands given to active run command
    run_only_args: OrderedDict = field(default_factory=OrderedDict)
    # Flags given to both submit and run commands
    common_launch_args: OrderedDict = field(default_factory=OrderedDict)

    # CLI flags for override
    override_launch_args: Optional[dict] = None

    # Rendezvous TCP port for this launch. Chosen once, lazily, and cached so
    # every env entry of a single launch agrees while two launches differ.
    # Not a constructor argument.
    _rendezvous_port: Optional[int] = field(
        default=None, init=False, repr=False, compare=False
    )

    def build_scheduler_specific_arguments(
            self, system: "System", blocking: bool = True
    ):
        return NotImplementedError

    @staticmethod
    def _kv_arg_tokens(k: str, v: Optional[str], quote_value: bool = False) -> list[str]:
        """
        Default flag/value formatting: a bare flag (``k``) when ``v`` is
        falsy, otherwise a single GNU-style ``k=v`` token (e.g. Slurm's
        ``--nodes=2``, Flux's ``--env=FOO``).

        :param k: The flag.
        :param v: The flag's value, or a falsy value (``None``/``""``) for a
                  bare flag.
        :param quote_value: If True, shell-quote the value before joining
                             (used when the returned token(s) will be written
                             into a batch-script line that a shell later
                             parses; not used for argv passed directly to
                             ``exec``, which needs no quoting).
        :return: A list containing the single formatted token.
        """
        if not v:
            return [k]
        val = shlex.quote(v) if quote_value else v
        return [f"{k}={val}"]

    def format_submit_arg(self, k: str, v: Optional[str], quote_value: bool = False) -> list[str]:
        """
        Formats a single (flag, value) pair from ``submit_only_args`` into
        the token(s) used on the submit command's argv or a batch-script
        directive line. Default: GNU-style ``k=v`` as one token. Override
        for schedulers whose submit flags don't accept ``=`` (e.g. LSF's
        ``bsub`` short options, which need ``-nnodes`` and ``2`` as two
        separate tokens).

        :param k: The flag.
        :param v: The flag's value, or a falsy value for a bare flag.
        :param quote_value: See :meth:`_kv_arg_tokens`.
        :return: A list of one or more tokens representing this flag/value pair.
        """
        return self._kv_arg_tokens(k, v, quote_value)

    def format_common_arg(self, k: str, v: Optional[str], quote_value: bool = False) -> list[str]:
        """As :meth:`format_submit_arg`, for ``common_launch_args``."""
        return self._kv_arg_tokens(k, v, quote_value)

    def format_run_arg(self, k: str, v: Optional[str], quote_value: bool = False) -> list[str]:
        """As :meth:`format_submit_arg`, for ``run_only_args``."""
        return self._kv_arg_tokens(k, v, quote_value)

    def build_command_string_and_batch_script(
            self, system: "System", blocking: bool = True, cli_env_only: bool = False,
            for_launch_cmd: bool = True,
    ) -> (str, list[str]):
        """
        Returns the strings used for a launch command as well as a batch script
        full launcher script, which can be saved as a batch
        script, for the given system and launcher configuration.
        This script usually performs node/resource allocation and manages I/O.

        :param system: The system to use.
        :param blocking: Is the job interactive of blocking
        :param cli_env_only: Append environment variables to CLI not a launch script
        :param for_launch_cmd:  Some args should not be in both the header and launch cmnd. Ex: flux --dependency=afterany:XXX
        :return: A tuple of (shell script as a string, list of command-line arguments).
        """
        env_vars = system.environment_variables()
        passthrough_env_vars = system.passthrough_environment_variables()

        header = StringIO()
        header.write("#!/bin/sh\n")
        cmd_args = []

        # Enable the system to apply some customization to the scheduler
        # instance. This has to happen *before* the arguments are built:
        # ``customize_scheduler`` is the only producer of scheduler fields such
        # as ``ld_preloads`` (set from ``LBANN_USE_THIS_RCCL``), and
        # ``build_scheduler_specific_arguments`` is what turns them into
        # ``--env=LD_PRELOAD`` / ``--export=ALL,LD_PRELOAD``. With the calls the
        # other way around the field was still None when it was read, so a
        # blocking launch silently dropped the preload -- while ``--bg``
        # appeared to work, because ``launcher_script`` makes a second pass over
        # the same instance. The customizations that are *dict* edits
        # (``common_launch_args`` and friends) always survived either order,
        # since ``launch_command`` consumes those dicts itself.
        system.customize_scheduler(self)

        self.build_scheduler_specific_arguments(system, blocking)

        if self.override_launch_args:
            for k,v in self.override_launch_args.items():
                arg_overridden = False
                remove_arg = False
                if "~" in k:
                    k = k.replace("~", "")
                    remove_arg = True
                if k in self.common_launch_args:
                    if remove_arg:
                        self.common_launch_args.pop(k, None)
                    else:
                        tmp = self.common_launch_args[k]
                        self.common_launch_args[k] = v
                        arg_overridden = True

                if k in self.run_only_args:
                    if remove_arg:
                        self.run_only_args.pop(k, None)
                    else:
                        tmp = self.run_only_args[k]
                        self.run_only_args[k] = v
                        arg_overridden = True

                if k in self.submit_only_args:
                    if remove_arg:
                        self.submit_only_args.pop(k, None)
                    else:
                        tmp = self.submit_only_args[k]
                        self.submit_only_args[k] = v
                        arg_overridden = True

                if not arg_overridden and not remove_arg:
                    self.common_launch_args[k] = v

        if not blocking: # Only add batch script header items on non-blocking calls
            # These header lines are only ever emitted into the shell script
            # produced by ``launcher_script`` (in ``launch_command`` the header
            # is discarded). The scheduler parses them and, being written to a
            # file executed by /bin/sh, they are exposed to the shell, so quote
            # every interpolated value to keep user-controlled data (e.g. a job
            # name) a single inert token rather than shell syntax.
            prefix = self.batch_script_prefix()
            for k,v in self.submit_only_args.items():
                if not for_launch_cmd and k == '--dependency':
                    continue
                header.write(
                    f"{prefix} " + " ".join(self.format_submit_arg(k, v, quote_value=True)) + "\n"
                )
            for k,v in self.common_launch_args.items():
                if not for_launch_cmd and k == '--dependency':
                    continue
                header.write(
                    f"{prefix} " + " ".join(self.format_common_arg(k, v, quote_value=True)) + "\n"
                )

        # ``cli_env_only`` means the caller will not be writing a launch script
        # we control -- an ephemeral run, or a user-supplied ``--batch-script``
        # that must be copied verbatim -- so the scheduler command line is the
        # only channel the environment has. That is true regardless of whether
        # the job blocks: gating this on ``blocking`` as well meant a
        # non-blocking immutable-script submission (``--batch-script`` with
        # ``--bg``) fell through to the ``else`` branch, whose ``header`` buffer
        # is discarded by ``launch_command``, silently dropping the entire
        # launcher-injected tuning block.
        if len(env_vars):
            if cli_env_only:
                self.cli_env_arg(env_vars)
            else:
                for e in env_vars:
                    header.write(parse_env_list(*e))

        # Same hole for passthrough variables. ``blocking`` is kept as an
        # additional trigger because a blocking run has always delivered these
        # on the launch command line (where they work equally well) and jobs
        # rely on that; ``cli_env_only`` closes the non-blocking immutable case.
        if len(passthrough_env_vars):
            if blocking or cli_env_only:
                self.cli_env_arg(passthrough_env_vars)
            else:
                for k, v in passthrough_env_vars:
                    header.write(f"export {k}={v}\n")

        return (header.getvalue(), cmd_args)

    def batch_script_prefix(self) -> str:
        """
        Returns scheduler specific prefix for batch scripts
        :return: scheduler specific prefix for batch scripts
        """
        raise NotImplementedError

    def blocking_launch_command(self) -> list[str]:
        """
        Returns scheduler specific command for interactive (blocking) jobs
        :return: scheduler specific command for interactive (blocking) jobs
        """
        raise NotImplementedError

    def nonblocking_launch_command(self) -> list[str]:
        """
        Returns scheduler specific command for non-blocking batch jobs
        :return: scheduler specific command for non-blocking batch jobs
        """
        raise NotImplementedError

    def rendezvous_port(self) -> int:
        """
        The rendezvous TCP port for this launch, chosen once (lazily) and
        cached on the instance so that every environment entry generated for
        a single launch agrees on the port, while two separate launches (two
        scheduler instances) get different ports.

        ``TORCHRUN_HPC_MASTER_PORT`` remains the documented user override:
        the trampoline honors it when set, so exporting it explicitly still
        pins the port.

        :return: The chosen TCP port.
        """
        if self._rendezvous_port is None:
            self._rendezvous_port = pick_rendezvous_port()
        return self._rendezvous_port

    @staticmethod
    def expand_cli_env(env_list: list[tuple]) -> "OrderedDict[str, str]":
        """
        Collapse an ordered environment list into the final
        ``{name: value}`` mapping a POSIX shell would produce after running
        the equivalent sequence of ``export`` statements -- but without a
        shell.

        On the ephemeral blocking path env vars are moved onto the scheduler
        CLI (e.g. flux ``--env=``), where no shell interprets them. Passing
        the raw script-authored values there would (a) leave literal quotes
        in the value, (b) collapse duplicate keys, silently dropping one of
        two ``LD_LIBRARY_PATH`` entries, and (c) never expand ``${VAR}``.

        This processes the list IN ORDER, keeping an overlay of what has been
        assigned so far on top of ``os.environ``. Each value is expanded and
        dequoted against ``os.environ`` plus the overlay-so-far, so a later
        ``LD_LIBRARY_PATH=B:${LD_LIBRARY_PATH}`` naturally incorporates an
        earlier ``LD_LIBRARY_PATH=A:...`` exactly as the shell would (yielding
        ``B:A:<original>``). The final overlay has one fully expanded entry
        per key, in first-seen order. The values are verbatim argv elements
        (destined for ``exec``), so they are NOT shell-quoted.

        :param env_list: The env entries, each a ``(name, value[, comment])``
                         tuple; bare comment/no-op entries (length < 2) are
                         skipped.
        :return: The expanded, deduplicated mapping.
        """
        overlay: "OrderedDict[str, str]" = OrderedDict()
        for e in env_list:
            if len(e) < 2:
                continue
            k, v = e[0], e[1]
            # overlay takes precedence over os.environ for expansion, so a
            # later entry sees the values assigned by earlier entries.
            lookup = ChainMap(overlay, os.environ)
            overlay[k] = shell_expand_assignment(str(v), lookup)
        return overlay

    def cli_env_arg(self, env_list: list[tuple[str,str]]) -> None:
        """
        How should environment variables be passed to launched command.
        Append them the to the submit_only_args
        :return: None
        """
        raise NotImplementedError

    def ephemeral_environment(self, system: "System") -> Optional[dict[str, str]]:
        """
        The complete environment to start an ephemeral (no launch folder) job
        in, or None to hand the job the launcher's own environment unchanged.

        The launcher has exactly three channels for the system's tuned
        environment block, and an ephemeral run rules out one of them: there
        is no launch script for :meth:`launcher_script` to write ``export``
        lines into. Every scheduler that has a submit/run command uses the
        second -- ``launch_command`` is built with ``cli_env_only=True``, so
        :meth:`cli_env_arg` puts the block on that command line (Slurm's
        ``--export=``, Flux's ``--env=``) -- and for those, None is correct:
        the environment is already travelling.

        This third channel exists for a scheduler with no launch command at
        all to hang arguments off (``--local``, whose
        :meth:`launch_command` is ``[]``), which was therefore left with no
        channel whatsoever and silently ran ephemeral jobs with the system's
        entire environment block missing.

        :param system: The system to take the environment from.
        :return: A complete environment mapping for the child, or None to
                 inherit the launcher's.
        """
        return None

    def launch_command(
            self, system: "System", blocking: bool = True, cli_env_only: bool = False
    ) -> list[str]:
        """
        Returns the launch command for this scheduler. Returns the
        command prefix before the program to run.

        :param blocking: Whether to launch a command that waits for the
                         command to finish (True), or launch a batch
                         script that immediately returns (False).
        :return: The command prefix as a list of strings (one per argument).
        """
        (header_lines, cmd_args) = self.build_command_string_and_batch_script(
            system, blocking, cli_env_only,
        )

        # Both commands get the submit args
        for k,v in self.common_launch_args.items():
            cmd_args += self.format_common_arg(k, v)
        if self.emit_submit_args_on_launch_command(blocking):
            for k,v in self.submit_only_args.items():
                cmd_args += self.format_submit_arg(k, v)
        if not blocking:
            return self.nonblocking_launch_command() + cmd_args

        # For interactive jobs add the run args (if the scheduler permits it)
        if self.enable_run_args_on_launch_command():
            for k,v in self.run_only_args.items():
                cmd_args += self.format_run_arg(k, v)

        # With no launch directory there is no launch script, so the parallel
        # run command has nowhere else to go and belongs on the command line.
        # Schedulers whose blocking launch command is already the parallel
        # launcher (srun, flux run) report False here and are unaffected.
        if not self.work_dir and self.require_parallel_internal_run_command(blocking):
            cmd_args += self.internal_script_run_command().split()
            for k, v in self.run_only_args.items():
                cmd_args += self.format_run_arg(k, v)

        return self.blocking_launch_command() + cmd_args

    def export_hostlist(self) -> str:
        """
        Returns a shell cmmand to set the hostlist of the job to an environment variable.
        :return: string that exports hostlist to environment variable
        """
        raise NotImplementedError

    def require_parallel_internal_run_command(self, blocking: bool) -> bool:
        """
        Returns scheduler specific command for use in a batch or interactive submitted script
        :return: bool indicating if a special run command is required
        """
        if not blocking:
            return True
        else:
            return False

    def script_runs_once_per_task(self, blocking: bool) -> bool:
        """
        Does the generated launch script execute once per task, or once for
        the whole allocation?

        These are the only two possibilities, and they are the complement of
        :meth:`require_parallel_internal_run_command`: if the script has to
        carry the parallel run command (``srun``/``flux run``/``jsrun``) then
        the script itself is what the submit command runs -- once, at
        allocation scope -- and the tasks are forked from inside it.
        Otherwise the launch command *is* the parallel launcher and the script
        is the per-task program.

        Anything in the script that names a rank is only meaningful in the
        second case; at allocation scope the scheduler's per-task variables
        are either unset (Flux, LSF) or a meaningless constant (Slurm's
        one-task batch step reports ``SLURM_PROCID=0``).

        :param blocking: Whether the launch command waits for the job.
        :return: True if the script body runs once per task.
        """
        return not self.require_parallel_internal_run_command(blocking)

    def enable_run_args_on_launch_command(self) -> bool:
        """
        Allow scheduler to explicitly enable or disable appending the runtime
        arguments to the launch command.
        :return: bool indicating if run arguments are appended to launch command
        """
        return True

    def emit_submit_args_on_launch_command(self, blocking: bool) -> bool:
        """
        Should the submit-only arguments be appended to the launch command?

        True for schedulers whose submit and run commands are the same
        program, so that a flag accepted by one is accepted by the other
        (srun, flux run). LSF is the exception: inside an existing allocation
        its blocking launch command is jsrun, which shares no options with
        bsub.

        :return: bool indicating if submit arguments are appended to the
                 launch command
        """
        return True

    def internal_script_run_command(self) -> str:
        """
        Returns scheduler specific command for use in a batch or interactive submitted script
        :return: string scheduler specific command for use in a batch or interactive submitted script
        """
        raise NotImplementedError

    def ephemeral_identity_wrapper(self) -> list[str]:
        """Return an optional argv wrapper that announces a live job ID.

        A blocking ephemeral launch has no generated script in which to put
        scheduler bookkeeping.  Backends that learn their job/step identity
        only after the parallel launcher starts may return a small wrapper
        here.  It is inserted between the scheduler command and the user's
        command; the wrapper must ultimately ``exec \"$@\"`` so the user's
        argv and signal behavior are preserved.

        The default is appropriate for local and schedulers that do not
        expose a runtime identity this way.
        """
        return []

    @classmethod
    def cancel_command(cls, identifier: str) -> list[str]:
        """Return the native command that cancels ``identifier``.

        Scheduler implementations own the syntax for cancelling their jobs
        or job steps.  Keeping it beside the launch-command definitions avoids
        duplicating backend knowledge in command-line front ends.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not define a cancellation command"
        )

    def launcher_script(
        self,
        system: "System",
        command: str,
        args: Optional[list[str]] = None,
        blocking: bool = True,
        save_hostlist: bool = False,
        launch_dir: str = "",
    ) -> str:
        """
        Returns the full launcher script, which can be saved as a batch
        script, for the given system and launcher configuration.
        This script usually performs node/resource allocation and manages I/O.

        :param system: The system to use.
        :param command: The command to launch
        :param args: Optional list of argument for the command to launch
        :param blocking: Launch the comamnd interactively if true, else in a batch job
        :params save_hostlist: Add local scripting to capture the list of hosts the command is launched on
        :params launch_dir: Folder used for running the command
        :return: A shell script as a string.
        """
        script = ""
        # Launch command only use the cmd_args to construct the shell script to be launched
        (header_lines, cmd_args) = self.build_command_string_and_batch_script(
            system, blocking, False, for_launch_cmd=False
        )
        # This ``cmd_args`` list is joined into the internal run command that is
        # written into the shell script below (unlike ``launch_command``'s
        # cmd_args, which are argv elements executed without a shell). Quote
        # every value so it reaches the run command as a single verbatim token
        # instead of being re-parsed by /bin/sh.
        # For batch jobs add any common args to the internal command
        if not blocking:
            for k,v in self.common_launch_args.items():
                cmd_args += self.format_common_arg(k, v, quote_value=True)
        # For jobs that require a parallel internal command add any run args
        if self.require_parallel_internal_run_command(blocking):
            for k,v in self.run_only_args.items():
                cmd_args += self.format_run_arg(k, v, quote_value=True)

        # Configure header and command line with scheduler job options
        script += header_lines
        script += "\n"
        # The job runs from the launch directory, so re-add the directory the
        # user launched from: that is what keeps the command's sibling modules
        # importable (for torchrun-hpc, nothing else adds it at all). Nothing
        # in the launcher ever chdirs -- the working directory is handed to the
        # scheduler as an argument -- so the process cwd here is still the
        # invocation directory. It used to be computed as
        # ``dirname(launch_dir)``, which coincides with the invocation
        # directory only when the launch directory sits directly beneath it: an
        # absolute ``-l /p/lustre1/shared/runs/job1`` instead put that
        # directory's unrelated (and plausibly group-writable) parent ahead of
        # site-packages on every rank's import path.
        invocation_directory = os.getcwd()
        if launch_dir != invocation_directory:
            logger.info(
                f"Callee directory: {invocation_directory} - and {launch_dir}"
            )
            # Quote the literal path so it cannot inject shell syntax, while
            # leaving the trailing ${PYTHONPATH} reference to expand as
            # intended.
            script += f"export PYTHONPATH={shlex.quote(invocation_directory)}:" + "${PYTHONPATH}\n"
        if save_hostlist:
            hostlist_file = os.path.join(launch_dir, "hpc_launcher_hostlist.txt")
            script += self.export_hostlist()
            write_hostlist = (
                "echo ${HPC_LAUNCHER_HOSTLIST} > " + shlex.quote(hostlist_file) + "\n"
            )
            if self.script_runs_once_per_task(blocking):
                # This script *is* the per-task program, so exactly one task
                # may write the file. Test the scheduler's own per-task
                # variable directly: a ``RANK`` snapshot taken by an earlier
                # ``export`` is what stopped working the moment the same block
                # moved to allocation scope.
                script += f'if [ "{self.get_parallel_rank_env_variable()}" = "0" ]; then\n'
                script += "    " + write_hostlist
                script += "fi\n\n"
            else:
                # The script runs once, ahead of the parallel run command, so
                # it is already the single writer -- and no rank variable is
                # set for it to test. The old guard compared Slurm's
                # meaningless batch-step ``SLURM_PROCID=0`` (true by luck) or
                # an unset Flux/LSF variable (so the file was silently never
                # created).
                script += write_hostlist + "\n"

        if system.active_system_params:
            system_params = system.active_system_params
            if system_params.fraction_max_gpu_mem and system_params.fraction_max_gpu_mem != 1.0:
                script += f'export HPC_LAUNCHER_MAX_GPU_MEM={system_params.fraction_max_gpu_mem}\n'

        if self.require_parallel_internal_run_command(blocking):
            script += self.internal_script_run_command()
            script += " ".join(cmd_args)
            script += " "

        script += f"{command}"

        # Quote each command argument so values containing shell metacharacters
        # (spaces, ';', '&', '$(...)', backticks, redirections, ...) survive as
        # a single verbatim token rather than being split or interpreted when
        # the scheduler executes this script under /bin/sh.
        for arg in args:
            script += f" {shlex.quote(arg)}"

        script += "\n"

        return script

    def internal_script(self, system: "System") -> Optional[str]:
        """
        Returns the script that runs on each process within the allocated job.
        This script is optional, and usually sets up additional elements (e.g.,
        environment variables, working directory, profiler) in case external
        variables are cleaned up by the job scheduler.

        :param system: The system to use.
        :return: A shell script as a string, or None if no internal script is
                 required.
        """
        # By default, no internal script is required
        return None

    def get_job_id(self, output: str) -> Optional[str]:
        """
        Parses and returns the job ID from a batch job submission (running in
        the background). Returns ``None`` if parsing cannot be performed.

        :param output: Console outputs of the batch submission.
        :return: A string containing the job ID, or None if the output cannot
                 be parsed.
        """
        return None

    @classmethod
    def in_allocation(cls) -> bool:
        """
        Is this process already running inside an allocation owned by this
        scheduler (e.g. a salloc/sbatch, flux alloc, or lalloc/bsub -Is shell)?

        A blocking launch from inside an allocation runs as a nested job step
        rather than requesting a new allocation, and several argument-building
        decisions key off that. Schedulers that can recognize their allocation
        from the environment override this; the default, which also serves the
        local scheduler, is never inside one.

        :return: True if inside an allocation of this scheduler.
        """
        return False

    @classmethod
    def num_nodes_in_allocation(cls) -> Optional[int]:
        """
        When running inside an allocation of this scheduler (see
        :meth:`in_allocation`), report how many nodes it holds. Each
        scheduler consults only its own environment; the scheduler-agnostic
        ``hpc_launcher.schedulers.num_nodes_in_current_allocation`` asks each
        scheduler in turn.

        :return: Number of nodes in the allocation, or None when not inside
                 an allocation of this scheduler (the default, which also
                 serves the local scheduler).
        """
        return None

    @classmethod
    def get_parallel_rank_env_variable(cls) -> str:
        """
        When running under an allocation, return the environment variable to get the current rank

        :return: environment variable for rank in an allocation
        """
        raise NotImplementedError

    @classmethod
    def get_parallel_configuration(cls) -> tuple[int, int, int, int]:
        """
        Using scheduler environment variables report the parallel configuration
        of the run.

        :return: A tuple of integers in the format
                 (world_size, rank, local_world_size, local_rank)
        """
        raise NotImplementedError

    def dynamically_configure_rendezvous_protocol(self, protocol: str) -> list[str]:
        """
        Configure the rendezvous protocol at runtime for a tool like PyTorch to establish
        distributed communication.

        :param protocol: Field to select which protocol to use for the rendezvous
        :return: An init_method string that conforms to
                 https://pytorch.org/docs/stable/distributed.html.
        """
        raise NotImplementedError

    def setup_rendezvous_protocol(self, protocol: str) -> list[str]:
        """
        Setup a protocol for a tool like PyTorch to use to establish
        distributed communication.

        :param protocol: Field to select which protocol to use for the rendezvous
        :return: A list of strings that are added to the torchrun-hpc launch environment.
        """
        env_list = []
        env_list.append(("TORCHRUN_HPC_SCHEDULER", type(self).__name__))
        env_list.extend(self.dynamically_configure_rendezvous_protocol(protocol))
        if protocol.lower() == "tcp":
            env_list.append(
                (
                    "TORCHRUN_HPC_RDV_PROTOCOL",
                    '"tcp://${TORCHRUN_HPC_MASTER_ADDR}:${TORCHRUN_HPC_MASTER_PORT}"',
                )
            )
        elif protocol.lower() == "mpi":
            env_list.append(("TORCHRUN_HPC_RDV_PROTOCOL", "mpi://"))
        else:
            msg = f"Unsupported rendezvous protocol {protocol}"
            raise Exception(msg)
        return env_list

    def create_launch_folder_name(
        self,
        command: str,
        folder_prefix: str = "launch",
        launch_dir: Optional[str] = None,
    ) -> (str, str):
        """
        Create a folder name for the launcher based on the command.

        :param command: The command line to run.
        :param folder_prefix: Specializable prefix for the folder name
        :param launch_dir: [Optional] Name of launch directory, None, or "." for <cwd>
        :return: A tuple of strings with the the command as a possible folder name, and the folder name.
        """
        # Remove spaces and semi-colons from the command sequence
        # command_as_folder_name = "batch_script"
        # if command:
        command_as_folder_name = (
            os.path.basename(command).replace(" ", "_").replace(";", "-")
        )

        if launch_dir == ".":
            folder_name = os.getcwd()
        elif launch_dir == "":
            # Create a folder for the output and error logs
            # Timestamp is of the format YYYY-MM-DD_HHhMMmSSs_UUID
            short_uuid = uuid.uuid4().hex[:8]
            folder_name = f'{folder_prefix}-{self.job_name or command_as_folder_name}_{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}_{short_uuid}'
        else:
            folder_name = launch_dir

        return (command_as_folder_name, folder_name)

    def create_launch_folder(
        self,
        folder_name: str,
        blocking: bool = True,
        script_file: Optional[str] = None,
        dry_run: bool = False,
    ) -> (str, str):
        """
        Create a folder and associated launch script if approrpiate.

        :param folder_name: The name of the folder for containing all of the launch artifacts.
        :param blocking: If True, the job should run from the launch folder.
        :param script_file: If given, saves the output script to this file.
        :param dry_run: If True, only sets up the job and does not launch it.
        :return: The filename for the launch script as a string.
        """

        should_make_folder = folder_name != None
        copy_script_to_filename = False

        # Create a temporary file or a script file, if given
        if script_file is not None:
            # Destination the script would be written/copied to inside the folder.
            dest = os.path.abspath(os.path.join(folder_name, os.path.basename(script_file)))
            # Only treat ``script_file`` as an *input* batch script to copy in
            # when it exists AND is a different file from the destination. A
            # re-run with the same ``-l``/``-o`` resolves source and destination
            # to the same path; copying it onto itself raises SameFileError, so
            # in that case we simply regenerate the script in place.
            is_input_batch_script = os.path.exists(script_file) and not (
                os.path.exists(dest) and os.path.samefile(script_file, dest)
            )
            if is_input_batch_script:
                # A batch file was provided
                filename = dest
                # Plan to copy provided file into the launch directory
                copy_script_to_filename = True
            else:
                # Create a new batch file with the provided name
                if os.path.dirname(script_file) and not dry_run:
                    msg = f"Unsupported script_file {script_file} - cannot be a full path"
                    raise Exception(msg)
                else:
                    filename = os.path.abspath(os.path.join(folder_name, script_file))

            # Warn if this file exists
            if os.path.exists(filename):
                logger.warning(f"Overwriting existing file {filename}")

        else:
            filename = os.path.abspath(os.path.join(folder_name, "launch.sh"))

        if self.out_log_file is None:
            self.out_log_file = os.path.abspath(os.path.join(folder_name, "out.log"))
        else:
            if not os.path.isabs(self.out_log_file):
                log_file = os.path.abspath(os.path.join(folder_name, self.out_log_file))
                self.out_log_file = log_file

        if self.err_log_file is None:
            self.err_log_file = os.path.abspath(os.path.join(folder_name, "err.log"))
        else:
            if not os.path.isabs(self.err_log_file):
                log_file = os.path.abspath(os.path.join(folder_name, self.err_log_file))
                self.err_log_file = log_file

        stub_file = ""
        if should_make_folder and not dry_run:
            os.makedirs(folder_name, exist_ok=True)
            if copy_script_to_filename:
                shutil.copy(script_file, filename)

        return filename

    def launch(
        self,
        system: "System",
        folder_name: Optional[str],
        filename: Optional[str],
        command: str,
        args: Optional[list[str]] = None,
        override_launch_args: Optional[dict] = None,
        blocking: bool = True,
        setup_only: bool = False,
        color_stderr: bool = False,
        dry_run: bool = False,
        save_hostlist: bool = False,
        immutable_launch_script: bool = False,
    ) -> "LaunchResult":
        """
        Launches the given command and arguments uaing this launcher.

        :param system: The system to use for launching the job.
        :param folder_name: The name of the folder for containing all of the launch artifacts.
        :param filename: The filename for the launch script
        :param command: The command line to run.
        :param args: The arguments to use for the command.
        :param blocking: If True, blocks until the job is complete
                         and redirects/duplicates outputs to the terminal.
        :param setup_only: If True, only sets up the job and does not launch it.
        :param color_stderr: If True, colors stderr terminal outputs in red.
        :param run_from_launch_dir: If True, runs the command from the launch directory.
        :params save_hostlist: Add local scripting to capture the list of hosts the command is launched on
        :params immutable_launch_script: It True, do not modify the script and put any system env arguments on the CLI command
        :return: A :class:`LaunchResult` carrying the job ID (if any) and the
                 exit status the caller should propagate.
        """

        self.override_launch_args = override_launch_args

        # If the command is run from a directory
        if folder_name:
            # Change the working directory to the launch folder
            if not self.work_dir:
                self.work_dir = os.path.abspath(folder_name)
            # There is no need to use the following at the moment:
            # elif shutil.which(command):
            #     command = os.path.abspath(shutil.which(command))

        # If the command exists as a file, use its absolute path
        if command and os.path.isfile(command):
            command = os.path.abspath(command)

        # Warn about relative-path arguments that will not resolve once the job
        # changes into the launch directory. The job runs from the
        # launch folder, but command arguments are emitted verbatim, so a
        # relative path that exists in the invocation directory but not under the
        # launch directory would silently fail to open. Document behavior in
        # launch_cli.md; here we surface a clear warning.
        if folder_name and args:
            launch_dir_abs = os.path.abspath(folder_name)
            cwd = os.getcwd()
            if launch_dir_abs != cwd:
                for arg in args:
                    if not arg or os.path.isabs(arg):
                        continue
                    in_cwd = os.path.exists(os.path.join(cwd, arg))
                    in_launch_dir = os.path.exists(os.path.join(launch_dir_abs, arg))
                    if in_cwd and not in_launch_dir:
                        logger.warning(
                            f"Argument '{arg}' is a relative path that exists in "
                            f"the current directory but not in the launch "
                            f"directory '{launch_dir_abs}'. The job runs from the "
                            f"launch directory, so this path will not resolve; "
                            f"pass an absolute path (or use '-l .') instead."
                        )

        use_launch_folder = folder_name or filename
        cmd = self.launch_command(system, blocking, not use_launch_folder or immutable_launch_script)

        if not use_launch_folder: # Launch job and trace outputs live
            # Run interactive script
            full_cmdline = cmd + self.ephemeral_identity_wrapper() + [command]

            for arg in args:
                full_cmdline += [arg]

            if setup_only:
                logger.warning(f'To launch, run: {" ".join(full_cmdline)}')
                return LaunchResult(job_id=None, returncode=0)

            logger.info(f'Launching {" ".join(full_cmdline)}')

            if not dry_run:
                # Same live tee-ing as the launch-folder blocking branch
                # below, minus the files: an ephemeral run is defined by
                # creating none, so the console is the only destination.
                # This branch used to call
                # ``subprocess.run(..., capture_output=True)`` and print the
                # result afterwards, which (a) showed the user nothing until
                # the job exited -- a blank terminal for the length of a
                # training run, with every byte held in the launcher's RSS,
                # and the two streams no longer interleaved in the order they
                # were written -- and (b) skipped ``console_pipe``'s
                # ``start_new_session`` and signal forwarding, so a SIGTERM to
                # the launcher killed the launcher alone and reparented the
                # still-running job to PID 1.
                returncode = run_process_with_live_output(
                    full_cmdline,
                    color_stderr=color_stderr,
                    env=self.ephemeral_environment(system),
                )
                # Only the exit status decides success: plenty of successful
                # programs (and schedulers) log to stderr.
                if returncode:
                    logger.error(
                        f"Interactive scheduler exited with error code {returncode}"
                    )
                return LaunchResult(job_id=None, returncode=returncode)
            return LaunchResult(job_id=None, returncode=0)
        else:
            full_cmdline = cmd + [filename]
            logger.info(f"Script filename: {filename}")
            if not dry_run and not immutable_launch_script:
                with open(filename, "w") as fp:
                    fp.write(
                        self.launcher_script(system, command, args, blocking, save_hostlist, os.path.dirname(filename))
                    )

                    fp.write(f"\n# Launch command: " + " ".join(full_cmdline) + "\n")
                    if self.command_line:
                        fp.write(
                            f"# User command invoked: " + " ".join(self.command_line) + "\n"
                        )
                os.chmod(filename, 0o700)

            if setup_only:
                logger.warning(f'To launch, run: {" ".join(full_cmdline)}')
                return LaunchResult(job_id=None, returncode=0)

            logger.info(f'Launching {" ".join(full_cmdline)}')

            if dry_run:
                return LaunchResult(job_id=None, returncode=0)

            if blocking:  # Launch job and trace outputs live
               with open(self.out_log_file, "wb") as out_file:
                   with open(self.err_log_file, "wb") as err_file:

                       returncode = run_process_with_live_output(
                           full_cmdline,
                           out_file=out_file,
                           err_file=err_file,
                           color_stderr=color_stderr,
                       )
               # In this mode, there is no job ID; propagate the child's status.
               return LaunchResult(job_id=None, returncode=returncode)
            else:
                # Run batch script and get job ID
                process = subprocess.run(full_cmdline, capture_output=True)
                # Always show the user what the submit command said on stderr,
                # but never treat it as a failure signal: schedulers routinely
                # warn on an otherwise successful submission (e.g. sbatch's
                # "can't honor --ntasks-per-node ... Ignoring" notice, or a site
                # job_submit plugin's slurm.log_user() message). Only the exit
                # status decides -- otherwise a queued job's ID is thrown away
                # and the launcher reports "error code 0".
                sys.stderr.buffer.write(process.stderr)
                if process.returncode:
                    logging.error(
                        f"Batch scheduler exited with error code {process.returncode}"
                    )
                    # A genuine failure often explains itself on stdout; do not
                    # swallow it.
                    sys.stdout.buffer.write(process.stdout)
                    return LaunchResult(job_id=None, returncode=process.returncode)
                # Successful non-blocking submission: the job is still running,
                # so there is no exit code to report yet.
                return LaunchResult(
                    job_id=self.get_job_id(process.stdout.decode()), returncode=None
                )
