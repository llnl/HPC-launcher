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
"""
Regression tests for the *ephemeral* blocking launch path -- a ``launch``
with no ``-l``/``-o``, which is the documented default
(``launch_cli.md``: "If not set, it will ... run the command without
creating any files if the job is blocking").

That path used to run the job with a bare
``subprocess.run(..., capture_output=True)`` while the ``-l`` path went
through ``console_pipe.run_process_with_live_output``. One root cause, three
user-visible consequences, one per group of tests below:

- (buffering) nothing reached the terminal until the job exited. A
  multi-hour training run showed a blank terminal for hours, stdout/stderr
  interleaving was lost, and every byte of output accumulated in the
  launcher's RSS.
- (orphaning) the path skipped ``console_pipe``'s ``start_new_session``
  and signal-forwarding machinery, so a SIGTERM to the launcher killed the
  launcher alone and left the scheduler child reparented to PID 1, still
  running and now unkillable by job-cleanup tooling that only knows the
  launcher's pid.
- (adjacent) ``Scheduler.launch`` never calls ``launcher_script`` on this
  path, so for ``--local`` -- whose ``launch_command()`` is ``[]`` and which
  therefore has no command line to hang ``--env=``-style arguments off --
  the system's entire environment block (``NCCL_*``, ``FI_CXI_*``,
  ``MIOPEN_*``, ...) had no channel at all and was silently dropped.

``--local`` is used throughout: it is the one backend that genuinely runs in
any test environment, and it is also the backend with the most to lose
(no launch command means no fallback channel).
"""
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from hpc_launcher.cli import console_pipe
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers import scheduler as scheduler_mod
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems import configure

LAUNCH = [sys.executable, "-m", "hpc_launcher.cli.launch"]

# Wall-clock budgets for the two end-to-end tests below. They are deliberately
# lopsided so that neither test can decide the wrong way because the node was
# busy: the *observation* window is many times longer than the operation needs
# on an idle node, while the child that the observation races against outlives
# the window by a factor of three. See the individual docstrings.
_OBSERVE_TIMEOUT = 20.0
_CHILD_LIFETIME = 60


@pytest.mark.parametrize(
    ("scheduler", "identity_env", "expected"),
    [
        (
            SlurmScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0),
            {"SLURM_PROCID": "0", "SLURM_JOB_ID": "1234", "SLURM_STEP_ID": "7"},
            "HPC_LAUNCHER_STEP_ID=1234.7",
        ),
        (
            FluxScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0),
            {"FLUX_TASK_RANK": "0", "FLUX_JOB_ID": "f5wWq9"},
            "HPC_LAUNCHER_JOB_ID=f5wWq9",
        ),
    ],
)
def test_ephemeral_scheduler_wrapper_reports_identity_and_preserves_argv(
    scheduler, identity_env, expected
):
    """Rank zero announces its scheduler identity before execing user code."""
    user_code = "import sys; print(repr(sys.argv[1:]))"
    user_args = ["space separated", "semi;colon", "$(literal)"]
    proc = subprocess.run(
        scheduler.ephemeral_identity_wrapper()
        + [sys.executable, "-c", user_code]
        + user_args,
        capture_output=True,
        text=True,
        env={**os.environ, **identity_env},
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.strip() == expected
    assert proc.stdout.strip() == repr(user_args)


@pytest.mark.parametrize(
    ("scheduler", "identity_env"),
    [
        (
            SlurmScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0),
            {"SLURM_PROCID": "1", "SLURM_JOB_ID": "1234", "SLURM_STEP_ID": "7"},
        ),
        (
            FluxScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0),
            {"FLUX_TASK_RANK": "1", "FLUX_JOB_ID": "f5wWq9"},
        ),
    ],
)
def test_ephemeral_scheduler_wrapper_is_quiet_off_rank_zero(
    scheduler, identity_env
):
    """A multi-task launch emits exactly one identity announcement."""
    proc = subprocess.run(
        scheduler.ephemeral_identity_wrapper() + ["/bin/true"],
        capture_output=True,
        text=True,
        env={**os.environ, **identity_env},
    )

    assert proc.returncode == 0
    assert proc.stderr == ""


@pytest.mark.parametrize(
    "scheduler",
    [
        SlurmScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0),
        FluxScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0),
    ],
)
def test_ephemeral_launch_inserts_identity_wrapper(
    scheduler, monkeypatch, stub_system
):
    """The no-launch-directory path places the wrapper before user argv."""
    captured = {}

    def _fake_runner(command, **kwargs):
        captured["command"] = command
        return 0

    monkeypatch.setattr(scheduler_mod, "run_process_with_live_output", _fake_runner)
    user_argv = ["/bin/echo", "space separated", "semi;colon"]

    result = scheduler.launch(
        stub_system,
        folder_name=None,
        filename=None,
        command=user_argv[0],
        args=user_argv[1:],
        blocking=True,
    )

    expected_tail = scheduler.ephemeral_identity_wrapper() + user_argv
    assert captured["command"][-len(expected_tail):] == expected_tail
    assert result.returncode == 0


def _read_line_with_timeout(stream, timeout: float):
    """
    Return the first line ``stream`` produces, or None if none arrives within
    ``timeout`` seconds. The read happens on a daemon thread so a stream that
    never produces anything cannot wedge the test session.
    """
    result = queue.Queue()

    def _reader():
        try:
            result.put(stream.readline())
        except Exception:  # pragma: no cover - stream closed under us
            result.put("")

    threading.Thread(target=_reader, daemon=True).start()
    try:
        return result.get(timeout=timeout)
    except queue.Empty:
        return None


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` names a live (non-zombie) process."""
    try:
        with open(f"/proc/{pid}/stat") as fp:
            # State is the field after the parenthesised comm, which may
            # itself contain spaces/parens -- split on the last ')'.
            state = fp.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


def _ppid_of(pid: int) -> int:
    """The parent pid of ``pid``, for failure messages (0 if it is gone)."""
    try:
        with open(f"/proc/{pid}/stat") as fp:
            return int(fp.read().rsplit(")", 1)[1].split()[1])
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return 0


def _children_of(pid: int) -> list[int]:
    """The pids of ``pid``'s direct children, via ``ps --ppid``."""
    proc = subprocess.run(
        ["ps", "--ppid", str(pid), "-o", "pid="], capture_output=True, text=True
    )
    return [int(line) for line in proc.stdout.split() if line.strip()]


# ---------------------------------------------------------------------------
# Buffering -- the ephemeral path must stream, not buffer until exit.
# ---------------------------------------------------------------------------
def test_ephemeral_blocking_launch_uses_live_output(monkeypatch, stub_system):
    """
    Structural form of the streaming guarantee, and the primary regression
    test for this path: an ephemeral blocking launch must be run by
    ``console_pipe.run_process_with_live_output`` -- the same machinery the
    ``-l`` path (``scheduler.py``'s ``if blocking:`` branch) already uses --
    rather than by a private ``subprocess.run``.

    This is asserted structurally rather than by timing on purpose. Any
    wall-clock assertion of the form "the second line arrived at least N
    seconds after the first" is a coin flip on a loaded node; "the ephemeral
    path and the ``-l`` path go through one runner" is the property that
    actually has to hold, and it cannot be true intermittently. The
    end-to-end timing behaviour it buys is covered separately, with a
    generous budget, by ``test_ephemeral_output_arrives_before_the_job_exits``.

    ``out_file``/``err_file`` must stay None: an ephemeral run is defined by
    creating no files, so this is console-only tee-ing.
    """
    calls = []

    def _fake_runner(command, out_file=None, err_file=None, **kwargs):
        calls.append((command, out_file, err_file, kwargs))
        return 7

    monkeypatch.setattr(scheduler_mod, "run_process_with_live_output", _fake_runner)

    scheduler = LocalScheduler(nodes=1, procs_per_node=1, gpus_per_proc=0)
    result = scheduler.launch(
        stub_system,
        None,  # no launch folder
        None,  # no launch script
        "/bin/echo",
        ["hello"],
    )

    assert len(calls) == 1, (
        "the ephemeral blocking path did not go through "
        "run_process_with_live_output, so it neither streams output nor "
        "inherits console_pipe's signal forwarding"
    )
    command, out_file, err_file, _ = calls[0]
    assert command[-2:] == ["/bin/echo", "hello"]
    assert out_file is None and err_file is None, (
        "an ephemeral run must not open any files; output goes to the "
        "console only"
    )
    assert result.returncode == 7, (
        "the runner's exit code must still be propagated to the caller"
    )


def test_live_output_without_files_still_isolates_the_child(tmp_path):
    """
    ``run_process_with_live_output`` used to short-circuit to
    ``run_process_without_files`` -- a plain ``subprocess.run(" ".join(cmd),
    shell=True)`` -- whenever no output files and no stderr colouring were
    requested. That is exactly the shape of call the fixed ephemeral path
    makes, so routing the ephemeral path through the tee-er would have
    changed nothing on its own: the short-circuit has no
    ``start_new_session``, so the child stays in the launcher's process
    group and none of ``_run_process``'s SIGINT/SIGTERM forwarding is
    installed.

    Assert the property that forwarding depends on: the child leads its own
    process group. (It also incidentally re-joins the command with spaces
    and hands it to a shell, so any argument containing a space or a shell
    metacharacter was re-parsed -- another reason for the ephemeral path not
    to use it.)
    """
    reporter = tmp_path / "reporter.sh"
    result_file = tmp_path / "pgid.txt"
    # No spaces anywhere in the command: the pre-fix shell short-circuit
    # would mangle those, and this test is about the process group, not
    # about quoting.
    reporter.write_text(f"ps -o pgid= -p $$ > {result_file}\n")

    code = console_pipe.run_process_with_live_output(["/bin/sh", str(reporter)])

    assert code == 0
    child_pgid = int(result_file.read_text().strip())
    assert child_pgid != os.getpgrp(), (
        f"the child ran in the launcher's own process group ({child_pgid}); "
        "console_pipe cannot forward a signal to the job's process tree "
        "without killing itself, and grandchildren are unreachable"
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="uses /proc and ps --ppid"
)
def test_ephemeral_output_arrives_before_the_job_exits():
    """
    End-to-end complement to the structural test above, phrased so that it
    cannot be decided by how fast the node happens to be.

    The job prints one line and then sleeps for ``_CHILD_LIFETIME`` seconds.
    The assertion is not "the line arrived quickly" but "the line arrived
    *while the job was still running*" -- which is true only if output is
    being streamed, and false by construction if output is buffered until
    exit. The ``_OBSERVE_TIMEOUT`` read budget only has to be comfortably
    shorter than ``_CHILD_LIFETIME`` for that reasoning to hold; it is a
    third of it, against an operation (launcher start-up plus one write)
    that takes well under a second on an idle node.
    """
    emitter = (
        "import sys, time; "
        "print('EARLY-LINE', flush=True); "
        f"time.sleep({_CHILD_LIFETIME})"
    )
    proc = subprocess.Popen(
        LAUNCH + ["--local", "-N1", "--", sys.executable, "-u", "-c", emitter],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        # Own session so the cleanup below can take down the whole tree
        # regardless of whether the fix is in place.
        start_new_session=True,
    )
    try:
        line = _read_line_with_timeout(proc.stdout, _OBSERVE_TIMEOUT)
        assert proc.poll() is None, (
            "the job exited early; the test did not measure what it meant to"
        )
        assert line is not None and "EARLY-LINE" in line, (
            f"no output reached the console within {_OBSERVE_TIMEOUT}s while "
            f"the job was still running (it sleeps for {_CHILD_LIFETIME}s "
            f"after printing); got {line!r}. The ephemeral path is buffering "
            "the job's output until exit."
        )
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# Orphaning -- a SIGTERM to the launcher must not orphan the scheduler child.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="uses /proc and ps --ppid"
)
def test_ephemeral_sigterm_does_not_orphan_the_child():
    """
    Signal the launcher (and only the launcher) with SIGTERM, the way a
    scheduler's own time-limit enforcement, a supervisor, or a plain ``kill``
    would, and require the job to die with it.

    Before the fix the launcher was blocked in ``subprocess.run`` with no
    handler installed, so the default disposition terminated the launcher
    immediately and the job -- in the launcher's process group but with no
    one left to signal it -- was reparented to PID 1 and kept running to
    completion. On a real system that is a full-size job still holding an
    allocation after its launcher is gone.

    Only the launcher pid is signalled (never the group), otherwise the test
    would pass trivially.
    """
    proc = subprocess.Popen(
        LAUNCH + ["--local", "-N1", "--", "/bin/sleep", str(_CHILD_LIFETIME)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    child = None
    try:
        # Wait for the job itself to appear beneath the launcher.
        deadline = time.time() + _OBSERVE_TIMEOUT
        while time.time() < deadline:
            children = _children_of(proc.pid)
            if children:
                child = children[0]
                break
            time.sleep(0.1)
        assert child is not None, (
            f"the launcher started no child within {_OBSERVE_TIMEOUT}s"
        )

        os.kill(proc.pid, signal.SIGTERM)  # the launcher alone
        proc.wait(timeout=_OBSERVE_TIMEOUT)

        # The child is not ours any more once the launcher dies, so poll
        # /proc rather than waiting on it. The budget is the same generous
        # one, against a child that would otherwise live _CHILD_LIFETIME.
        deadline = time.time() + _OBSERVE_TIMEOUT
        while time.time() < deadline and _pid_alive(child):
            time.sleep(0.1)
        assert not _pid_alive(child), (
            f"pid {child} (the job) survived a SIGTERM to the launcher and "
            f"was still running {_OBSERVE_TIMEOUT}s later, reparented to "
            f"ppid {_ppid_of(child)}; the ephemeral path does not forward "
            "signals to the job"
        )
    finally:
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


# ---------------------------------------------------------------------------
# Adjacent -- the ephemeral --local path dropped the system environment.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /usr/bin/env output"
)
def test_ephemeral_local_job_receives_the_system_environment(tmp_path):
    """
    The launcher's job is, more than anything else, to put the system's tuned
    environment in front of the user's command: on an El Capitan-class node
    that is ``NCCL_SOCKET_IFNAME``, ``FI_CXI_*``, ``MIOPEN_*`` and friends,
    and getting it wrong is the difference between a job that runs and one
    that hangs in RCCL.

    Every other mode delivers it -- ``-l`` writes it into the launch script,
    and a real scheduler's ephemeral run puts it on the submit command line
    via ``cli_env_arg``. ``--local`` has neither: ``Scheduler.launch`` never
    calls ``launcher_script`` without a launch folder, and
    ``LocalScheduler.launch_command()`` is ``[]``, so there was no channel at
    all and the block was silently dropped. The failure is invisible --
    right output, right exit code, wrong network settings.

    Compare the ephemeral run against the names the system itself says it
    injects, rather than against a hard-coded list, so this stays meaningful
    on any system (and skips on one that tunes nothing, e.g. a generic
    x86 box).
    """
    system, _, _, _ = configure.configure_launch(None, 1, 1, 1, None, None)
    expected = [e[0] for e in system.environment_variables() if len(e) >= 2]
    expected += [e[0] for e in system.passthrough_environment_variables() if len(e) >= 2]
    if not expected:
        pytest.skip("this system injects no environment variables")

    proc = subprocess.run(
        LAUNCH + ["--local", "-N1", "--", "/usr/bin/env"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    seen = {
        line.split("=", 1)[0] for line in proc.stdout.splitlines() if "=" in line
    }
    missing = sorted(set(expected) - seen)
    assert not missing, (
        f"an ephemeral 'launch --local' ran the job without {len(missing)} of "
        f"the {len(set(expected))} environment variables this system asks for: "
        f"{missing}"
    )
