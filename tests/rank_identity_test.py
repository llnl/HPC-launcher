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
Rank identity must be published by the component that knows it.

The defects below are three faces of one mistake: the generated shell script
published rank identity, and the script does not always run where a rank
exists.

- **``RANK``** -- ``export RANK=${SLURM_PROCID}`` was written into the
  *batch* script for a ``--bg`` submission. That script runs once, at allocation
  scope, before ``srun``/``flux run`` ever forks a task, so the value was
  frozen and inherited unchanged by every task: ``0`` under Slurm (whose
  batch step really is a one-task step with ``SLURM_PROCID=0``), empty under
  Flux (``FLUX_TASK_RANK`` is actively unset for an initial program) and
  empty under LSF. Slurm's plausible-looking ``0`` is the worse outcome: every
  rank passes a rank-0 guard, so checkpointing/logging/evaluation become an
  N-way concurrent write to one path. The same line was also expanded on the
  *launch host* for the ephemeral CLI-env path, where the scheduler's rank
  variable is not set at all.
- **``--save-hostlist``, which depends on ``RANK``** -- the
  ``if [ "${RANK}" = "0" ]`` guard was
  evaluated in that same batch scope. Under Slurm it saw ``0`` and the file was
  written by luck; under Flux and LSF it compared ``""`` to ``"0"`` and the
  hostlist file was silently never created.
- **``LOCAL_RANK``** -- it was set to ``local_rank % len(visible_devices)``, a
  *device index*. The launcher always passes ``--gpus-per-task`` (default 1),
  so each task sees exactly one device and every rank on the node reported
  ``LOCAL_RANK=0``. Note that the round-robin device *selection* is correct
  and deliberately unchanged (see ``gpu_visibility_test.py``); only its reuse
  as an identity was wrong.
- **``NODE_RANK``** -- documented, and set nowhere.

The fix moves all of it into ``torchrun_hpc_trampoline``, which is executed
once per task by construction and already computes the rank it hands to
``init_process_group``.

Two techniques are used here, for two different scopes:

- *Generated-script tests* assert on the text of a real ``launcher_script``
  output, and -- for the ones that turn on batch-versus-task scope -- actually
  execute that script with stub ``srun``/``scontrol``/``flux`` programs on
  ``PATH`` that set the per-task variables the way the real tools do. Executing
  the script is the only way to show that a line ran at the wrong scope.
- *Trampoline tests* drive the real ``main()`` in-process with a stubbed
  scheduler view, because the identity a rank publishes has to be checked for a
  multi-rank, multi-node world and no such world can be created here.
"""
import json
import os
import re
import subprocess
import sys

import pytest

from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.lsf import LSFScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems.system import GenericSystem

from conftest import require_torch

# Every test here builds commands or scripts and runs them, if at all, under
# stub scheduler programs; none touches a real allocation. Inside one, Slurm
# would build a nested job step instead and reject NODES=2 in a smaller job.
pytestmark = pytest.mark.usefixtures("outside_allocation")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCHRUN_HPC_CLI_MD = os.path.join(REPO_ROOT, "torchrun-hpc_cli.md")

# A four-task job spread over two nodes: big enough that a per-task value and
# an allocation-scope constant cannot be confused for one another.
NODES = 2
PROCS_PER_NODE = 2
NTASKS = NODES * PROCS_PER_NODE


def _torchrun_style_script(scheduler_cls, blocking, launch_dir,
                           command="python", args=None,
                           save_hostlist=True):
    """
    Build the launch script a ``torchrun-hpc`` invocation would generate.

    ``setup_rendezvous_protocol`` is the hook that used to contribute the
    ``RANK`` entry, so it has to be part of the system's environment for these
    tests to see what a real torchrun-hpc job sees.
    """
    scheduler = scheduler_cls(nodes=NODES, procs_per_node=PROCS_PER_NODE,
                              gpus_per_proc=1)
    system = GenericSystem()
    system.extend_environment_variables(
        scheduler.setup_rendezvous_protocol("tcp"))
    return scheduler.launcher_script(
        system, command, args if args is not None else [], blocking=blocking,
        save_hostlist=save_hostlist, launch_dir=str(launch_dir),
    )


# ---------------------------------------------------------------------------
# The generated script must not publish a rank it cannot know
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheduler_cls",
                         [SlurmScheduler, FluxScheduler, LSFScheduler])
def test_batch_script_publishes_no_rank(scheduler_cls, tmp_path, monkeypatch):
    """
    A ``--bg`` script runs once at allocation scope, so any ``export RANK=``
    in it is a constant that every task then inherits. There is no correct
    value to write there -- the only correct action is to write nothing and
    let the per-task component publish the rank.

    This also covers the "emitted twice" defect: the rendezvous env list and
    the ``--save-hostlist`` block each contributed one ``export RANK=`` line to
    the same script.
    """
    monkeypatch.delenv("LSB_HOSTS", raising=False)

    script = _torchrun_style_script(scheduler_cls, blocking=False,
                                    launch_dir=tmp_path)

    assert "export RANK=" not in script, (
        "the batch script exports a rank at allocation scope; it runs once, "
        "before any task exists, so every task inherits this one constant:\n"
        f"{script}"
    )


@pytest.mark.parametrize("scheduler_cls",
                         [SlurmScheduler, FluxScheduler, LSFScheduler])
def test_batch_scope_hostlist_write_is_unguarded(scheduler_cls, tmp_path,
                                                 monkeypatch):
    """
    The ``--save-hostlist`` write is guarded by a rank-0 test so that exactly
    one process writes the file. In a ``--bg`` script that guard is evaluated
    at allocation scope, where the scheduler's rank variable is either a
    meaningless ``0`` (Slurm) or unset (Flux, LSF) -- so the guard either
    passes by luck or silently drops the file.

    Allocation scope already *is* "exactly one process", so the guard has
    nothing to do there and must not be emitted.
    """
    monkeypatch.delenv("LSB_HOSTS", raising=False)

    script = _torchrun_style_script(scheduler_cls, blocking=False,
                                    launch_dir=tmp_path)

    assert "hpc_launcher_hostlist.txt" in script, script
    assert "if [" not in script, (
        "the batch script guards the hostlist write on a rank that does not "
        f"exist at allocation scope:\n{script}"
    )


@pytest.mark.parametrize("scheduler_cls", [SlurmScheduler, FluxScheduler])
def test_per_task_hostlist_guard_uses_the_native_rank_variable(
        scheduler_cls, tmp_path):
    """
    An interactive ``srun``/``flux run`` executes the generated script itself,
    once per task, so there the guard is both needed and meaningful. It must
    read the scheduler's own per-task variable directly rather than a
    ``RANK`` snapshot taken by an earlier ``export`` -- a snapshot is exactly
    what stops working the moment the script moves to allocation scope.
    """
    script = _torchrun_style_script(scheduler_cls, blocking=True,
                                    launch_dir=tmp_path)

    native = scheduler_cls.get_parallel_rank_env_variable()
    assert f'if [ "{native}" = "0" ]' in script, (
        f"the per-task hostlist guard does not read {native}:\n{script}"
    )
    assert "export RANK=" not in script, (
        "rank identity is published by the trampoline, not by a shell "
        f"snapshot in the launch script:\n{script}"
    )


def test_lsf_inside_an_allocation_guards_with_the_ompi_rank(tmp_path,
                                                            monkeypatch):
    """
    LSF's blocking launch command is ``jsrun`` only when we are already inside
    an allocation (``LSB_HOSTS`` set); that is the one LSF configuration in
    which the generated script is the per-task program and therefore needs the
    guard. Outside an allocation ``bsub -Is`` runs the script once and jsrun
    lives inside it, which is allocation scope again.
    """
    monkeypatch.setenv("LSB_HOSTS", "lassen1 lassen1 lassen2 lassen2")

    script = _torchrun_style_script(LSFScheduler, blocking=True,
                                    launch_dir=tmp_path)

    assert 'if [ "${OMPI_COMM_WORLD_RANK}" = "0" ]' in script, script
    assert "export RANK=" not in script, script


@pytest.mark.parametrize("scheduler_cls", [SlurmScheduler, FluxScheduler])
def test_ephemeral_cli_env_does_not_pin_a_launch_host_rank(scheduler_cls,
                                                           monkeypatch):
    """
    On the ephemeral path there is no script at all: the environment is folded
    onto the scheduler command line and every ``${VAR}`` is expanded
    *in-process, on the launch host*. A ``RANK`` entry there is expanded
    against a login shell that has no ``SLURM_PROCID``/``FLUX_TASK_RANK``, so
    it pins an empty rank onto every task of the job.
    """
    for var in ("SLURM_PROCID", "FLUX_TASK_RANK", "OMPI_COMM_WORLD_RANK"):
        monkeypatch.delenv(var, raising=False)

    scheduler = scheduler_cls(nodes=NODES, procs_per_node=PROCS_PER_NODE,
                              gpus_per_proc=1)
    system = GenericSystem()
    system.extend_environment_variables(
        scheduler.setup_rendezvous_protocol("tcp"))

    cmd = scheduler.launch_command(system, blocking=True, cli_env_only=True)

    bare_rank = re.compile(r"(?<![A-Za-z0-9_])RANK(?![A-Za-z0-9_])")
    offenders = [t for t in cmd if bare_rank.search(t)]
    assert not offenders, (
        f"a launch-host RANK was pinned onto the job's command line: "
        f"{offenders}"
    )


# ---------------------------------------------------------------------------
# The same thing, demonstrated by executing the generated script
# ---------------------------------------------------------------------------
_STUB_TASK_LOOP = """\
i=0
while [ "$i" -lt "$STUB_NTASKS" ]; do
    STUB_TASK_ID="$i" {per_task_vars} "$@"
    i=$((i + 1))
done
"""

_SKIP_OWN_FLAGS = """\
while [ $# -gt 0 ]; do
    case "$1" in
        -*) shift ;;
        *) break ;;
    esac
done
"""


def _write_stub(directory, name, body):
    path = os.path.join(directory, name)
    with open(path, "w") as fh:
        fh.write("#!/bin/sh\n" + body)
    os.chmod(path, 0o755)
    return path


def _make_stub_bin(tmp_path):
    """
    Build a directory of stub scheduler programs that reproduce the one
    property that matters here: the per-task variables are set by the *task
    launcher*, at task launch, and by nothing earlier.

    ``srun``/``flux run`` consume their own options (all of which start with
    ``-``) and then run the remaining command once per task, exporting the
    variables the real tool exports. ``scontrol``/``flux hostlist`` answer the
    hostname queries the generated scripts make.
    """
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    stub_bin = str(stub_bin)

    _write_stub(
        stub_bin, "srun",
        _SKIP_OWN_FLAGS
        + _STUB_TASK_LOOP.format(
            per_task_vars='SLURM_PROCID="$i" SLURM_LOCALID="$i"'),
    )
    _write_stub(stub_bin, "scontrol", "echo stub-node0\necho stub-node1\n")
    _write_stub(
        stub_bin, "flux",
        'sub="$1"\nshift\ncase "$sub" in\n'
        '    hostlist) echo "stub-node0 stub-node1" ;;\n'
        "    run)\n"
        + _SKIP_OWN_FLAGS
        + _STUB_TASK_LOOP.format(
            per_task_vars='FLUX_TASK_RANK="$i" FLUX_TASK_LOCAL_ID="$i"')
        + '        ;;\n    *) echo "stub flux: $sub" >&2; exit 1 ;;\nesac\n',
    )
    # The generated flux script pipes its hostlist through /bin/hostlist,
    # which does not exist off a real LC machine; provide it so the stub run
    # is quiet and the master address resolves.
    _write_stub(stub_bin, "hostlist", 'cut -d" " -f1\n')
    return stub_bin


# Per scheduler: the environment its submit command leaves in the batch
# script's own environment. Slurm's batch step is a real one-task step and
# does set SLURM_PROCID=0 -- that is precisely why the frozen value looks
# plausible. Flux actively unsets FLUX_TASK_RANK for an initial program.
_BATCH_SCOPE_ENV = {
    SlurmScheduler: {
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NTASKS": str(NTASKS),
        "SLURM_NNODES": str(NODES),
        "SLURM_JOB_NODELIST": "stub-node[0-1]",
    },
    FluxScheduler: {
        "FLUX_JOB_SIZE": str(NTASKS),
        "FLUX_JOB_NNODES": str(NODES),
    },
}


def _run_generated_script(scheduler_cls, tmp_path, blocking, extra_env=None):
    """
    Generate a real launch script whose "user program" reports the rank it was
    handed, execute it under the stub scheduler programs, and return the
    ``(task_id, RANK)`` pairs it recorded.
    """
    stub_bin = _make_stub_bin(tmp_path)
    report = tmp_path / "rank_report.txt"
    reporter = tmp_path / "report_rank.sh"
    reporter.write_text(
        '#!/bin/sh\n'
        'echo "${STUB_TASK_ID:-0} ${RANK-<UNSET>}" >> "$RANK_REPORT"\n'
    )
    reporter.chmod(0o755)

    script_text = _torchrun_style_script(scheduler_cls, blocking=blocking,
                                         launch_dir=tmp_path,
                                         command=str(reporter))
    script = tmp_path / "launch.sh"
    script.write_text(script_text)
    script.chmod(0o700)

    env = os.environ.copy()
    env["PATH"] = stub_bin + os.pathsep + env["PATH"]
    env["STUB_NTASKS"] = str(NTASKS)
    env["RANK_REPORT"] = str(report)
    env.pop("RANK", None)
    env.update(_BATCH_SCOPE_ENV[scheduler_cls])
    env.update(extra_env or {})

    proc = subprocess.run(["/bin/sh", str(script)], env=env, cwd=str(tmp_path),
                          capture_output=True, universal_newlines=True,
                          timeout=120)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

    pairs = []
    if report.exists():
        for line in report.read_text().splitlines():
            task_id, rank = line.split(" ", 1)
            pairs.append((task_id, rank))
    return pairs, script_text


@pytest.mark.parametrize("scheduler_cls", [SlurmScheduler, FluxScheduler])
def test_bg_tasks_do_not_inherit_a_frozen_rank(scheduler_cls, tmp_path):
    """
    Execute the real ``--bg`` launch script under stub schedulers and check
    what each of the four tasks was actually told its rank was.

    Before the fix this printed ``RANK=0`` for tasks 0-3 under Slurm (a rank
    that three of the four tasks do not have) and ``RANK=`` under Flux. A task
    may legitimately be told nothing -- the trampoline publishes the rank
    later, in the task -- but it must never be told a rank that contradicts
    the one the scheduler gave it.
    """
    pairs, script_text = _run_generated_script(scheduler_cls, tmp_path,
                                               blocking=False)

    assert len(pairs) == NTASKS, f"{pairs}\n{script_text}"
    contradicted = [(task, rank) for task, rank in pairs
                    if rank != "<UNSET>" and rank != task]
    assert not contradicted, (
        "tasks were handed a rank that is not theirs (task_id, RANK): "
        f"{contradicted}\nfrom script:\n{script_text}"
    )


@pytest.mark.parametrize("scheduler_cls", [SlurmScheduler, FluxScheduler])
def test_bg_save_hostlist_writes_the_file(scheduler_cls, tmp_path):
    """
    ``--bg --save-hostlist`` must produce ``hpc_launcher_hostlist.txt``.

    Under Slurm this passed before the fix -- the batch step's
    ``SLURM_PROCID=0`` made the rank-0 guard true by luck -- so that
    parametrization is a non-regression check. Under Flux the guard compared
    an empty ``FLUX_TASK_RANK`` against ``"0"``, the file was silently never
    created, and this is the reproducer.
    """
    _run_generated_script(scheduler_cls, tmp_path, blocking=False)

    hostlist = tmp_path / "hpc_launcher_hostlist.txt"
    assert hostlist.exists(), (
        "--save-hostlist produced no hostlist file; the rank-0 guard was "
        "evaluated at allocation scope, where the scheduler's rank variable "
        "is not set"
    )
    assert hostlist.read_text().strip(), "the hostlist file is empty"


def test_interactive_save_hostlist_is_written_exactly_once(tmp_path):
    """
    Control for the case that always worked: interactively the script *is* the
    per-task program, so the guard is load-bearing -- exactly one of the four
    tasks may write the file. Running the generated script four times with the
    per-task variables set is what ``srun launch.sh`` does.
    """
    stub_bin = _make_stub_bin(tmp_path)
    reporter = tmp_path / "report_rank.sh"
    reporter.write_text(
        '#!/bin/sh\necho "${SLURM_PROCID} ${RANK-<UNSET>}" >> "$RANK_REPORT"\n'
    )
    reporter.chmod(0o755)

    script_text = _torchrun_style_script(SlurmScheduler, blocking=True,
                                         launch_dir=tmp_path,
                                         command=str(reporter))
    script = tmp_path / "launch.sh"
    script.write_text(script_text)
    script.chmod(0o700)

    hostlist = tmp_path / "hpc_launcher_hostlist.txt"
    writes = 0
    for task in range(NTASKS):
        env = os.environ.copy()
        env["PATH"] = stub_bin + os.pathsep + env["PATH"]
        env["RANK_REPORT"] = str(tmp_path / "rank_report.txt")
        env["SLURM_JOB_NODELIST"] = "stub-node[0-1]"
        env["SLURM_PROCID"] = str(task)
        env["SLURM_LOCALID"] = str(task % PROCS_PER_NODE)
        env.pop("RANK", None)
        if hostlist.exists():
            hostlist.unlink()
        proc = subprocess.run(["/bin/sh", str(script)], env=env,
                              cwd=str(tmp_path), capture_output=True,
                              universal_newlines=True, timeout=120)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        writes += hostlist.exists()

    assert writes == 1, (
        f"{writes} of {NTASKS} tasks wrote the hostlist file; exactly one "
        f"must:\n{script_text}"
    )


# ---------------------------------------------------------------------------
# The trampoline is the thing that knows the rank
# ---------------------------------------------------------------------------
_IDENTITY_VARS = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "NODE_RANK")


def _run_trampoline(monkeypatch, tmp_path, *, world_size, rank,
                    local_world_size, local_rank, visible_devices="0",
                    inherited=None):
    """
    Drive the real ``torchrun_hpc_trampoline.main()`` in-process and return
    the identity environment its user script was handed.

    In-process, with ``torch.cuda.is_available`` forced False and
    ``init_process_group`` stubbed, because these assertions are about a
    *multi-rank, multi-node* world and this sandbox has neither a working
    accelerator collective nor a reachable rendezvous address for one. Nothing
    between the scheduler's numbers and the published environment is stubbed:
    the scheduler lookup, ``get_parallel_configuration``, the device selection
    and ``main()`` itself all run for real.

    :param inherited: Environment the task inherits from its launch script --
                      used to model the frozen ``RANK`` a ``--bg`` batch script
                      used to leave behind.
    """
    require_torch()
    import hpc_launcher.torch.torchrun_hpc_trampoline as tramp

    report = tmp_path / f"identity_{rank}.json"
    user_script = tmp_path / f"report_identity_{rank}.py"
    user_script.write_text(
        "import json, os\n"
        f"with open({str(report)!r}, 'w') as fh:\n"
        f"    json.dump({{k: os.environ.get(k) for k in {list(_IDENTITY_VARS)!r}}}, fh)\n"
    )

    for var in _IDENTITY_VARS:
        monkeypatch.delenv(var, raising=False)
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("HPC_LAUNCHER_MAX_GPU_MEM", raising=False)
    if visible_devices is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    for k, v in (inherited or {}).items():
        monkeypatch.setenv(k, v)

    monkeypatch.setenv("TORCHRUN_HPC_SCHEDULER", "slurm")
    monkeypatch.setenv("SLURM_NTASKS", str(world_size))
    monkeypatch.setenv("SLURM_PROCID", str(rank))
    monkeypatch.setenv("SLURM_LOCALID", str(local_rank))
    monkeypatch.setenv("SLURM_NNODES", str(world_size // local_world_size))
    monkeypatch.setenv("TORCHRUN_HPC_RDV_PROTOCOL", "tcp://127.0.0.1:29500")

    monkeypatch.setattr(tramp.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(tramp.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(tramp.dist, "init_process_group",
                        lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv",
                        ["torchrun_hpc_trampoline.py", str(user_script)])

    tramp.main()

    return json.loads(report.read_text())


def test_trampoline_overrides_a_frozen_batch_scope_rank(monkeypatch, tmp_path):
    """
    The frozen ``RANK``, from the task's side. Model a job whose batch script
    left ``RANK=0`` in the environment of all four tasks (Slurm's behaviour),
    and check that the rank each task's script finally reads is its own.

    This is the half the trampoline owns: it already computes this rank
    and hands it to ``init_process_group``, so publishing it costs nothing and
    is correct on every scheduler and both launch paths.
    """
    seen = _run_trampoline(monkeypatch, tmp_path, world_size=8, rank=5,
                           local_world_size=4, local_rank=1,
                           inherited={"RANK": "0"})

    assert seen["RANK"] == "5", seen
    assert seen["WORLD_SIZE"] == "8", seen


@pytest.mark.parametrize("local_rank", [0, 1, 2, 3])
def test_local_rank_is_the_node_local_rank_not_a_device_index(
        local_rank, monkeypatch, tmp_path):
    """
    A single visible device is the *default* configuration, not an edge
    case: the launcher always passes ``--gpus-per-task`` (default 1) and the
    scheduler confines each task to its own GPU, so
    ``local_rank % len(visible)`` is ``0`` for every rank on the node.
    ``LOCAL_RANK`` must be the rank's position on the node regardless.
    """
    seen = _run_trampoline(monkeypatch, tmp_path, world_size=8,
                           rank=4 + local_rank, local_world_size=4,
                           local_rank=local_rank, visible_devices="0")

    assert seen["LOCAL_RANK"] == str(local_rank), seen


def test_local_ranks_are_distinct_across_a_node(monkeypatch, tmp_path):
    """
    The property downstream code actually relies on: on one node, no two ranks
    claim the same local rank. Local-leader election, per-node-rank sharding
    and per-local-rank log file names are all silently wrong when they do.
    """
    local_ranks = [
        _run_trampoline(monkeypatch, tmp_path, world_size=8, rank=r,
                        local_world_size=4, local_rank=r,
                        visible_devices="0")["LOCAL_RANK"]
        for r in range(4)
    ]

    assert local_ranks == ["0", "1", "2", "3"], local_ranks


def test_device_index_and_local_rank_are_independent(monkeypatch, tmp_path):
    """
    The two quantities must stay separate in *both* directions: the device
    index keeps round-robining over the visible list (that is what prevents
    "invalid device ordinal" when a rank is granted fewer GPUs than expected),
    while ``LOCAL_RANK`` keeps naming the rank.
    """
    require_torch()
    import hpc_launcher.torch.torchrun_hpc_trampoline as tramp

    seen = _run_trampoline(monkeypatch, tmp_path, world_size=4, rank=3,
                           local_world_size=4, local_rank=3,
                           visible_devices="0")

    # Device selection: unchanged, still wraps onto the one granted device.
    assert tramp._select_local_device_id(3) == 0
    # Identity: the rank's own place on the node.
    assert seen["LOCAL_RANK"] == "3", seen


@pytest.mark.parametrize(
    "world_size,rank,local_world_size,expected",
    [
        (8, 0, 4, "0"),
        (8, 3, 4, "0"),
        (8, 4, 4, "1"),
        (8, 7, 4, "1"),
        # One rank per node: the node rank is the rank.
        (4, 2, 1, "2"),
        # Single-rank job.
        (1, 0, 1, "0"),
    ],
)
def test_node_rank_is_published(world_size, rank, local_world_size, expected,
                                monkeypatch, tmp_path):
    """
    ``NODE_RANK`` was documented in ``torchrun-hpc_cli.md`` and set
    nowhere, so a script written against the documented contract -- or a
    HuggingFace/DeepSpeed-style script that expects it next to
    ``RANK``/``LOCAL_RANK`` -- died with a ``KeyError`` at startup. Both
    inputs are already in hand where the rank is published.
    """
    seen = _run_trampoline(monkeypatch, tmp_path, world_size=world_size,
                           rank=rank, local_world_size=local_world_size,
                           local_rank=rank % local_world_size)

    assert seen["NODE_RANK"] == expected, seen


# ---------------------------------------------------------------------------
# The documentation has to say which quantity it means
# ---------------------------------------------------------------------------
def _torchrun_doc():
    with open(TORCHRUN_HPC_CLI_MD) as fh:
        return fh.read()


def test_doc_table_distinguishes_local_rank_from_the_device_index():
    """
    The table said "Local rank on the node" while the doc's own worked example
    used the same variable as a device index. After the fix they are different
    numbers, so the table has to say which one it is describing.
    """
    doc = _torchrun_doc()
    rows = [line for line in doc.splitlines()
            if line.startswith("| `LOCAL_RANK`")]

    assert rows, "no LOCAL_RANK row in the environment-variable table"
    assert "device" in rows[0].lower(), (
        "the LOCAL_RANK row does not say that it is not a device index: "
        f"{rows[0]}"
    )


def test_doc_example_does_not_use_local_rank_as_a_device_index():
    """
    The worked example is what users copy. It passed ``LOCAL_RANK`` straight
    to ``set_device``/``cuda()``/``device_ids``, which is exactly the
    conflation the fix separates.
    """
    doc = _torchrun_doc()

    for pattern in (
            "torch.cuda.set_device(local_rank)",
            "cuda(local_rank)",
            "device_ids=[local_rank]",
            "output_device=local_rank",
    ):
        assert pattern not in doc, (
            f"the worked example still uses LOCAL_RANK as a device index: "
            f"{pattern}"
        )
