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
Per-scheduler allocation detection: ``Scheduler.in_allocation()`` and
``Scheduler.num_nodes_in_allocation()``. Every scheduler recognizes *its
own* allocation from the environment and nothing else's. Pure environment
tests -- no scheduler binaries (``flux resource info`` is stubbed), no
system autodetection.
"""
import subprocess
from unittest.mock import patch

import pytest

from hpc_launcher.schedulers import num_nodes_in_current_allocation
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.schedulers.lsf import LSFScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler

# Scheduler -> the environment variable that marks its allocation.
_MARKERS = {
    SlurmScheduler: ("SLURM_JOB_ID", "424242"),
    FluxScheduler: ("FLUX_URI", "local:///run/flux/local"),
    LSFScheduler: ("LSB_HOSTS", "host1 host1 host2 host2"),
}

_ALL_ALLOC_VARS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NUM_NODES",
    "FLUX_URI",
    "LSB_HOSTS",
    "LLNL_NUM_COMPUTE_NODES",
)


@pytest.fixture
def no_allocation(monkeypatch):
    """Clear every scheduler's allocation marker so tests are hermetic."""
    for var in _ALL_ALLOC_VARS:
        monkeypatch.delenv(var, raising=False)


def _fake_flux_resource_info(nodes: int):
    """A subprocess.run stand-in that answers `flux resource info`."""

    def run(cmd, *args, **kwargs):
        assert cmd == ["flux", "resource", "info"], cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout=f"{nodes} Nodes, {nodes * 64} Cores, {nodes * 4} GPUs\n", stderr=""
        )

    return run


# ---------------------------------------------------------------------------
# in_allocation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheduler_cls", list(_MARKERS), ids=lambda c: c.__name__)
def test_in_allocation_tracks_own_marker(scheduler_cls, no_allocation, monkeypatch):
    var, value = _MARKERS[scheduler_cls]
    assert scheduler_cls.in_allocation() is False

    monkeypatch.setenv(var, value)
    assert scheduler_cls.in_allocation() is True
    # Also callable on an instance, which is how system code reaches it.
    assert scheduler_cls(nodes=1, procs_per_node=1, gpus_per_proc=0).in_allocation() is True


@pytest.mark.parametrize("scheduler_cls", list(_MARKERS), ids=lambda c: c.__name__)
def test_in_allocation_ignores_other_schedulers_markers(
    scheduler_cls, no_allocation, monkeypatch
):
    # Set every marker *except* this scheduler's own: being inside a Flux
    # instance says nothing about a Slurm job step, and so on.
    for other_cls, (var, value) in _MARKERS.items():
        if other_cls is not scheduler_cls:
            monkeypatch.setenv(var, value)
    assert scheduler_cls.in_allocation() is False


def test_local_scheduler_is_never_in_allocation(no_allocation, monkeypatch):
    for var, value in _MARKERS.values():
        monkeypatch.setenv(var, value)
    assert LocalScheduler.in_allocation() is False
    assert LocalScheduler.num_nodes_in_allocation() is None


# ---------------------------------------------------------------------------
# num_nodes_in_allocation: each scheduler reads only its own environment
# ---------------------------------------------------------------------------
def test_slurm_node_count(no_allocation, monkeypatch):
    assert SlurmScheduler.num_nodes_in_allocation() is None
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "4")
    with patch("subprocess.run") as run:
        assert SlurmScheduler.num_nodes_in_allocation() == 4
        run.assert_not_called()


def test_slurm_node_count_asks_squeue_when_env_incomplete(no_allocation, monkeypatch):
    # A shell with only SLURM_JOB_ID exported (e.g. after ssh to an
    # allocated node) still runs srun as a nested step, so the size has to
    # come from the controller.
    monkeypatch.setenv("SLURM_JOB_ID", "329549")

    def squeue(cmd, *args, **kwargs):
        assert cmd == ["squeue", "-h", "-j", "329549", "-o", "%D"], cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")

    with patch("subprocess.run", side_effect=squeue):
        assert SlurmScheduler.num_nodes_in_allocation() == 1

    # No squeue on the host, or a job the controller no longer knows about:
    # report "unknown" rather than fail.
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert SlurmScheduler.num_nodes_in_allocation() is None
    gone = lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
    with patch("subprocess.run", side_effect=gone):
        assert SlurmScheduler.num_nodes_in_allocation() is None


def test_slurm_node_count_ignores_flux_and_lsf(no_allocation, monkeypatch):
    # Regression: this used to shell out to `flux resource info` when
    # FLUX_URI was set and to read LLNL_NUM_COMPUTE_NODES as a fallback.
    monkeypatch.setenv("FLUX_URI", "local:///run/flux/local")
    monkeypatch.setenv("LSB_HOSTS", "h1 h2")
    monkeypatch.setenv("LLNL_NUM_COMPUTE_NODES", "2")
    with patch("subprocess.run") as run:
        assert SlurmScheduler.num_nodes_in_allocation() is None
        run.assert_not_called()


def test_lsf_node_count_prefers_lc_export(no_allocation, monkeypatch):
    monkeypatch.setenv("LSB_HOSTS", "h1 h1 h2 h2 h3 h3")
    monkeypatch.setenv("LLNL_NUM_COMPUTE_NODES", "2")
    assert LSFScheduler.num_nodes_in_allocation() == 2


def test_lsf_node_count_from_lsb_hosts(no_allocation, monkeypatch):
    # LSB_HOSTS lists a host once per slot; count distinct hosts.
    monkeypatch.setenv("LSB_HOSTS", "h1 h1 h2 h2 h3 h3")
    assert LSFScheduler.num_nodes_in_allocation() == 3


def test_lsf_node_count_requires_lsf_allocation(no_allocation, monkeypatch):
    monkeypatch.setenv("LLNL_NUM_COMPUTE_NODES", "2")
    assert LSFScheduler.num_nodes_in_allocation() is None


def test_flux_node_count(no_allocation, monkeypatch):
    with patch("subprocess.run", side_effect=_fake_flux_resource_info(8)) as run:
        assert FluxScheduler.num_nodes_in_allocation() is None
        run.assert_not_called()
        monkeypatch.setenv("FLUX_URI", "local:///run/flux/local")
        assert FluxScheduler.num_nodes_in_allocation() == 8


# ---------------------------------------------------------------------------
# num_nodes_in_current_allocation: asks each scheduler in turn
# ---------------------------------------------------------------------------
def test_current_allocation_prefers_flux_inside_slurm(no_allocation, monkeypatch):
    # A Flux instance started inside a Slurm job: both markers are set and
    # the Flux instance is the allocation the user is working in.
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "16")
    monkeypatch.setenv("FLUX_URI", "local:///run/flux/local")
    with patch("subprocess.run", side_effect=_fake_flux_resource_info(2)):
        assert num_nodes_in_current_allocation() == 2


def test_current_allocation_falls_through_when_flux_probe_fails(
    no_allocation, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "16")
    monkeypatch.setenv("FLUX_URI", "local:///run/flux/local")
    broken = lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
    with patch("subprocess.run", side_effect=broken):
        assert num_nodes_in_current_allocation() == 16
