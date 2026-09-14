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
Regression tests: scheduler argument dictionaries and the System auxiliary
environment-variable list must be per-instance state, not shared class-level
state. No torch, no scheduler binaries, no autodetection -- systems are
constructed directly.
"""
import pytest

from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler
from hpc_launcher.systems.lc.el_capitan_family import ElCapitan
from hpc_launcher.systems.system import GenericSystem

# These tests assert that allocation-selection flags such as --partition
# appear on a blocking launch command; inside an existing allocation Slurm
# deliberately drops them (a nested job step cannot change partition).
pytestmark = pytest.mark.usefixtures("outside_allocation")


def test_scheduler_args_not_shared_between_instances(stub_system):
    # First scheduler: populate submit/run/common args via a real launch
    # command construction.
    first = SlurmScheduler(
        nodes=4,
        procs_per_node=2,
        gpus_per_proc=1,
        queue="pbatch",
        time_limit=60,
    )
    first_cmd = first.launch_command(stub_system, blocking=True)
    assert any("--gpus-per-task" in c for c in first_cmd)
    assert any("--partition" in c for c in first_cmd)
    assert any("--time" in c for c in first_cmd)

    # Second, bare scheduler of the same class must not see any of the
    # first scheduler's flags.
    second = SlurmScheduler(nodes=2, procs_per_node=1, gpus_per_proc=0)
    second_cmd = second.launch_command(stub_system, blocking=True)

    for leaked_flag in ("--gpus-per-task", "--partition", "--time"):
        assert not any(
            leaked_flag in c for c in second_cmd
        ), f"{leaked_flag} leaked from a prior SlurmScheduler instance: {second_cmd}"


def test_scheduler_args_not_shared_across_classes(stub_system):
    first = SlurmScheduler(
        nodes=4,
        procs_per_node=2,
        gpus_per_proc=1,
        queue="pbatch",
        time_limit=60,
    )
    first_cmd = first.launch_command(stub_system, blocking=True)
    assert any("--gpus-per-task" in c for c in first_cmd)
    assert any("--partition" in c for c in first_cmd)
    assert any("--time" in c for c in first_cmd)

    second = FluxScheduler(nodes=2, procs_per_node=1, gpus_per_proc=0)
    second_cmd = second.launch_command(stub_system, blocking=True)

    for leaked_flag in ("--gpus-per-task", "--partition", "--time"):
        assert not any(
            leaked_flag in c for c in second_cmd
        ), f"{leaked_flag} leaked from a SlurmScheduler into a fresh FluxScheduler: {second_cmd}"


def test_aux_env_list_not_shared():
    ElCapitan("tioga").extend_environment_variables([("LEAKED_VAR", "1")])
    leaked = [
        e for e in GenericSystem().environment_variables() if "LEAKED" in str(e)
    ]
    assert leaked == []
