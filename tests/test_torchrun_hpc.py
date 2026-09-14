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
import pytest

import subprocess
import shutil
import os
import re
import sys
import shutil

from hpc_launcher.systems import autodetect
from hpc_launcher.systems.lc.sierra_family import Sierra
from hpc_launcher.schedulers import get_schedulers

from conftest import require_torch, skip_unless_allocation_fits


def check_hostlist_file(exp_dir: str, stdout_buffer, num_ranks):
    hostlist = os.path.join(exp_dir, "hpc_launcher_hostlist.txt")
    with open(hostlist) as f:
        s = f.read()
        s = s.strip("]\n")
        cluster_list = re.split(r"[,\s]+", s)
        hosts = []
        for cluster in cluster_list:
            if cluster == "lassen710" and (
                (isinstance(autodetect.autodetect_current_system(), Sierra))
                or os.getenv("LSB_HOSTS")
            ):
                continue

            if "[" in cluster:
                (hostname, inst_array) = cluster.split("[")
                # This only works up to two nodes
                instances = re.split(r"[,-]+", inst_array)
                for i in instances:
                    hosts.append(hostname + i)
            else:
                hosts.append(cluster)

        i = 0
        matched = []
        unmatched = []
        for h in hosts:
            regex = re.compile(
                ".*({}) reporting it is rank ({}) of ({}).*".format(h, i, num_ranks),
                re.MULTILINE | re.DOTALL,
            )
            match = regex.match(stdout_buffer)
            if match:
                assert (
                    match.group(2) == str(i) and match.group(3) == str(num_ranks)
                ), f"{match.group(1)} has the incorrect rank in test {exp_dir}"
                print(
                    f"\n{match.group(1)} is correctly reporting that it was assigned rank {match.group(2)} of {match.group(3)}"
                )
                matched.append(h)
                i += 1
                if i == num_ranks:
                    break
            else:
                unmatched.append(h)
                print(
                    f"{h} not found in output in test {exp_dir} - only {i} found: {matched}"
                )

        assert (
            len(matched) == num_ranks
        ), f"Incorrect number of ranks reported, required {num_ranks} -- matched: {matched} and unmatched: {unmatched}"


@pytest.mark.parametrize("local", [True, False])
def test_launcher_one_node(local):
    require_torch()
    if (
        not local
        and not shutil.which("srun")
        and not shutil.which("flux")
        and not shutil.which("jsrun")
    ):
        pytest.skip("No distributed launcher found")

    # Get full path to torch_dist_driver.py
    if local:
        driver_file = os.path.join(os.path.dirname(__file__), "torch_driver.py")
    else:
        driver_file = os.path.join(os.path.dirname(__file__), "torch_dist_driver.py")

    cmd = [
        sys.executable,
        "-m",
        "hpc_launcher.cli.torchrun_hpc",
        "-l",
        "-v",
        "--local" if local else "-n1",
        "-N1",
        "--save-hostlist",
        driver_file,
    ]
    proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
    exp_dir = None
    m = re.search(r"^.*Script filename: (\S+)$", proc.stderr, re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
        check_hostlist_file(exp_dir, proc.stdout, 1)
    else:
        assert False, f"Unable to find expected hostlist: hpc_launcher_hostlist.txt"

    assert proc.returncode == 0

    if exp_dir:
        shutil.rmtree(exp_dir, ignore_errors=True)


@pytest.mark.parametrize("num_nodes", [2])
@pytest.mark.parametrize("procs_per_node", [1])
@pytest.mark.parametrize("rdv", ["tcp"])
#@pytest.mark.parametrize("rdv", ("mpi", "tcp"))
@pytest.mark.parametrize("scheduler_type", ("flux", "slurm", "lsf"))
def test_launcher_multinode(num_nodes, procs_per_node, rdv, scheduler_type):
    if (
        (scheduler_type == "slurm" and (not shutil.which("srun") or shutil.which("srun") and shutil.which("jsrun")))
        or (
            scheduler_type == "flux"
            and (not shutil.which("flux") or not os.path.exists("/run/flux/local"))
        )
        or (scheduler_type == "lsf" and not shutil.which("jsrun"))
    ):
        pytest.skip("No distributed launcher found")

    skip_unless_allocation_fits(get_schedulers()[scheduler_type], num_nodes)

    require_torch()

    if rdv == "mpi":
        try:
            import mpi4py
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mpi not found")

        try:
            import mpi_rdv
        except (ImportError, ModuleNotFoundError):
            pytest.skip("mpi_rdv not found")

    # Get full path to torch_dist_driver.py
    driver_file = os.path.join(os.path.dirname(__file__), "torch_dist_driver.py")

    cmd = [
        sys.executable,
        "-m",
        "hpc_launcher.cli.torchrun_hpc",
        "-l",
        "-v",
        f"-N{num_nodes}",
        f"-n{procs_per_node}",
        f"-r{rdv}",
        "--save-hostlist",
        driver_file,
    ]
    proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
    exp_dir = None
    m = re.search(r"^.*Script filename: (\S+)$", proc.stderr, re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
        check_hostlist_file(exp_dir, proc.stdout, num_nodes * procs_per_node)
    else:
        assert False, f"Unable to find expected hostlist: hpc_launcher_hostlist.txt"

    regex = re.compile(
        ".*Initializing distributed PyTorch using protocol: ({})://.*".format(rdv),
        re.MULTILINE | re.DOTALL,
    )
    match = regex.match(proc.stdout)
    if match:
        assert (
            match.group(1) == rdv
        ), f"{match.group(1)} is the incorrect rendezvous protocol: requested {rdv}"
    else:
        assert False, f"Unable to detect a valid rendezvous protocol for test {rdv}"
    assert proc.returncode == 0

    if exp_dir:
        shutil.rmtree(exp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_launcher_multinode(2, 1, "tcp", "slurm")
    test_launcher_multinode(2, 1, "tcp", "flux")
    test_launcher_multinode(2, 1, "tcp", "lsf")
    test_launcher_multinode(2, 1, "mpi", "slurm")
    test_launcher_multinode(2, 1, "mpi", "flux")
    test_launcher_multinode(2, 1, "mpi", "lsf")
