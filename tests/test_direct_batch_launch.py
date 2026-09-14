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
import socket

from hpc_launcher.systems import autodetect
from hpc_launcher.systems.lc.sierra_family import Sierra
from hpc_launcher.schedulers import get_schedulers

from conftest import skip_unless_allocation_fits

def check_num_lines(stdout_buffer, num_ranks):
    count = sum(1 for line in stdout_buffer.splitlines() if line.strip())
    assert (
        count == num_ranks
    ), f"Incorrect number of ranks reported, required {num_ranks} -- matched: {count}"


def check_num_hosts(stdout_buffer, num_ranks):
        hostname = socket.gethostname()
        if not re.search(r'^[A-Za-z]+\d+$', hostname): # Check to see if this is a standard hostnameXXXX style
            return
        
        host = re.sub(r'\d+', '', hostname)
        regex = re.compile(
            ".*({}\\d+).*".format(host),
            re.MULTILINE,
        )
        matches = regex.findall(stdout_buffer)
        assert (
            len(matches) == num_ranks
        ), f"Incorrect number of ranks reported, required {num_ranks} -- matched: {matches}"


@pytest.mark.parametrize("local", [True, False])
def test_launcher_one_node(local):
    if (
        not local
        and not shutil.which("srun")
        and not shutil.which("flux")
        and not shutil.which("jsrun")
    ):
        pytest.skip("No distributed launcher found")

    # Get full path to test_batch_script.py
    driver_file = os.path.join(os.path.dirname(__file__), "test_batch_script.sh")

    cmd = [
        sys.executable,
        "-m",
        "hpc_launcher.cli.launch",
        "-v",
        "--local" if local else "-n1",
        "-N1",
        "-l",
        "--batch-script",
        driver_file,
    ]
    proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
    check_num_lines(proc.stdout, 1)
    check_num_hosts(proc.stdout, 1)
    exp_dir = None
    m = re.search(r"^.*Script filename: (\S+)$", proc.stderr, re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
    else:
        assert False, f"Unable to find expected launch directory"

    assert proc.returncode == 0

    if exp_dir:
        shutil.rmtree(exp_dir, ignore_errors=True)


@pytest.mark.parametrize("num_nodes", [2])
@pytest.mark.parametrize("procs_per_node", [1])
@pytest.mark.parametrize("scheduler_type", ("flux", "slurm", "lsf"))
def test_launcher_multinode(num_nodes, procs_per_node, scheduler_type):
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

    # Get full path to torch_dist_driver.py
    driver_file = os.path.join(os.path.dirname(__file__), "test_batch_script.sh")

    cmd = [
        sys.executable,
        "-m",
        "hpc_launcher.cli.launch",
        "-v",
        f"-N{num_nodes}",
        f"-n{procs_per_node}",
        "-l",
        "--batch-script",
        driver_file,
    ]
    proc = subprocess.run(cmd, universal_newlines=True, capture_output=True)
    check_num_lines(proc.stdout, num_nodes)
    check_num_hosts(proc.stdout, num_nodes)
    exp_dir = None
    m = re.search(r"^.*Script filename: (\S+)$", proc.stderr, re.MULTILINE | re.DOTALL)
    if m:
        script = m.group(1)
        exp_dir = os.path.dirname(script)
    else:
        assert False, f"Unable to find expected launch directory: {proc.stderr}"

    if exp_dir:
        shutil.rmtree(exp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_launcher_one_node(True)
    test_launcher_one_node(False)
    test_launcher_multinode(2, 1, "slurm")
    test_launcher_multinode(2, 1, "flux")
    test_launcher_multinode(2, 1, "lsf")
