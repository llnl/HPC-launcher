# Copyright (c) 2014-2026, Lawrence Livermore National Security, LLC.
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


def test_torchrun_hpc_relimport():
    scheduler_type = "slurm"
    if ((scheduler_type == "slurm" and
         (not shutil.which("srun")
          or shutil.which("srun") and shutil.which("jsrun"))) or
        (scheduler_type == "flux" and
         (not shutil.which("flux") or not os.path.exists("/run/flux/local")))
            or (scheduler_type == "lsf" and not shutil.which("jsrun"))):
        pytest.skip("No distributed launcher found")

    skip_unless_allocation_fits(get_schedulers()[scheduler_type], 1)

    require_torch()

    cmd = [
        sys.executable,
        "-m",
        "hpc_launcher.cli.torchrun_hpc",
        "-l",
        "-v",
        "-N",
        "1",
        "-n",
        "1",
        "-m",
        "relimport",
        "4.75",
    ]
    cwd = os.path.join(os.path.dirname(__file__), "e2e")
    proc = subprocess.run(cmd,
                          universal_newlines=True,
                          capture_output=True,
                          cwd=cwd)
    exp_dir = None

    result = proc.stdout.splitlines()[-1].strip()
    assert proc.returncode == 0
    assert result == "8"

    if exp_dir:
        shutil.rmtree(exp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_torchrun_hpc_relimport()
