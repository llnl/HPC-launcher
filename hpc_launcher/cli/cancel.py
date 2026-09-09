# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# SPDX-License-Identifier: (Apache-2.0)
"""Cancel a job or job step through its native scheduler."""

import argparse
import os
import subprocess
import sys
from typing import Optional

from hpc_launcher.schedulers import get_schedulers
from hpc_launcher.systems import autodetect


SCHEDULERS = ("flux", "slurm", "lsf", "local")
_IDENTITY_VARIABLES = ("HPC_LAUNCHER_STEP_ID", "HPC_LAUNCHER_JOB_ID")


def _parse_identifier(
    parser: argparse.ArgumentParser, value: Optional[str]
) -> tuple[str, Optional[str]]:
    """Resolve an ID value and any scheduler implied by its marker name."""
    implied_scheduler = None

    if value in _IDENTITY_VARIABLES:
        variable = value
        value = os.getenv(variable)
        if value is None:
            parser.error(f"{variable} is not set")
        implied_scheduler = (
            "slurm" if variable == "HPC_LAUNCHER_STEP_ID" else "flux"
        )
    elif value and "=" in value:
        variable, assigned_value = value.split("=", 1)
        if variable not in _IDENTITY_VARIABLES:
            parser.error(
                "identifier assignments must use HPC_LAUNCHER_STEP_ID or "
                "HPC_LAUNCHER_JOB_ID"
            )
        value = assigned_value
        implied_scheduler = (
            "slurm" if variable == "HPC_LAUNCHER_STEP_ID" else "flux"
        )
    elif value is None:
        present = [(name, os.getenv(name)) for name in _IDENTITY_VARIABLES]
        present = [(name, item) for name, item in present if item]
        if not present:
            parser.error(
                "an identifier is required (or export HPC_LAUNCHER_STEP_ID/"
                "HPC_LAUNCHER_JOB_ID)"
            )
        if len(present) > 1:
            parser.error(
                "both HPC_LAUNCHER_STEP_ID and HPC_LAUNCHER_JOB_ID are set; "
                "pass the intended identifier explicitly"
            )
        variable, value = present[0]
        implied_scheduler = (
            "slurm" if variable == "HPC_LAUNCHER_STEP_ID" else "flux"
        )

    # Do not let an identifier become an option to the native command.
    # Scheduler IDs never contain whitespace or start with a dash.
    if not value or value.startswith("-") or any(c.isspace() for c in value):
        parser.error(f"invalid job identifier: {value!r}")

    return value, implied_scheduler


def _select_scheduler(
    parser: argparse.ArgumentParser,
    requested: Optional[str],
    implied: Optional[str],
) -> str:
    if requested and implied and requested != implied:
        parser.error(
            f"the identifier implies scheduler {implied!r}, but "
            f"--scheduler requested {requested!r}"
        )
    if requested:
        return requested
    if implied:
        return implied

    detected = autodetect.find_scheduler()
    if detected in SCHEDULERS:
        return detected
    parser.error("could not detect a scheduler; specify --scheduler")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hpc-launcher-cancel",
        description="Cancel an HPC Launcher job or job step.",
    )
    parser.add_argument(
        "identifier",
        nargs="?",
        help=(
            "raw scheduler ID, HPC_LAUNCHER_* variable name, or "
            "HPC_LAUNCHER_*=ID assignment"
        ),
    )
    parser.add_argument(
        "--scheduler",
        "-s",
        choices=SCHEDULERS,
        help="scheduler backend (autodetected when omitted)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print the native cancellation command",
    )
    args = parser.parse_args(argv)

    identifier, implied = _parse_identifier(parser, args.identifier)
    scheduler = _select_scheduler(parser, args.scheduler, implied)

    scheduler_class = get_schedulers()[scheduler]
    try:
        command = scheduler_class.cancel_command(identifier)
    except ValueError as error:
        parser.error(str(error))
    if args.verbose:
        print("Cancelling with: " + " ".join(command), file=sys.stderr)

    try:
        return subprocess.run(command).returncode
    except FileNotFoundError:
        print(
            f"hpc-launcher-cancel: cancellation command not found: {command[0]}",
            file=sys.stderr,
        )
        return 127


if __name__ == "__main__":
    sys.exit(main())
