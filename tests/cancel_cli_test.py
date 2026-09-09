# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (Apache-2.0)
"""Tests for the hpc-launcher-cancel command."""

import subprocess

import pytest

from hpc_launcher.cli import cancel
from hpc_launcher.schedulers.flux import FluxScheduler
from hpc_launcher.schedulers.local import LocalScheduler
from hpc_launcher.schedulers.lsf import LSFScheduler
from hpc_launcher.schedulers.slurm import SlurmScheduler


@pytest.mark.parametrize(
    ("scheduler_class", "identifier", "expected"),
    [
        (SlurmScheduler, "1234.7", ["scancel", "1234.7"]),
        (FluxScheduler, "f5wWq9", ["flux", "cancel", "f5wWq9"]),
        (LSFScheduler, "98765", ["bkill", "98765"]),
        (LocalScheduler, "4321", ["kill", "-TERM", "4321"]),
    ],
)
def test_scheduler_defines_its_cancel_command(
    scheduler_class, identifier, expected
):
    assert scheduler_class.cancel_command(identifier) == expected


def test_local_scheduler_requires_a_pid():
    with pytest.raises(ValueError, match="must be a process ID"):
        LocalScheduler.cancel_command("not-a-pid")


@pytest.mark.parametrize(
    ("scheduler", "identifier", "expected"),
    [
        ("slurm", "1234.7", ["scancel", "1234.7"]),
        ("flux", "f5wWq9", ["flux", "cancel", "f5wWq9"]),
        ("lsf", "98765", ["bkill", "98765"]),
        ("local", "4321", ["kill", "-TERM", "4321"]),
    ],
)
def test_cancel_uses_native_scheduler_command(
    scheduler, identifier, expected, monkeypatch
):
    calls = []

    def _run(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cancel.subprocess, "run", _run)

    assert cancel.main(["--scheduler", scheduler, identifier]) == 0
    assert calls == [expected]


def test_step_assignment_selects_slurm(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cancel.subprocess,
        "run",
        lambda command: calls.append(command)
        or subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(
        cancel.autodetect,
        "find_scheduler",
        lambda: pytest.fail("an explicit step marker must not autodetect"),
    )

    assert cancel.main(["HPC_LAUNCHER_STEP_ID=1234.7"]) == 0
    assert calls == [["scancel", "1234.7"]]


def test_identifier_can_come_from_environment(monkeypatch):
    calls = []
    monkeypatch.setenv("HPC_LAUNCHER_JOB_ID", "f5wWq9")
    monkeypatch.delenv("HPC_LAUNCHER_STEP_ID", raising=False)
    monkeypatch.setattr(
        cancel.autodetect,
        "find_scheduler",
        lambda: pytest.fail("an HPC_LAUNCHER_JOB_ID marker implies Flux"),
    )
    monkeypatch.setattr(
        cancel.subprocess,
        "run",
        lambda command: calls.append(command)
        or subprocess.CompletedProcess(command, 0),
    )

    assert cancel.main([]) == 0
    assert calls == [["flux", "cancel", "f5wWq9"]]


def test_job_assignment_selects_flux(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cancel.subprocess,
        "run",
        lambda command: calls.append(command)
        or subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(
        cancel.autodetect,
        "find_scheduler",
        lambda: pytest.fail("an explicit job marker must not autodetect"),
    )

    assert cancel.main(["HPC_LAUNCHER_JOB_ID=f5wWq9"]) == 0
    assert calls == [["flux", "cancel", "f5wWq9"]]


def test_native_failure_is_returned(monkeypatch):
    monkeypatch.setattr(
        cancel.subprocess,
        "run",
        lambda command: subprocess.CompletedProcess(command, 9),
    )
    assert cancel.main(["--scheduler", "lsf", "1234"]) == 9


@pytest.mark.parametrize("identifier", ["--all", "two ids", ""])
def test_unsafe_or_empty_identifiers_are_rejected(identifier):
    with pytest.raises(SystemExit) as error:
        cancel.main(["--scheduler", "flux", identifier])
    assert error.value.code == 2


def test_step_marker_cannot_be_sent_to_wrong_scheduler():
    with pytest.raises(SystemExit) as error:
        cancel.main(
            ["--scheduler", "flux", "HPC_LAUNCHER_STEP_ID=1234.7"]
        )
    assert error.value.code == 2


def test_missing_native_command_returns_127(monkeypatch, capsys):
    def _missing(command):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(cancel.subprocess, "run", _missing)
    assert cancel.main(["--scheduler", "lsf", "1234"]) == 127
    assert "command not found: bkill" in capsys.readouterr().err
