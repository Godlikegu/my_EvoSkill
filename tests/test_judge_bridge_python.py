"""Tests for ``JudgeRunner``'s python interpreter selection.

The judge subprocess MUST run in the per-task venv when the manifest
declares one (so its dependencies match what the agent code was written
against), and fall back to the harness ``sys.executable`` otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

from myevoskill.judge.bridge import JudgeRunner


def _base_manifest(**runtime_env) -> dict:
    return {
        "task_id": "demo",
        "source_task_dir": "tasks/demo",
        "primary_output_path": "output/x.npz",
        "judge_adapter_path": "evaluation/judge_adapter.py",
        "runtime_env": runtime_env,
    }


def test_uses_manifest_python_when_present(tmp_path: Path):
    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    runner = JudgeRunner(
        repo_root=tmp_path,
        manifest=_base_manifest(
            backend="per_task_venv",
            ready=True,
            python_executable=str(fake_python),
        ),
        log_root=tmp_path / "logs",
    )
    assert runner.python_executable == str(fake_python)


def test_falls_back_when_manifest_python_missing(tmp_path: Path):
    runner = JudgeRunner(
        repo_root=tmp_path,
        manifest=_base_manifest(
            backend="harness_python_fallback",
            ready=True,
            python_executable="",
        ),
        log_root=tmp_path / "logs",
    )
    assert runner.python_executable == sys.executable


def test_falls_back_when_manifest_python_does_not_exist(tmp_path: Path):
    """A stale manifest pointing at a nuked venv must NOT crash the
    harness; we silently fall back to sys.executable."""
    runner = JudgeRunner(
        repo_root=tmp_path,
        manifest=_base_manifest(
            backend="per_task_venv",
            ready=True,
            python_executable=str(tmp_path / "does_not_exist" / "python"),
        ),
        log_root=tmp_path / "logs",
    )
    assert runner.python_executable == sys.executable


def test_explicit_argument_overrides_manifest(tmp_path: Path):
    fake_python = tmp_path / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    runner = JudgeRunner(
        repo_root=tmp_path,
        manifest=_base_manifest(
            backend="per_task_venv",
            ready=True,
            python_executable=str(fake_python),
        ),
        log_root=tmp_path / "logs",
        python_executable="/explicit/override/python",
    )
    assert runner.python_executable == "/explicit/override/python"
