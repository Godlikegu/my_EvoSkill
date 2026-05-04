"""Hard isolation tests for DistillUniverse.

These tests ensure that *any* attempt to read a valid-split task raises
``ValidationLeakError``, even when the underlying file exists. They are
the contract test backing the user requirement: "we may distill skills
freely as long as no valid-task code is read."
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myevoskill.distill.universe import DistillUniverse, ValidationLeakError


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """A fake repo layout matching ``MyEvoSkill/`` next to ``tasks/``."""
    repo = tmp_path / "MyEvoSkill"
    (repo / "artifacts" / "logs" / "Model" / "TrainTask" / "run-1").mkdir(parents=True)
    (repo / "artifacts" / "logs" / "Model" / "ValidTask" / "run-1").mkdir(parents=True)
    (tmp_path / "tasks" / "TrainTask").mkdir(parents=True)
    (tmp_path / "tasks" / "ValidTask").mkdir(parents=True)
    (tmp_path / "tasks" / "TrainTask" / "src").mkdir()
    (tmp_path / "tasks" / "ValidTask" / "src").mkdir()
    (tmp_path / "tasks" / "TrainTask" / "src" / "main.py").write_text(
        "# train reference\nprint('train')\n", encoding="utf-8"
    )
    (tmp_path / "tasks" / "ValidTask" / "src" / "main.py").write_text(
        "# valid reference - SECRET\nprint('valid')\n", encoding="utf-8"
    )
    (
        repo / "artifacts" / "logs" / "Model" / "TrainTask" / "run-1" / "trajectory.jsonl"
    ).write_text('{"kind":"assistant_text","text":"hi"}\n', encoding="utf-8")
    (
        repo / "artifacts" / "logs" / "Model" / "ValidTask" / "run-1" / "trajectory.jsonl"
    ).write_text('{"kind":"assistant_text","text":"valid"}\n', encoding="utf-8")
    return repo


@pytest.fixture
def universe(tmp_repo: Path) -> DistillUniverse:
    return DistillUniverse(
        repo_root=tmp_repo,
        train_task_ids=("TrainTask",),
        valid_task_ids=("ValidTask",),
        model_slug="Model",
    )


# ---------------------------------------------------------- positive (allowed)


def test_read_train_task_src_ok(universe: DistillUniverse) -> None:
    text = universe.read_task_file("TrainTask", "src/main.py")
    assert "train" in text


def test_list_train_runs_ok(universe: DistillUniverse) -> None:
    runs = universe.list_runs("TrainTask")
    assert len(runs) == 1
    assert runs[0].name == "run-1"


def test_read_train_log_ok(universe: DistillUniverse) -> None:
    runs = universe.list_runs("TrainTask")
    text = universe.read_log_file("TrainTask", runs[0], "trajectory.jsonl")
    assert "assistant_text" in text


# ---------------------------------------------------------- negative (denied)


def test_read_valid_task_src_denied(universe: DistillUniverse) -> None:
    with pytest.raises(ValidationLeakError):
        universe.read_task_file("ValidTask", "src/main.py")


def test_list_valid_runs_denied(universe: DistillUniverse) -> None:
    with pytest.raises(ValidationLeakError):
        universe.list_runs("ValidTask")


def test_read_valid_log_denied(universe: DistillUniverse, tmp_repo: Path) -> None:
    fake_run = tmp_repo / "artifacts" / "logs" / "Model" / "ValidTask" / "run-1"
    with pytest.raises(ValidationLeakError):
        universe.read_log_file("ValidTask", fake_run, "trajectory.jsonl")


def test_unknown_task_denied(universe: DistillUniverse) -> None:
    with pytest.raises(ValidationLeakError):
        universe.read_task_file("NotInAnySplit", "src/main.py")


def test_path_escape_denied(universe: DistillUniverse, tmp_repo: Path) -> None:
    """A relative path that points outside the train task dir must be denied."""
    with pytest.raises(ValidationLeakError):
        universe.read_task_file("TrainTask", "../ValidTask/src/main.py")


# ---------------------------------------------------------- audit


def test_audit_log_records_attempts(
    universe: DistillUniverse, tmp_path: Path
) -> None:
    audit = tmp_path / "audit.jsonl"
    universe.bind_audit_log(audit)

    universe.read_task_file("TrainTask", "src/main.py")
    with pytest.raises(ValidationLeakError):
        universe.read_task_file("ValidTask", "src/main.py")

    lines = [json.loads(l) for l in audit.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["allowed"] is True and lines[0]["task_id"] == "TrainTask"
    assert lines[1]["allowed"] is False and lines[1]["task_id"] == "ValidTask"


def test_assert_no_valid_access_clean(universe: DistillUniverse) -> None:
    universe.read_task_file("TrainTask", "src/main.py")
    universe.assert_no_valid_access()  # must not raise


def test_assert_no_valid_access_dirty_after_attempt(
    universe: DistillUniverse,
) -> None:
    """A *denied* valid access does not constitute a leak (file content
    never crossed the boundary), so assert_no_valid_access must still pass.
    """
    with pytest.raises(ValidationLeakError):
        universe.read_task_file("ValidTask", "src/main.py")
    universe.assert_no_valid_access()  # denied attempts are OK
