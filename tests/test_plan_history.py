"""Commit G: per-round plan.md snapshot recorder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myevoskill.harness.plan_history import (
    HISTORY_FILENAME,
    PlanHistoryRecorder,
    _count_diff_lines,
)


# ------------------------------------------------------------ helpers


def _setup(tmp_path: Path) -> tuple[Path, Path, PlanHistoryRecorder]:
    workspace = tmp_path / "workspace"
    log_root = tmp_path / "log"
    workspace.mkdir()
    log_root.mkdir()
    rec = PlanHistoryRecorder(workspace_root=workspace, log_root=log_root)
    return workspace, log_root, rec


# ------------------------------------------------------------ snapshot tests


def test_snapshot_writes_per_round_md_and_appends_history(tmp_path: Path) -> None:
    workspace, log_root, rec = _setup(tmp_path)

    plan = workspace / "plan.md"
    plan.write_text("## Round 1\n- hyp: try FBP\n", encoding="utf-8")
    snap1 = rec.snapshot(1)

    plan.write_text("## Round 1\n- hyp: try FBP\n## Round 2\n- hyp: try iter\n", encoding="utf-8")
    snap2 = rec.snapshot(2)

    assert (log_root / "plan_round_01.md").read_text(encoding="utf-8").startswith("## Round 1")
    assert "## Round 2" in (log_root / "plan_round_02.md").read_text(encoding="utf-8")
    assert snap1.note == "ok"
    assert snap2.note == "ok"
    assert snap1.diff_lines >= 1
    # Round 2 added 2 lines (heading + bullet) on top of round 1.
    assert snap2.diff_lines >= 2

    history_path = log_root / HISTORY_FILENAME
    lines = [json.loads(l) for l in history_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["round"] for r in lines] == [1, 2]
    assert lines[0]["snapshot_path"] == "plan_round_01.md"
    assert lines[1]["snapshot_path"] == "plan_round_02.md"
    assert lines[0]["sha256"] != lines[1]["sha256"]


def test_snapshot_handles_missing_plan(tmp_path: Path) -> None:
    workspace, log_root, rec = _setup(tmp_path)

    # No plan.md ever created.
    snap = rec.snapshot(1)
    assert snap.note == "missing"
    assert snap.size_bytes == 0
    # An empty placeholder file is still emitted so the per-round series
    # is contiguous.
    assert (log_root / "plan_round_01.md").exists()
    assert (log_root / "plan_round_01.md").read_text(encoding="utf-8") == ""


def test_snapshot_marks_unchanged(tmp_path: Path) -> None:
    workspace, log_root, rec = _setup(tmp_path)
    (workspace / "plan.md").write_text("frozen plan\n", encoding="utf-8")
    s1 = rec.snapshot(1)
    s2 = rec.snapshot(2)
    assert s1.note == "ok"
    assert s2.note == "unchanged"
    assert s2.diff_lines == 0
    assert s1.sha256 == s2.sha256


def test_snapshot_marks_empty(tmp_path: Path) -> None:
    workspace, log_root, rec = _setup(tmp_path)
    (workspace / "plan.md").write_text("   \n\n", encoding="utf-8")
    s = rec.snapshot(1)
    assert s.note == "empty"


def test_read_history_round_trip(tmp_path: Path) -> None:
    workspace, log_root, rec = _setup(tmp_path)
    (workspace / "plan.md").write_text("a\n", encoding="utf-8")
    rec.snapshot(1)
    (workspace / "plan.md").write_text("a\nb\n", encoding="utf-8")
    rec.snapshot(2)
    history = rec.read_history()
    assert len(history) == 2
    assert history[0]["round"] == 1
    assert history[1]["round"] == 2
    assert history[0]["sha256"] != history[1]["sha256"]
    # Snapshot path is recorded relative to the log_root for portability.
    assert "/" not in history[0]["snapshot_path"].replace("\\", "/").lstrip("./").split("/", 1)[0]


def test_history_path_property(tmp_path: Path) -> None:
    workspace, log_root, rec = _setup(tmp_path)
    assert rec.history_path == log_root / HISTORY_FILENAME


def test_snapshot_path_for_zero_pads(tmp_path: Path) -> None:
    workspace, log_root, rec = _setup(tmp_path)
    assert rec.snapshot_path_for(1).name == "plan_round_01.md"
    assert rec.snapshot_path_for(12).name == "plan_round_12.md"


def test_handles_non_utf8_bytes_gracefully(tmp_path: Path) -> None:
    workspace, log_root, rec = _setup(tmp_path)
    # Write some bytes that are not valid UTF-8. The recorder should
    # decode with errors="replace" and not crash.
    (workspace / "plan.md").write_bytes(b"hello \xff world\n")
    snap = rec.snapshot(1)
    assert snap.note == "ok"
    text = (log_root / "plan_round_01.md").read_text(encoding="utf-8")
    assert "hello" in text and "world" in text


# ------------------------------------------------------------ diff helper


def test_count_diff_lines_zero_when_identical() -> None:
    assert _count_diff_lines("a\nb\n", "a\nb\n") == 0


def test_count_diff_lines_pure_addition() -> None:
    # Adding 3 lines to an empty file -> 3 diff lines (all '+').
    assert _count_diff_lines("", "a\nb\nc\n") == 3


def test_count_diff_lines_replacement_counts_both_sides() -> None:
    # Replacing one line counts as 1 removal + 1 addition = 2.
    assert _count_diff_lines("a\nb\nc\n", "a\nB\nc\n") == 2
