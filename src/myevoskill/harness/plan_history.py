"""Per-round plan.md snapshotting (commit G).

Why
====

The harness already enforces that the agent must edit ``plan.md`` before
each new code modification (see :mod:`plan_guard`). However the *content*
of ``plan.md`` is not retained anywhere outside the live workspace -- on a
PASS we wipe the workspace, and even on FAIL the file only ever holds the
*latest* round, because the agent rewrites it in place.

For reflection / failure-mode audits we want the full history: what did
the agent claim to be doing in round 1 vs round 2 vs ...? This module
takes a snapshot of ``plan.md`` per round into the run's log directory:

    artifacts/logs/<task>/<run_id>/
        plan_round_01.md
        plan_round_02.md
        ...
        plan_history.jsonl

``plan_history.jsonl`` is one JSON object per round with metadata
(``round``, ``timestamp``, ``size_bytes``, ``sha256``, ``snapshot_path``,
``diff_lines``) so downstream tooling does not have to re-read every md
file just to count or hash them. The diff is line-based vs the previous
round's snapshot (for round 1 it reads "(initial round)").

The recorder is deliberately *write-only*. It does not parse plan.md or
try to extract round headers; it just preserves the raw bytes the agent
wrote at the moment the round closed.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


PLAN_FILENAME = "plan.md"
HISTORY_FILENAME = "plan_history.jsonl"


@dataclass(frozen=True)
class PlanSnapshot:
    """One round's snapshot record."""

    round_index: int
    timestamp: float
    snapshot_path: Path
    size_bytes: int
    sha256: str
    diff_lines: int
    note: str  # human-readable status: "ok", "missing", "empty", "unchanged"


class PlanHistoryRecorder:
    """Persist per-round copies of ``plan.md`` into the run log directory.

    Parameters
    ----------
    workspace_root:
        The agent's workspace root. ``workspace_root / "plan.md"`` is the
        live file edited by the agent through the SDK.
    log_root:
        The run's log directory. Snapshots are written into it and the
        ``plan_history.jsonl`` index sits alongside.
    """

    def __init__(self, *, workspace_root: Path, log_root: Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self._history_path = self.log_root / HISTORY_FILENAME
        # Cache of the previous snapshot's text so we can compute a diff
        # without re-reading the previous file. `None` until first snapshot.
        self._prev_text: str | None = None

    # ------------------------------------------------------------------ paths

    @property
    def history_path(self) -> Path:
        return self._history_path

    def snapshot_path_for(self, round_index: int) -> Path:
        return self.log_root / f"plan_round_{int(round_index):02d}.md"

    # ------------------------------------------------------------------ snap

    def snapshot(self, round_index: int) -> PlanSnapshot:
        """Copy the current plan.md into the log dir and append history.

        Always emits a record, even if plan.md is missing or unchanged --
        the reflection audit relies on a contiguous per-round series.
        """

        live_path = self.workspace_root / PLAN_FILENAME
        if not live_path.exists():
            text = ""
            note = "missing"
        else:
            try:
                text = live_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:  # pragma: no cover - filesystem oddity
                text = ""
                note = f"read_error:{exc!r}"
            else:
                note = "ok" if text.strip() else "empty"

        # Did the file change since last round?
        prev = self._prev_text
        if prev is not None and prev == text and note == "ok":
            note = "unchanged"

        snapshot_path = self.snapshot_path_for(round_index)
        snapshot_path.write_text(text, encoding="utf-8")

        sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        diff_lines = _count_diff_lines(prev or "", text)

        record = PlanSnapshot(
            round_index=int(round_index),
            timestamp=time.time(),
            snapshot_path=snapshot_path,
            size_bytes=len(text.encode("utf-8", errors="replace")),
            sha256=sha,
            diff_lines=diff_lines,
            note=note,
        )

        # Append to plan_history.jsonl atomically (one line per round).
        line = json.dumps(
            {
                "round": record.round_index,
                "timestamp": record.timestamp,
                "snapshot_path": str(snapshot_path.relative_to(self.log_root)),
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "diff_lines": record.diff_lines,
                "note": record.note,
            },
            ensure_ascii=False,
        )
        with self._history_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        self._prev_text = text
        return record

    # ----------------------------------------------------------------- read

    def read_history(self) -> list[dict[str, object]]:
        """Return the parsed history. Useful for tests + reflection tools."""

        if not self._history_path.exists():
            return []
        out: list[dict[str, object]] = []
        for raw in self._history_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out


# --------------------------------------------------------------------- helpers


def _count_diff_lines(old: str, new: str) -> int:
    """Number of unified-diff lines (additions + removals) between old/new.

    A pure additive append of N lines yields N. Identical inputs yield 0.
    """

    if old == new:
        return 0
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, n=0, lineterm="")
    count = 0
    for line in diff:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count
