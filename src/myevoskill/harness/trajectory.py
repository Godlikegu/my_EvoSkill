"""Trajectory writer.

Records the event kinds we care about for debugging and skill distillation:

    1. ``assistant_text``   - assistant prose
    2. ``assistant_thinking`` - raw SDK thinking blocks, debug only
    3. ``tool_call``         - tool name + sanitised input
    4. ``tool_result``       - stdout/stderr or text returned by the tool
    5. ``env_feedback``      - synthetic message we inject (judge result,
                               plan-guard reminder, hook denial)

Each event is one JSON line, written to ``trajectory.jsonl`` under the run's
log root. The writer is process-safe via a per-instance ``threading.Lock``.

The writer is the *only* place we serialise SDK message objects. Everything
above it speaks plain dicts.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping


class TrajectoryWriter:
    """Append-only JSONL writer for one run."""

    def __init__(self, log_root: Path, run_id: str) -> None:
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.path = self.log_root / "trajectory.jsonl"
        self.run_id = run_id
        self._lock = threading.Lock()
        # Always start fresh; callers can copy old files if they want.
        self.path.write_text("", encoding="utf-8")

    # --------------------------------------------------------------- helpers

    def _emit(self, kind: str, round_index: int, payload: Mapping[str, Any]) -> None:
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "round": int(round_index),
            "kind": kind,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # --------------------------------------------------------------- writers

    def assistant_text(self, round_index: int, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._emit("assistant_text", round_index, {"text": text})

    def assistant_thinking(self, round_index: int, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._emit("assistant_thinking", round_index, {"text": text})

    def tool_call(
        self,
        round_index: int,
        tool_name: str,
        tool_input: Mapping[str, Any],
        tool_use_id: str | None = None,
    ) -> None:
        self._emit(
            "tool_call",
            round_index,
            {
                "tool": tool_name,
                "tool_use_id": tool_use_id,
                "input": _summarise_tool_input(tool_input),
            },
        )

    def tool_result(
        self,
        round_index: int,
        tool_use_id: str | None,
        text: str,
        is_error: bool,
    ) -> None:
        text = text or ""
        if len(text) > 4000:
            text = text[:2000] + f"\n... [{len(text) - 4000} chars truncated] ...\n" + text[-2000:]
        self._emit(
            "tool_result",
            round_index,
            {"tool_use_id": tool_use_id, "text": text, "is_error": bool(is_error)},
        )

    def env_feedback(self, round_index: int, kind: str, payload: Mapping[str, Any]) -> None:
        self._emit("env_feedback", round_index, {"feedback_kind": kind, "payload": dict(payload)})

    def round_marker(self, round_index: int, status: str, payload: Mapping[str, Any]) -> None:
        self._emit("round_marker", round_index, {"status": status, "payload": dict(payload)})

    def read_clean_events(self) -> list[dict[str, Any]]:
        """Return trajectory events suitable as a distillation starting point.

        Raw logs keep all SDK detail for debugging. The clean view removes
        assistant thinking and deduplicates tool calls by ``tool_use_id`` so
        downstream skill distillation does not learn from duplicated action
        records or private scratch reasoning.
        """

        return read_clean_events(self.path)


# --------------------------------------------------------------------- helpers


def _summarise_tool_input(tool_input: Mapping[str, Any]) -> dict[str, Any]:
    """Trim long string fields to keep the trajectory readable."""

    out: dict[str, Any] = {}
    for k, v in dict(tool_input or {}).items():
        if isinstance(v, str) and len(v) > 2000:
            out[k] = v[:1000] + f"\n... [{len(v) - 2000} chars truncated] ...\n" + v[-1000:]
        else:
            out[k] = v
    return out


def read_clean_events(path: Path) -> list[dict[str, Any]]:
    seen_tool_calls: set[str] = set()
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") == "assistant_thinking":
            continue
        if rec.get("kind") == "tool_call":
            tool_use_id = str(rec.get("tool_use_id") or "")
            if tool_use_id:
                if tool_use_id in seen_tool_calls:
                    continue
                seen_tool_calls.add(tool_use_id)
        out.append(rec)
    return out


def write_clean_events(input_path: Path, output_path: Path) -> int:
    """Write the distillation-clean trajectory view as JSONL.

    The raw trajectory remains untouched. The clean view is intentionally
    line-delimited JSON so it can be streamed by downstream distillation jobs.
    """

    events = read_clean_events(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for rec in events:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return len(events)
