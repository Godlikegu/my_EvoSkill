"""Best-effort cleanup for Claude Code child process trees."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    name: str
    command_line: str


@dataclass
class ReapResult:
    killed: list[ProcessInfo] = field(default_factory=list)
    remaining: list[ProcessInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.remaining and not self.errors

    def to_dict(self) -> dict:
        return {
            "killed": [_proc_to_dict(p) for p in self.killed],
            "remaining": [_proc_to_dict(p) for p in self.remaining],
            "errors": list(self.errors),
        }


def reap_descendant_processes(
    *,
    root_pid: int | None,
    markers: Sequence[str],
    grace_seconds: float = 1.0,
    kill: bool = True,
) -> ReapResult:
    """Kill Bash/Python descendants of *root_pid* whose lineage matches markers.

    The root process itself is never killed. A descendant is eligible only when
    it or one of its descendants/ancestors under the root mentions the current
    run marker (run id, workspace path, or sandbox path). This keeps the sweep
    scoped to the active harness run.
    """

    result = ReapResult()
    if root_pid is None:
        return result

    marker_set = _normalise_markers(markers)
    if not marker_set:
        return result

    try:
        table = {p.pid: p for p in list_processes()}
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"process table unavailable: {exc!r}")
        return result

    descendants = _descendants(root_pid, table)
    candidates = _matching_process_subtree(root_pid, table, descendants, marker_set)
    if not candidates:
        return result

    if kill:
        for proc in sorted(candidates, key=lambda p: _depth(p.pid, table), reverse=True):
            if proc.pid == os.getpid():
                continue
            try:
                os.kill(proc.pid, signal.SIGTERM)
                result.killed.append(proc)
            except ProcessLookupError:
                result.killed.append(proc)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"failed to terminate {proc.pid}: {exc!r}")

        if grace_seconds > 0:
            time.sleep(grace_seconds)

    try:
        after = {p.pid: p for p in list_processes()}
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"post-kill process table unavailable: {exc!r}")
        return result

    still_desc = _descendants(root_pid, after)
    remaining = _matching_process_subtree(root_pid, after, still_desc, marker_set)
    result.remaining = sorted(remaining, key=lambda p: p.pid)
    return result


def list_processes() -> list[ProcessInfo]:
    if os.name == "nt":
        return _list_processes_windows()
    return _list_processes_posix()


def make_run_markers(*, run_id: str, workspace_root: Path, sandbox_root: Path) -> list[str]:
    raw = [run_id, str(workspace_root), str(sandbox_root)]
    out: list[str] = []
    for item in raw:
        if item and item not in out:
            out.append(item)
        slash = item.replace("\\", "/")
        if slash and slash not in out:
            out.append(slash)
    return out


def _list_processes_windows() -> list[ProcessInfo]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
        "ConvertTo-Json -Compress",
    ]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = completed.stdout.strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    return [
        ProcessInfo(
            pid=int(row.get("ProcessId") or 0),
            ppid=int(row.get("ParentProcessId") or 0),
            name=str(row.get("Name") or ""),
            command_line=str(row.get("CommandLine") or ""),
        )
        for row in data
        if int(row.get("ProcessId") or 0) > 0
    ]


def _list_processes_posix() -> list[ProcessInfo]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,comm=,args="],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out: list[ProcessInfo] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        pid, ppid, name = parts[:3]
        command_line = parts[3] if len(parts) > 3 else name
        try:
            out.append(ProcessInfo(int(pid), int(ppid), name, command_line))
        except ValueError:
            continue
    return out


def _descendants(root_pid: int, table: dict[int, ProcessInfo]) -> set[int]:
    children: dict[int, list[int]] = {}
    for proc in table.values():
        children.setdefault(proc.ppid, []).append(proc.pid)
    seen: set[int] = set()
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen


def _matching_process_subtree(
    root_pid: int,
    table: dict[int, ProcessInfo],
    descendants: Iterable[int],
    markers: set[str],
) -> set[ProcessInfo]:
    descendant_set = set(descendants)
    matched_lineage: set[int] = set()
    for pid in descendant_set:
        if _matches_markers(table.get(pid), markers):
            matched_lineage.update(_lineage_to_root(pid, root_pid, table))

    out: set[ProcessInfo] = set()
    for pid in descendant_set:
        proc = table.get(pid)
        if proc is None:
            continue
        lineage = _lineage_to_root(pid, root_pid, table)
        if matched_lineage.intersection(lineage) and _is_bash_or_python(proc):
            out.add(proc)
    return out


def _lineage_to_root(pid: int, root_pid: int, table: dict[int, ProcessInfo]) -> list[int]:
    lineage: list[int] = []
    cur = pid
    while cur and cur != root_pid and cur in table:
        lineage.append(cur)
        cur = table[cur].ppid
    return lineage


def _depth(pid: int, table: dict[int, ProcessInfo]) -> int:
    depth = 0
    cur = pid
    seen: set[int] = set()
    while cur in table and cur not in seen:
        seen.add(cur)
        depth += 1
        cur = table[cur].ppid
    return depth


def _normalise_markers(markers: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for marker in markers:
        marker = str(marker or "").strip()
        if not marker:
            continue
        out.add(marker.lower())
        out.add(marker.replace("\\", "/").lower())
    return out


def _matches_markers(proc: ProcessInfo | None, markers: set[str]) -> bool:
    if proc is None:
        return False
    haystack = f"{proc.name} {proc.command_line}".replace("\\", "/").lower()
    return any(marker in haystack for marker in markers)


def _is_bash_or_python(proc: ProcessInfo) -> bool:
    text = f"{proc.name} {proc.command_line}".lower()
    return any(token in text for token in ("bash", "sh ", "python"))


def _proc_to_dict(proc: ProcessInfo) -> dict:
    cmd = proc.command_line
    if len(cmd) > 500:
        cmd = cmd[:250] + " ... " + cmd[-250:]
    return {
        "pid": proc.pid,
        "ppid": proc.ppid,
        "name": proc.name,
        "command_line": cmd,
    }
