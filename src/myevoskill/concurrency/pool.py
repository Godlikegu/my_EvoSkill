"""Run multiple tasks concurrently as isolated subprocesses.

Why subprocesses (not threads or asyncio.gather)?
    1. Each task gets its *own* Claude Agent SDK session and its own Node.js
       claude-code CLI subprocess; running them in the same Python process
       is supported but failure isolation is much weaker.
    2. The Claude CLI persists conversation history under
       ``~/.claude/projects/...``. By spawning a child Python with a
       per-run ``HOME`` we can wipe that directory at the end without
       touching the user's other sessions.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


@dataclass
class SubprocessOutcome:
    """Result of one subprocess task run, parsed from its stdout."""

    task_id: str
    success: bool
    verdict: str
    runtime_seconds: float
    stdout_tail: str
    stderr_tail: str
    summary_path: str | None
    error: str | None


def run_tasks_parallel(
    *,
    repo_root: Path,
    task_ids: Iterable[str],
    max_workers: int = 2,
    extra_run_args: Mapping[str, Any] | None = None,
    timeout_seconds: int = 7800,  # a bit more than 2h to allow judge cleanup
) -> list[SubprocessOutcome]:
    """Run all *task_ids* concurrently, each in its own subprocess.

    Returns a list of :class:`SubprocessOutcome`, one per task.
    """

    repo_root = Path(repo_root).resolve()
    task_ids = list(task_ids)
    extra = dict(extra_run_args or {})

    outcomes: list[SubprocessOutcome] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_one_subprocess, repo_root, tid, extra, timeout_seconds): tid
            for tid in task_ids
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                outcomes.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.exception("subprocess outer failure for %s", tid)
                outcomes.append(
                    SubprocessOutcome(
                        task_id=tid,
                        success=False,
                        verdict="ERROR",
                        runtime_seconds=0.0,
                        stdout_tail="",
                        stderr_tail="",
                        summary_path=None,
                        error=repr(exc),
                    )
                )
    return outcomes


# --------------------------------------------------------------------- worker


def _run_one_subprocess(
    repo_root: Path,
    task_id: str,
    extra_run_args: Mapping[str, Any],
    timeout_seconds: int,
) -> SubprocessOutcome:
    """Spawn one ``python -m myevoskill.cli run`` for *task_id*."""

    # Per-run temp HOME to isolate the claude CLI's history.
    tmp_home = Path(tempfile.mkdtemp(prefix=f"myevoskill_home_{task_id}_"))
    env = os.environ.copy()
    env["HOME"] = str(tmp_home)
    env["USERPROFILE"] = str(tmp_home)  # Windows
    env["PYTHONIOENCODING"] = "utf-8"
    # Pin our own myevoskill on PYTHONPATH.
    env["PYTHONPATH"] = os.pathsep.join(
        [str((repo_root / "src").resolve()), env.get("PYTHONPATH", "")]
    )

    cmd = [
        sys.executable,
        "-m",
        "myevoskill.cli",
        "run-task",
        "--task-id",
        task_id,
        "--repo-root",
        str(repo_root),
        "--json",
    ]
    for k, v in extra_run_args.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])

    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=str(repo_root),
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        ok = completed.returncode == 0
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if exc.stdout else ""
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        ok = False
    finally:
        # Always nuke the per-run HOME so we don't bloat user disk with
        # claude conversation caches.
        shutil.rmtree(tmp_home, ignore_errors=True)

    runtime = time.time() - started
    summary_path: str | None = None
    verdict = "ERROR"
    error: str | None = None

    # The CLI prints a single JSON line at the end of stdout when --json is on.
    last_json_line = ""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last_json_line = line
            break
    if last_json_line:
        try:
            payload = json.loads(last_json_line)
            verdict = str(payload.get("verdict", "ERROR"))
            summary_path = payload.get("summary_path")
            if not ok:
                error = payload.get("error")
        except json.JSONDecodeError as exc:
            error = f"unparsable cli json: {exc}"
    elif not ok:
        error = f"cli exited non-zero. stderr_tail={stderr[-500:]}"

    return SubprocessOutcome(
        task_id=task_id,
        success=ok and verdict == "PASS",
        verdict=verdict,
        runtime_seconds=runtime,
        stdout_tail=stdout[-2000:],
        stderr_tail=stderr[-2000:],
        summary_path=summary_path,
        error=error,
    )
