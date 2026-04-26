"""Hidden judge bridge.

The judge bridge is invoked *out-of-process* by the harness, after each round.

Inputs (from the harness):
    * task manifest        - tells us where the task source lives, what
                             the primary output path is, etc.
    * workspace root       - the agent_root of the just-finished round.

It runs the task-local ``evaluation/judge_adapter.py:evaluate_run``, captures
its ``JudgeResult``, persists a complete record to disk for debugging, and
returns a sanitised :class:`JudgeFeedback` to the harness. The agent will only
ever see the sanitised version.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


# Verdicts exposed to the agent.
PASS = "PASS"
FAIL = "FAIL"
INVALID = "INVALID"


@dataclass(frozen=True)
class JudgeFeedback:
    """Minimal signal returned from the judge to the agent.

    ``verdict`` is the only field the agent ever sees. ``metric_status``
    is recorded internally so the harness can optionally display per-metric
    pass/fail (without numeric values) when the operator wants verbose mode.
    """

    verdict: str  # PASS / FAIL / INVALID
    failure_tags: tuple[str, ...] = field(default_factory=tuple)
    metric_status: dict[str, bool] = field(default_factory=dict)

    @property
    def is_infrastructure_error(self) -> bool:
        infra_tags = {
            "judge_runtime_error",
            "judge_timeout",
            "judge_unparsable",
            "missing_judge_adapter",
        }
        return any(t.split(":", 1)[0] in infra_tags for t in self.failure_tags)


@dataclass
class JudgeRunResult:
    """Full judge artefact persisted to disk (NOT shown to the agent)."""

    feedback: JudgeFeedback
    judge_result_raw: dict[str, Any]
    runtime_seconds: float
    stdout: str
    stderr: str
    success: bool


class JudgeRunner:
    """Run a task-local judge_adapter.evaluate_run in a subprocess."""

    def __init__(
        self,
        *,
        repo_root: Path,
        manifest: Mapping[str, Any],
        log_root: Path,
        python_executable: str | None = None,
        timeout_seconds: int = 600,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.manifest = dict(manifest)
        self.log_root = Path(log_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        # Resolution order for the judge subprocess interpreter:
        #   1. explicit ``python_executable`` argument from the harness;
        #   2. ``runtime_env.python_executable`` recorded in the manifest
        #      (i.e. the per-task venv built by scripts/setup_task_env.sh);
        #   3. ``sys.executable`` of the harness process (last-resort
        #      fallback for tasks that don't need extra deps).
        manifest_py = ""
        runtime_env = self.manifest.get("runtime_env") or {}
        if isinstance(runtime_env, Mapping):
            manifest_py = str(runtime_env.get("python_executable") or "").strip()
        self.python_executable = (
            python_executable
            or (manifest_py if manifest_py and Path(manifest_py).exists() else "")
            or sys.executable
        )
        self.timeout_seconds = int(timeout_seconds)

    # ------------------------------------------------------------------ run

    def run(
        self,
        *,
        round_index: int,
        run_id: str,
        workspace_root: Path,
    ) -> JudgeRunResult:
        task_id = str(self.manifest["task_id"])
        task_root = self._resolve_task_root()
        adapter_path = task_root / "evaluation" / "judge_adapter.py"
        if not adapter_path.exists():
            return self._invalid(
                round_index,
                "missing_judge_adapter",
                f"judge_adapter.py not found at {adapter_path}",
            )

        # We invoke a small driver via -c to avoid persisting any helper
        # script on disk. The driver receives the manifest + workspace via
        # JSON on stdin and prints a JSON result on stdout.
        driver = textwrap.dedent(
            """
            import importlib.util, json, sys, traceback
            from pathlib import Path

            payload = json.loads(sys.stdin.read())
            adapter_path = Path(payload["adapter_path"])
            task_root = Path(payload["task_root"])
            workspace_root = Path(payload["workspace_root"])
            manifest = payload["manifest"]
            run_id = payload["run_id"]
            task_id = payload["task_id"]

            try:
                spec = importlib.util.spec_from_file_location(
                    "task_local_judge_adapter", adapter_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(module)
                from myevoskill.models import RunRecord
                record = RunRecord(
                    run_id=run_id,
                    task_id=task_id,
                    provider="claude_code_harness",
                    env_hash="",
                    skills_active=(),
                    workspace_root=workspace_root,
                )
                result = module.evaluate_run(task_root, record, manifest)
                from dataclasses import asdict
                out = {"ok": True, "judge_result": asdict(result)}
            except SystemExit as exc:
                out = {"ok": False, "error": f"SystemExit: {exc.code}",
                       "traceback": traceback.format_exc()}
            except Exception as exc:  # noqa: BLE001
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            sys.stdout.write(json.dumps(out))
            """
        ).strip()

        env = os.environ.copy()
        # Make sure myevoskill is importable in the judge subprocess and that
        # the task root is on sys.path (some adapters import sibling modules).
        repo_src = str((self.repo_root / "src").resolve())
        env["PYTHONPATH"] = os.pathsep.join(
            [repo_src, str(task_root.resolve()), env.get("PYTHONPATH", "")]
        )
        env["PYTHONIOENCODING"] = "utf-8"

        payload = {
            "adapter_path": str(adapter_path),
            "task_root": str(task_root),
            "workspace_root": str(workspace_root),
            "manifest": self.manifest,
            "run_id": run_id,
            "task_id": task_id,
        }

        start = time.time()
        try:
            completed = subprocess.run(
                [self.python_executable, "-c", driver],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
                cwd=str(self.repo_root),
            )
        except subprocess.TimeoutExpired as exc:
            return self._invalid(
                round_index, "judge_timeout", f"judge timed out: {exc}"
            )
        runtime = time.time() - start

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            return self._invalid(
                round_index,
                "judge_unparsable",
                f"judge stdout not JSON. stderr={stderr[-500:]}",
                stdout=stdout,
                stderr=stderr,
            )

        if not parsed.get("ok"):
            return self._invalid(
                round_index,
                "judge_runtime_error",
                str(parsed.get("error", "")),
                stdout=stdout,
                stderr=stderr,
            )

        raw = parsed["judge_result"]
        feedback = self._make_feedback(raw)
        result = JudgeRunResult(
            feedback=feedback,
            judge_result_raw=raw,
            runtime_seconds=runtime,
            stdout=stdout,
            stderr=stderr,
            success=True,
        )
        self._persist(round_index, result)
        return result

    # ------------------------------------------------------------ internals

    def _resolve_task_root(self) -> Path:
        # Mirror :func:`workspace.builder._resolve_task_root` so the harness
        # and judge always agree on where the task source lives. We try a few
        # candidate base directories so manifests written by older versions of
        # ``register-task`` still resolve.
        raw = str(self.manifest.get("source_task_dir") or "")
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate.resolve()
        bases = [
            self.repo_root,
            self.repo_root / "registry" / "tasks",
            self.repo_root.parent,
        ]
        for base in bases:
            resolved = (base / candidate).resolve()
            if resolved.exists():
                return resolved
        return (self.repo_root / candidate).resolve()

    def _make_feedback(self, raw: Mapping[str, Any]) -> JudgeFeedback:
        all_passed = bool(raw.get("all_metrics_passed"))
        failure_tags = tuple(str(t) for t in (raw.get("failure_tags") or []))
        failed_metrics = set(str(m) for m in (raw.get("failed_metrics") or []))
        metrics_actual = dict(raw.get("metrics_actual") or {})

        invalid_tags = {
            "missing_output",
            "missing_required_field",
            "invalid_output_schema",
        }
        if any(t.split(":", 1)[0] in invalid_tags for t in failure_tags):
            verdict = INVALID
        elif all_passed:
            verdict = PASS
        else:
            verdict = FAIL

        metric_status = {
            name: name not in failed_metrics for name in metrics_actual.keys()
        }
        return JudgeFeedback(
            verdict=verdict,
            failure_tags=failure_tags,
            metric_status=metric_status,
        )

    def _invalid(
        self,
        round_index: int,
        tag: str,
        detail: str,
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> JudgeRunResult:
        feedback = JudgeFeedback(verdict=INVALID, failure_tags=(tag,))
        result = JudgeRunResult(
            feedback=feedback,
            judge_result_raw={
                "task_id": self.manifest.get("task_id"),
                "all_metrics_passed": False,
                "metrics_actual": {},
                "failed_metrics": [],
                "failure_tags": [tag],
                "_detail": detail,
            },
            runtime_seconds=0.0,
            stdout=stdout,
            stderr=stderr,
            success=False,
        )
        self._persist(round_index, result)
        return result

    def _persist(self, round_index: int, result: JudgeRunResult) -> None:
        out = {
            "feedback": asdict(result.feedback),
            "judge_result": result.judge_result_raw,
            "runtime_seconds": result.runtime_seconds,
            "success": result.success,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
        path = self.log_root / f"judge_round_{round_index:02d}.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
