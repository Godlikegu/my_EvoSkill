"""Judge subprocess bridge for task runtime environments."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from .models import JudgeResult, RunRecord


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepend_pythonpath(existing: str, project_root: Path) -> str:
    src_path = str((Path(project_root).resolve() / "src").resolve())
    entries = [src_path]
    if existing:
        entries.extend(item for item in existing.split(os.pathsep) if item)
    merged: list[str] = []
    seen: set[str] = set()
    for item in entries:
        normalized = os.path.normcase(os.path.normpath(item))
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(item)
    return os.pathsep.join(merged)


def _judge_runner_env(project_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = _prepend_pythonpath(str(env.get("PYTHONPATH", "") or ""), project_root)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def invoke_judge_runner(
    python_executable: Path | str,
    *,
    mode: str,
    payload: Mapping[str, Any],
    project_root: Optional[Path] = None,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Run the task judge bridge inside one task runtime Python."""

    root = Path(project_root).resolve() if project_root else _default_project_root()
    command = [str(Path(python_executable).resolve()), "-m", "myevoskill.judge_runner", "--mode", mode]
    completed = subprocess.run(
        command,
        input=json.dumps(dict(payload), ensure_ascii=False),
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=_judge_runner_env(root),
        timeout=max(1, int(timeout_seconds)),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "judge subprocess failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr.strip() or '<empty>'}"
        )
    try:
        result = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"judge subprocess returned invalid JSON: {completed.stdout[:400]!r}"
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError("judge subprocess returned a non-object payload")
    if not bool(result.get("ok", False)):
        raise RuntimeError(str(result.get("error_message", "judge subprocess reported failure")))
    return result


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import judge module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _task_import_paths(task_root: Path):
    candidates = [Path(task_root).resolve(), (Path(task_root).resolve() / "src")]
    inserted: list[str] = []
    try:
        for candidate in reversed(candidates):
            candidate_str = str(candidate)
            if not candidate.exists():
                continue
            if candidate_str in sys.path:
                continue
            sys.path.insert(0, candidate_str)
            inserted.append(candidate_str)
        yield
    finally:
        for candidate_str in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(candidate_str)


def _coerce_run_record(payload: Mapping[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=str(payload.get("run_id", "") or ""),
        task_id=str(payload.get("task_id", "") or ""),
        provider=str(payload.get("provider", "") or ""),
        env_hash=str(payload.get("env_hash", "") or ""),
        skills_active=list(payload.get("skills_active", []) or []),
        workspace_root=Path(str(payload.get("workspace_root", "") or "")),
        provider_session_id=str(payload.get("provider_session_id", "") or ""),
        model_provider=str(payload.get("model_provider", "") or ""),
        model_name=str(payload.get("model_name", "") or ""),
        artifacts_uri=str(payload.get("artifacts_uri", "") or ""),
        transcript_uri=str(payload.get("transcript_uri", "") or ""),
        stdout=str(payload.get("stdout", "") or ""),
        stderr=str(payload.get("stderr", "") or ""),
        runtime_seconds=float(payload.get("runtime_seconds", 0.0) or 0.0),
        metadata=dict(payload.get("metadata") or {}),
    )


def _serialize_judge_result(result: JudgeResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "all_metrics_passed": bool(result.all_metrics_passed),
        "metrics_actual": dict(result.metrics_actual),
        "failed_metrics": list(result.failed_metrics),
        "failure_tags": list(result.failure_tags),
    }


def _handle_validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    judge_path = Path(str(payload.get("judge_path", "") or "")).resolve()
    task_root = judge_path.parent.parent
    with _task_import_paths(task_root):
        module = _load_module(judge_path, f"myevoskill_validate_{judge_path.stem}")
    evaluate = getattr(module, "evaluate_run", None)
    return {
        "ok": True,
        "callable_present": callable(evaluate),
        "generated_metric_names": list(getattr(module, "GENERATED_METRIC_NAMES", [])),
        "generated_contract_path": str(
            getattr(module, "GENERATED_REGISTRATION_CONTRACT_PATH", "") or ""
        ),
        "generated_ready": bool(getattr(module, "GENERATED_JUDGE_READY", False)),
    }


def _handle_evaluate(payload: Mapping[str, Any]) -> dict[str, Any]:
    judge_path = Path(str(payload.get("judge_path", "") or "")).resolve()
    task_root = Path(str(payload.get("task_root", "") or "")).resolve()
    manifest = dict(payload.get("manifest") or {})
    run_record = _coerce_run_record(dict(payload.get("run_record") or {}))
    with _task_import_paths(task_root):
        module = _load_module(judge_path, f"myevoskill_evaluate_{judge_path.stem}")
    evaluate = getattr(module, "evaluate_run", None)
    if not callable(evaluate):
        raise AttributeError(f"judge callable 'evaluate_run' not found in {judge_path}")
    result = evaluate(task_root, run_record, manifest)
    if not isinstance(result, JudgeResult):
        raise TypeError(f"judge callable returned unexpected type: {type(result)!r}")
    return {"ok": True, "judge_result": _serialize_judge_result(result)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Task judge subprocess bridge.")
    parser.add_argument("--mode", choices=["validate", "evaluate"], required=True)
    args = parser.parse_args(argv)

    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise RuntimeError("stdin payload must be a JSON object")
        if args.mode == "validate":
            result = _handle_validate(payload)
        else:
            result = _handle_evaluate(payload)
    except Exception as exc:  # pragma: no cover - subprocess error path
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


__all__ = ["invoke_judge_runner", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
