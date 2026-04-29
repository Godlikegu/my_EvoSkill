"""Batch driver for operator-only per-task reconstruction visualizations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from myevoskill.artifact_paths import model_slug


DEFAULT_VERDICT_ORDER = (
    "PASS",
    "FAIL",
    "ERROR",
    "TIMEOUT",
    "CRASHED",
    "KILLED",
    "ABORTED",
    "INVALID",
    "MISSING_SUMMARY",
)


@dataclass(frozen=True)
class SelectedRun:
    task_id: str
    run_id: str
    verdict: str
    summary_path: Path
    workspace_root: Path
    recon_path: Path
    selected_reason: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _tasks_root(repo_root: Path, value: str | None) -> Path:
    if value:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    return repo_root.parent / "tasks"


def _manifest(repo_root: Path, task_id: str) -> dict[str, Any]:
    path = repo_root / "registry" / "tasks" / f"{task_id}.json"
    if not path.exists():
        return {}
    return _load_json(path)


def _task_python(repo_root: Path, task_id: str, manifest: dict[str, Any]) -> Path:
    runtime_env = manifest.get("runtime_env") if isinstance(manifest, dict) else {}
    manifest_python = Path(str((runtime_env or {}).get("python_executable") or ""))
    if manifest_python.exists():
        return manifest_python
    fallback = repo_root / ".venvs" / task_id / "Scripts" / "python.exe"
    if fallback.exists():
        return fallback
    return Path(sys.executable)


def _primary_output_path(manifest: dict[str, Any]) -> Path:
    value = manifest.get("primary_output_path") if isinstance(manifest, dict) else None
    return Path(str(value or "output/reconstruction.npz"))


def _discover_task_ids(log_model_root: Path, explicit: list[str] | None) -> list[str]:
    if explicit:
        return sorted(dict.fromkeys(explicit))
    if not log_model_root.exists():
        return []
    return sorted(path.name for path in log_model_root.iterdir() if path.is_dir())


def _summary_mtime(summary: Path) -> float:
    try:
        return summary.stat().st_mtime
    except OSError:
        return 0.0


def _select_run(
    *,
    repo_root: Path,
    model_slug_value: str,
    task_id: str,
    manifest: dict[str, Any],
    prefer_verdict: str,
) -> SelectedRun | None:
    log_task_root = repo_root / "artifacts" / "logs" / model_slug_value / task_id
    summaries = sorted(log_task_root.glob("run-*/run_summary.json"), key=_summary_mtime, reverse=True)
    if not summaries:
        return None

    rows: list[tuple[Path, dict[str, Any]]] = []
    for summary in summaries:
        try:
            rows.append((summary, _load_json(summary)))
        except (OSError, json.JSONDecodeError):
            continue
    if not rows:
        return None

    order: tuple[str, ...]
    if prefer_verdict.upper() == "LATEST":
        order = ()
    else:
        preferred = prefer_verdict.upper()
        tail = tuple(v for v in DEFAULT_VERDICT_ORDER if v != preferred)
        order = (preferred,) + tail

    chosen: tuple[Path, dict[str, Any], str] | None = None
    if order:
        for verdict in order:
            candidates = [(p, data) for p, data in rows if str(data.get("verdict") or "").upper() == verdict]
            if candidates:
                summary, data = candidates[0]
                chosen = (summary, data, f"latest_{verdict.lower()}")
                break
    if chosen is None:
        summary, data = rows[0]
        chosen = (summary, data, "latest_summary")

    summary, data, reason = chosen
    workspace_root = Path(str(data.get("workspace_root") or ""))
    if not workspace_root.exists():
        return None
    recon_path = workspace_root / _primary_output_path(manifest)
    return SelectedRun(
        task_id=task_id,
        run_id=str(data.get("run_id") or summary.parent.name),
        verdict=str(data.get("verdict") or "UNKNOWN"),
        summary_path=summary,
        workspace_root=workspace_root,
        recon_path=recon_path,
        selected_reason=reason,
    )


def _parse_last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {"error": "missing_json_stdout", "stdout_tail": stdout[-2000:]}


def _run_one(
    *,
    repo_root: str,
    tasks_root: str,
    model_slug_value: str,
    task_id: str,
    prefer_verdict: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    repo = Path(repo_root)
    tasks = Path(tasks_root)
    manifest = _manifest(repo, task_id)
    selected = _select_run(
        repo_root=repo,
        model_slug_value=model_slug_value,
        task_id=task_id,
        manifest=manifest,
        prefer_verdict=prefer_verdict,
    )
    output_root = repo / "artifacts" / "visualizations" / model_slug_value / task_id
    output_dir = output_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"

    base_status: dict[str, Any] = {
        "task_id": task_id,
        "model_slug": model_slug_value,
        "status_path": str(status_path),
    }
    if selected is None:
        status = {
            **base_status,
            "status": "skipped",
            "error": "no_complete_run",
            "figures": [],
            "metrics": {},
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return status

    base_status.update(
        {
            "verdict": selected.verdict,
            "selected_run_id": selected.run_id,
            "selected_reason": selected.selected_reason,
            "summary_path": str(selected.summary_path),
            "workspace_root": str(selected.workspace_root),
            "recon_path": str(selected.recon_path),
        }
    )
    if not selected.recon_path.exists():
        status = {
            **base_status,
            "status": "skipped",
            "error": "recon_missing",
            "figures": [],
            "metrics": {},
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return status

    task_root = tasks / task_id
    visualize_py = task_root / "visualize.py"
    if not visualize_py.exists():
        status = {
            **base_status,
            "status": "skipped",
            "error": "visualize_py_missing",
            "figures": [],
            "metrics": {},
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return status

    python_exe = _task_python(repo, task_id, manifest)
    cmd = [
        str(python_exe),
        str(visualize_py),
        "--recon",
        str(selected.recon_path),
        "--output-dir",
        str(output_dir),
        "--task-root",
        str(task_root),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(task_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        payload = _parse_last_json_line(proc.stdout)
        status = {
            **base_status,
            "status": "ok" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "python_executable": str(python_exe),
            "figures": payload.get("figures") or [],
            "metrics": payload.get("metrics") or {},
            "error": payload.get("error"),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        status = {
            **base_status,
            "status": "failed",
            "exit_code": None,
            "python_executable": str(python_exe),
            "figures": [],
            "metrics": {},
            "error": "visualize_timeout",
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }

    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def _write_index(output_root: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "<!doctype html>",
        "<meta charset=\"utf-8\">",
        "<title>Model Reconstruction Visualizations</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px} .task{border-top:1px solid #ddd;padding:18px 0} img{max-width:100%;height:auto;border:1px solid #ddd} code{background:#f5f5f5;padding:2px 4px}</style>",
        "<h1>Model Reconstruction Visualizations</h1>",
    ]
    for record in records:
        task_id = record.get("task_id")
        lines.append(f"<section class=\"task\"><h2>{task_id}</h2>")
        lines.append(
            "<p>"
            f"status: <code>{record.get('status')}</code> "
            f"verdict: <code>{record.get('verdict', '')}</code> "
            f"run: <code>{record.get('selected_run_id', '')}</code>"
            "</p>"
        )
        if record.get("error"):
            lines.append(f"<p>error: <code>{record.get('error')}</code></p>")
        for fig in record.get("figures") or []:
            fig_path = Path(fig)
            try:
                rel = fig_path.resolve().relative_to(output_root.resolve()).as_posix()
            except ValueError:
                rel = fig_path.as_posix()
            lines.append(f"<figure><img src=\"{rel}\" alt=\"{task_id}\"><figcaption>{fig_path.name}</figcaption></figure>")
        lines.append("</section>")
    (output_root / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tasks-root", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--prefer-verdict", default="PASS")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)

    repo = _repo_root(args.repo_root)
    tasks = _tasks_root(repo, args.tasks_root)
    slug = model_slug(args.model_id)
    log_model_root = repo / "artifacts" / "logs" / slug
    task_ids = _discover_task_ids(log_model_root, args.task_ids)
    output_root = repo / "artifacts" / "visualizations" / slug
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [
            executor.submit(
                _run_one,
                repo_root=str(repo),
                tasks_root=str(tasks),
                model_slug_value=slug,
                task_id=task_id,
                prefer_verdict=args.prefer_verdict,
                timeout_seconds=args.timeout_seconds,
            )
            for task_id in task_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda item: str(item.get("task_id") or ""))
    summary = {
        "model_id": args.model_id,
        "model_slug": slug,
        "task_count": len(task_ids),
        "ok_count": sum(1 for r in records if r.get("status") == "ok"),
        "skipped_count": sum(1 for r in records if r.get("status") == "skipped"),
        "failed_count": sum(1 for r in records if r.get("status") == "failed"),
        "runtime_seconds": time.time() - started,
        "records": records,
    }
    (output_root / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_index(output_root, records)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
