"""Render notebook-oriented PNG visualizations for one model's task outputs.

This is an operator-only offline tool.  It reads hidden ground truth/reference
files and must not be called from agent sandboxes.  The driver intentionally
does not fall back to a generic contract renderer: if a task has no registered
renderer, it records a failure instead of producing a plausible-looking but
misleading figure.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from myevoskill.artifact_paths import model_slug

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from task_visualizers import RenderContext, get_renderer, registered_task_ids


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

    if prefer_verdict.upper() == "LATEST":
        chosen_summary, chosen_data = rows[0]
        reason = "latest_summary"
    else:
        preferred = prefer_verdict.upper()
        order = (preferred,) + tuple(v for v in DEFAULT_VERDICT_ORDER if v != preferred)
        chosen_summary = rows[0][0]
        chosen_data = rows[0][1]
        reason = "latest_summary"
        for verdict in order:
            candidates = [(p, data) for p, data in rows if str(data.get("verdict") or "").upper() == verdict]
            if candidates:
                chosen_summary, chosen_data = candidates[0]
                reason = f"latest_{verdict.lower()}"
                break

    workspace_root = Path(str(chosen_data.get("workspace_root") or ""))
    if not workspace_root.exists():
        return None
    recon_path = workspace_root / _primary_output_path(manifest)
    return SelectedRun(
        task_id=task_id,
        run_id=str(chosen_data.get("run_id") or chosen_summary.parent.name),
        verdict=str(chosen_data.get("verdict") or "UNKNOWN"),
        summary_path=chosen_summary,
        workspace_root=workspace_root,
        recon_path=recon_path,
        selected_reason=reason,
    )


def _mark_legacy_output_deprecated(repo_root: Path, model_slug_value: str) -> None:
    legacy = repo_root / "artifacts" / "visualizations" / model_slug_value
    if not legacy.exists():
        return
    marker = legacy / "DEPRECATED.txt"
    marker.write_text(
        "Deprecated visualization output. These images may have been produced by the old generic_contract renderer "
        "and should not be used for notebook-faithful review. Use artifacts/visualizations_notebook instead.\n",
        encoding="utf-8",
    )


def _clear_pngs(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() == ".png":
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _run_one(
    *,
    repo_root: str,
    tasks_root: str,
    model_slug_value: str,
    task_id: str,
    prefer_verdict: str,
    output_root: str,
    strict_renderer: bool,
) -> dict[str, Any]:
    repo = Path(repo_root)
    tasks = Path(tasks_root)
    output_task_root = Path(output_root) / task_id
    output_task_root.mkdir(parents=True, exist_ok=True)
    _clear_pngs(output_task_root)
    status_path = output_task_root / "status.json"

    base_status: dict[str, Any] = {
        "task_id": task_id,
        "model_slug": model_slug_value,
        "status_path": str(status_path),
        "figures": [],
        "metrics": {},
    }

    renderer = get_renderer(task_id)
    if renderer is None:
        status = {
            **base_status,
            "status": "failed" if strict_renderer else "skipped",
            "error": "missing_registered_renderer",
            "available_renderer_count": len(registered_task_ids()),
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return status

    manifest = _manifest(repo, task_id)
    selected = _select_run(
        repo_root=repo,
        model_slug_value=model_slug_value,
        task_id=task_id,
        manifest=manifest,
        prefer_verdict=prefer_verdict,
    )
    if selected is None:
        status = {**base_status, "status": "skipped", "error": "no_complete_run"}
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
        status = {**base_status, "status": "skipped", "error": "recon_missing"}
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        return status

    task_root = tasks / task_id
    try:
        payload = renderer(
            RenderContext(
                task_id=task_id,
                task_root=task_root,
                recon_path=selected.recon_path,
                output_dir=output_task_root,
                repo_root=repo,
                run_id=selected.run_id,
                verdict=selected.verdict,
            )
        )
        figures = [str(Path(fig)) for fig in payload.get("figures") or []]
        status = {
            **base_status,
            "status": "ok",
            "renderer": payload.get("renderer") or task_id,
            "figures": figures,
            "metrics": payload.get("metrics") or {},
            "error": None,
        }
    except Exception as exc:
        status = {
            **base_status,
            "status": "failed",
            "renderer": task_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc()[-4000:],
        }

    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tasks-root", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--prefer-verdict", default="PASS")
    parser.add_argument("--images-only", action="store_true", help="Accepted for clarity; PNG output is the default.")
    parser.add_argument("--strict-renderer", dest="strict_renderer", action="store_true", default=True)
    parser.add_argument("--no-strict-renderer", dest="strict_renderer", action="store_false")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    repo = _repo_root(args.repo_root)
    tasks = _tasks_root(repo, args.tasks_root)
    slug = model_slug(args.model_id)
    log_model_root = repo / "artifacts" / "logs" / slug
    task_ids = _discover_task_ids(log_model_root, args.task_ids)
    output_root = Path(args.output_root) if args.output_root else repo / "artifacts" / "visualizations_notebook" / slug
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _mark_legacy_output_deprecated(repo, slug)

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
                output_root=str(output_root),
                strict_renderer=bool(args.strict_renderer),
            )
            for task_id in task_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda item: str(item.get("task_id") or ""))
    summary = {
        "model_id": args.model_id,
        "model_slug": slug,
        "output_root": str(output_root),
        "task_count": len(task_ids),
        "registered_renderer_count": len(registered_task_ids()),
        "ok_count": sum(1 for r in records if r.get("status") == "ok"),
        "skipped_count": sum(1 for r in records if r.get("status") == "skipped"),
        "failed_count": sum(1 for r in records if r.get("status") == "failed"),
        "runtime_seconds": time.time() - started,
        "records": records,
    }
    (output_root / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
