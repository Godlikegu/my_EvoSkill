"""Black-box diagnostic summary for valid-split skill runs.

This script intentionally reads only run logs and judge JSON produced by
validation. It does not read task source, valid data, workspaces, or feed
anything back into distillation. Use the output as an operator-facing report
when an E2E transfer gate fails.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, Mapping) else None


def _summarise_run(run_dir: Path) -> dict[str, Any] | None:
    summary = _load_json(run_dir / "run_summary.json") or _load_json(run_dir / "summary.json")
    if summary is None:
        return None
    judges: list[dict[str, Any]] = []
    for judge_path in sorted(run_dir.glob("judge_round_*.json")):
        judge = _load_json(judge_path)
        if judge is None:
            continue
        jr = judge.get("judge_result") or {}
        fb = judge.get("feedback") or {}
        if not isinstance(jr, Mapping):
            jr = {}
        if not isinstance(fb, Mapping):
            fb = {}
        judges.append({
            "round": judge_path.stem.replace("judge_round_", ""),
            "verdict": fb.get("verdict"),
            "failed_metrics": list(jr.get("failed_metrics") or []),
            "metrics_actual": dict(jr.get("metrics_actual") or {}),
            "failure_tags": list(jr.get("failure_tags") or []),
        })
    last = judges[-1] if judges else {}
    return {
        "task_id": summary.get("task_id") or run_dir.parent.name,
        "run_id": summary.get("run_id") or run_dir.name,
        "verdict": summary.get("verdict"),
        "rounds_used": summary.get("rounds_used"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "last_failed_metrics": list(last.get("failed_metrics") or []),
        "last_metrics_actual": dict(last.get("metrics_actual") or {}),
        "last_failure_tags": list(last.get("failure_tags") or []),
    }


def diagnose(log_root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for task_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        for run_dir in sorted(task_dir.glob("run-*")):
            item = _summarise_run(run_dir)
            if item is not None:
                runs.append(item)

    verdicts = Counter(str(r.get("verdict") or "UNKNOWN") for r in runs)
    metric_clusters: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        failed = run.get("last_failed_metrics") or []
        key = ",".join(str(x) for x in failed) if failed else "<none>"
        metric_clusters[key].append(str(run.get("task_id")))
    return {
        "log_root": str(log_root),
        "n_runs": len(runs),
        "verdict_counts": dict(verdicts),
        "metric_clusters": dict(metric_clusters),
        "runs": runs,
    }


def _to_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Valid Failure Diagnosis",
        "",
        f"- log_root: `{report.get('log_root')}`",
        f"- runs: {report.get('n_runs')}",
        f"- verdict_counts: `{json.dumps(report.get('verdict_counts') or {}, ensure_ascii=False)}`",
        "",
        "## Metric Clusters",
    ]
    for metrics, tasks in sorted((report.get("metric_clusters") or {}).items()):
        lines.append(f"- `{metrics}`: {', '.join(tasks)}")
    lines.extend(["", "## Runs", "| task | verdict | rounds | failed metrics | metrics |", "| --- | --- | --- | --- | --- |"])
    for run in report.get("runs") or []:
        if not isinstance(run, Mapping):
            continue
        lines.append(
            "| {task} | {verdict} | {rounds} | {failed} | `{metrics}` |".format(
                task=run.get("task_id"),
                verdict=run.get("verdict"),
                rounds=run.get("rounds_used"),
                failed=",".join(str(x) for x in (run.get("last_failed_metrics") or [])),
                metrics=json.dumps(run.get("last_metrics_actual") or {}, ensure_ascii=False),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise valid run failures from logs only.")
    parser.add_argument("--log-root", required=True, help="artifacts/logs/<validation_slug>")
    parser.add_argument("--out", default=None, help="optional output path; .md writes Markdown, otherwise JSON")
    parser.add_argument("--markdown", action="store_true", help="print Markdown instead of JSON")
    args = parser.parse_args()

    log_root = Path(args.log_root).resolve()
    if not log_root.exists():
        parser.error(f"--log-root does not exist: {log_root}")
    report = diagnose(log_root)
    as_markdown = args.markdown or (args.out and str(args.out).lower().endswith(".md"))
    text = _to_markdown(report) if as_markdown else json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        target = Path(args.out).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
