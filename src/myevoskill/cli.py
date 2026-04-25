"""Single CLI entry point for MyEvoSkill.

Subcommands
-----------

* ``register-task``    Build / refresh a registry manifest for one task.
* ``run-task``         Run one registered task end-to-end (one process,
                       one Claude session, multi-round with judge feedback).
* ``run-batch``        Run several tasks concurrently in isolated subprocesses,
                       deleting per-run claude history afterwards.

Everything else (compilation, visualisation, legacy bootstrap, ...) lives in
its own module under ``myevoskill/`` and is invoked directly via
``python -m myevoskill.<module>`` for advanced users.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .concurrency import run_tasks_parallel
from .harness import HarnessConfig, run_task_once

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- helpers


def _load_manifest(repo_root: Path, task_id: str) -> dict[str, Any]:
    path = repo_root / "registry" / "tasks" / f"{task_id}.json"
    if not path.exists():
        raise SystemExit(
            f"manifest not found: {path}\n"
            f"Run `python -m myevoskill.cli register-task --task-id {task_id}` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_default_model(cli_model: str | None) -> str | None:
    """Resolve the model name with precedence:

    1. ``--model`` flag (explicit override).
    2. ``MYEVOSKILL_MODEL`` environment variable.
    3. ``model`` field in ``~/.claude/settings.json`` (so the harness uses
       the same model the user already configured for the Claude CLI,
       e.g. ``Vendor2/Claude-4.6-opus`` on a 3rd-party gateway).
    4. None - let the SDK pick its own default.
    """

    import os as _os

    if cli_model:
        return cli_model
    env_model = _os.environ.get("MYEVOSKILL_MODEL")
    if env_model:
        return env_model
    home = Path(_os.environ.get("USERPROFILE") or _os.environ.get("HOME") or "")
    settings_path = home / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            m = data.get("model")
            if isinstance(m, str) and m.strip():
                return m.strip()
        except (OSError, json.JSONDecodeError):
            pass
    return None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )


# --------------------------------------------------------------------- commands


def cmd_register_task(args: argparse.Namespace) -> int:
    """Delegate to the existing ``task_register`` CLI.

    The harness only consumes the manifest schema produced there, so we keep
    one source of truth instead of forking the registration logic.
    """

    from . import task_register

    repo_root = Path(args.repo_root).resolve()
    forwarded = ["--task-id", args.task_id, "--repo-root", str(repo_root)]
    if args.force:
        forwarded.append("--force")
    return int(task_register.main(forwarded) or 0)


def cmd_run_task(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest = _load_manifest(repo_root, args.task_id)

    resolved_model = _resolve_default_model(args.model)
    if resolved_model:
        logger.info("using model: %s", resolved_model)
    config = HarnessConfig(
        repo_root=repo_root,
        manifest=manifest,
        max_rounds=args.max_rounds,
        budget_seconds=args.budget_seconds,
        max_turns_per_round=args.max_turns_per_round,
        model=resolved_model,
        judge_python=args.judge_python,
        show_metric_status=bool(args.show_metric_status),
        keep_workspace_on_success=bool(args.keep_workspace),
    )

    outcome = run_task_once(config)

    payload = {
        "task_id": outcome.task_id,
        "run_id": outcome.run_id,
        "verdict": outcome.verdict,
        "rounds_used": outcome.rounds_used,
        "runtime_seconds": outcome.runtime_seconds,
        "summary_path": str(outcome.summary_path),
        "trajectory_path": str(outcome.trajectory_path),
        "log_root": str(outcome.log_root),
        "workspace_root": str(outcome.workspace_root),
        "error": outcome.error,
    }

    if args.json:
        # print exactly one JSON line at the very end so the parent process
        # (concurrency pool) can find it deterministically.
        print(json.dumps(payload))
    else:
        print(f"verdict: {payload['verdict']}")
        print(f"rounds:  {payload['rounds_used']}")
        print(f"summary: {payload['summary_path']}")
        print(f"traj:    {payload['trajectory_path']}")

    return 0 if outcome.verdict == "PASS" else 1


def cmd_run_batch(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    task_ids = args.task_ids
    extra: dict[str, Any] = {
        "max-rounds": args.max_rounds,
        "budget-seconds": args.budget_seconds,
        "max-turns-per-round": args.max_turns_per_round,
    }
    resolved_model = _resolve_default_model(args.model)
    if resolved_model:
        logger.info("using model: %s", resolved_model)
        extra["model"] = resolved_model
    if args.judge_python:
        extra["judge-python"] = args.judge_python

    outcomes = run_tasks_parallel(
        repo_root=repo_root,
        task_ids=task_ids,
        max_workers=args.max_workers,
        extra_run_args=extra,
        timeout_seconds=args.budget_seconds + 600,
    )

    # Persist a batch-level summary.
    summary_path = repo_root / "artifacts" / "logs" / f"batch_{int(__import__('time').time())}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_ids": list(task_ids),
        "outcomes": [asdict(o) for o in outcomes],
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = [o for o in outcomes if not o.success]
    print(f"\nbatch complete: {len(outcomes)} task(s), {len(failures)} failed.")
    print(f"summary: {summary_path}")
    for o in outcomes:
        print(f"  {o.task_id:30s} {o.verdict:8s} ({o.runtime_seconds:6.1f}s)")
    return 0 if not failures else 1


# --------------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myevoskill",
        description="MyEvoSkill harness CLI.",
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # register-task
    p_reg = sub.add_parser("register-task", help="register / refresh a task manifest")
    p_reg.add_argument("--task-id", required=True)
    p_reg.add_argument("--repo-root", default=".")
    p_reg.add_argument("--force", action="store_true")
    p_reg.set_defaults(func=cmd_register_task)

    # run-task
    p_run = sub.add_parser("run-task", help="run a single registered task")
    p_run.add_argument("--task-id", required=True)
    p_run.add_argument("--repo-root", default=".")
    p_run.add_argument("--max-rounds", type=int, default=4)
    p_run.add_argument("--budget-seconds", type=int, default=7200)
    p_run.add_argument("--max-turns-per-round", type=int, default=60)
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--judge-python", default=None)
    p_run.add_argument("--show-metric-status", action="store_true")
    p_run.add_argument("--keep-workspace", action="store_true")
    p_run.add_argument("--json", action="store_true", help="emit one JSON summary line at end")
    p_run.set_defaults(func=cmd_run_task)

    # run-batch
    p_batch = sub.add_parser("run-batch", help="run several tasks in parallel subprocesses")
    p_batch.add_argument("--repo-root", default=".")
    p_batch.add_argument("--task-ids", nargs="+", required=True)
    p_batch.add_argument("--max-workers", type=int, default=2)
    p_batch.add_argument("--max-rounds", type=int, default=4)
    p_batch.add_argument("--budget-seconds", type=int, default=7200)
    p_batch.add_argument("--max-turns-per-round", type=int, default=60)
    p_batch.add_argument("--model", default=None)
    p_batch.add_argument("--judge-python", default=None)
    p_batch.set_defaults(func=cmd_run_batch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
