"""Domain-level skill self-improvement loop for one split.

This is the orchestrator the user asked for: it does NOT distill a skill
per task. Instead, it treats every task in the split's ``train`` list as
a *training example of one domain*. Each epoch:

1.  Runs the harness on every train task in epoch 1, so the domain has
    one complete sweep before the first distillation. Later epochs only
    rerun train tasks that still have no PASS. If a previous epoch
    produced a skill pack, that pack is injected into the run via
    ``--skill-pack-dir`` so the agent gets the benefit of what we already
    distilled.
2.  Re-distills a *single* domain skill from **all currently-passing
    train episodes across all train tasks** (``synthesize_skill``
    natively merges them).
3.  Stops when every train task has at least one PASS run, or when
    ``--max-epochs`` is hit.

After the train loop, it runs ``validate-skill`` on one valid task only,
with the final skill injected. Validation is skipped unless every train
task has a PASS.

This script intentionally shells out to ``python -m myevoskill.cli``
for ``run-task``, ``distill-skill``, ``validate-skill`` so each agent
session runs in its own subprocess (matches how ``run-batch`` operates)
and so the loop can be killed/resumed without poisoning Python state.

Usage
-----

    python scripts/train_domain_skill.py \\
        --repo-root . \\
        --split registry/splits/wave_optics_v1.json \\
        --skill-id wave_optics_recon_v1 \\
        --model-id gateway-claude-4.6-opus \\
        --max-epochs 3 \\
        --max-rounds 5 \\
        --budget-seconds 5400

The script is *resumable*: it scans ``artifacts/logs/<model_slug>/<task>/run-*/run_summary.json``
on startup and treats any task that already has a PASS run as "done";
it will NOT re-run a passing task. To force a clean retrain, point
``--logs-fresh-after`` at a UNIX timestamp.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("train_domain_skill")


# --------------------------------------------------------------------- types


@dataclass
class TaskState:
    task_id: str
    verdict: str = "NO_RUN"          # PASS / FAIL / TIMEOUT / ERROR / NO_RUN
    last_run_id: str | None = None
    runs_done_this_session: int = 0


@dataclass
class EpochResult:
    epoch: int
    pass_ids: list[str]
    fail_ids: list[str]
    skill_pack_dir: Path | None
    distill_summary: dict | None = None
    per_task: dict[str, str] = field(default_factory=dict)  # task_id -> verdict


# --------------------------------------------------------------------- helpers


def _read_run_summary(p: Path) -> dict | None:
    sp = p / "run_summary.json"
    if not sp.exists():
        sp = p / "summary.json"
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scan_task_state(
    log_root: Path,
    task_id: str,
    fresh_after: float | None,
) -> TaskState:
    """Best verdict ever recorded for this task (or NO_RUN).

    If ``fresh_after`` is given (UNIX seconds), only runs whose mtime is
    >= that timestamp are considered.
    """

    state = TaskState(task_id=task_id, verdict="NO_RUN")
    tdir = log_root / task_id
    if not tdir.exists():
        return state
    runs = sorted(tdir.glob("run-*"))
    for rdir in runs:
        if fresh_after is not None and rdir.stat().st_mtime < fresh_after:
            continue
        s = _read_run_summary(rdir)
        if not s:
            continue
        v = str(s.get("verdict") or "?")
        if v == "PASS":
            state.verdict = "PASS"
            state.last_run_id = s.get("run_id") or rdir.name
            return state
        if state.verdict == "NO_RUN":
            state.verdict = v
            state.last_run_id = s.get("run_id") or rdir.name
    return state


def _python_exe() -> str:
    # Use the same interpreter that's running this script (we live inside
    # the evoskill conda env on the user's box).
    return sys.executable


def _run_subprocess(cmd: list[str], log_path: Path, env_extra: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    logger.info("$ %s", " ".join(cmd))
    logger.info("  log: %s", log_path)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def _model_slug_for_split(split: dict) -> str:
    return str(split.get("model_slug") or "default")


def _skill_dir_name(skill_id: str) -> str:
    """Mirror the distiller's Anthropic-compatible skill name normalisation."""

    raw = skill_id.lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", raw)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:64].strip("-") or "untitled-skill"


# --------------------------------------------------------------------- ops


def _invoke_run_task(
    *,
    repo_root: Path,
    task_id: str,
    model_id: str | None,
    llm_config: Path | None,
    max_rounds: int,
    budget_seconds: int,
    max_turns_per_round: int,
    skill_pack_dir: Path | None,
    sandbox_root: Path | None,
    artifact_model_slug: str | None,
    epoch_log_dir: Path,
) -> int:
    cmd = [
        _python_exe(),
        "-m", "myevoskill.cli",
        "run-task",
        "--task-id", task_id,
        "--repo-root", str(repo_root),
        "--max-rounds", str(max_rounds),
        "--budget-seconds", str(budget_seconds),
        "--max-turns-per-round", str(max_turns_per_round),
    ]
    if model_id:
        cmd += ["--model-id", model_id]
    if llm_config:
        cmd += ["--llm-config", str(llm_config)]
    if skill_pack_dir:
        cmd += ["--skill-pack-dir", str(skill_pack_dir)]
    if sandbox_root:
        cmd += ["--sandbox-root", str(sandbox_root)]
    if artifact_model_slug:
        cmd += ["--artifact-model-slug", artifact_model_slug]
    return _run_subprocess(cmd, epoch_log_dir / f"run_task_{task_id}.log")


def _invoke_distill_skill(
    *,
    repo_root: Path,
    split_path: Path,
    skill_id: str,
    out_root: Path,
    model_id: str | None,
    llm_config: Path | None,
    epoch_log_dir: Path,
) -> int:
    cmd = [
        _python_exe(),
        "-m", "myevoskill.cli",
        "distill-skill",
        "--repo-root", str(repo_root),
        "--split", str(split_path),
        "--skill-id", skill_id,
        "--out-root", str(out_root),
    ]
    if model_id:
        cmd += ["--model-id", model_id]
    if llm_config:
        cmd += ["--llm-config", str(llm_config)]
    return _run_subprocess(cmd, epoch_log_dir / "distill_skill.log")


def _invoke_validate_skill(
    *,
    repo_root: Path,
    split_path: Path,
    skill_pack_dir: Path,
    model_id: str | None,
    llm_config: Path | None,
    max_rounds: int,
    budget_seconds: int,
    max_turns_per_round: int,
    valid_task_id: str | None,
    artifact_model_slug: str | None,
    epoch_log_dir: Path,
) -> int:
    cmd = [
        _python_exe(),
        "-m", "myevoskill.cli",
        "validate-skill",
        "--repo-root", str(repo_root),
        "--split", str(split_path),
        "--skill-pack-dir", str(skill_pack_dir),
        "--max-rounds", str(max_rounds),
        "--budget-seconds", str(budget_seconds),
        "--max-turns-per-round", str(max_turns_per_round),
    ]
    if valid_task_id:
        cmd += ["--valid-task-ids", valid_task_id]
    if model_id:
        cmd += ["--model-id", model_id]
    if llm_config:
        cmd += ["--llm-config", str(llm_config)]
    if artifact_model_slug:
        cmd += ["--artifact-model-slug", artifact_model_slug]
    return _run_subprocess(cmd, epoch_log_dir / "validate_skill.log")


# --------------------------------------------------------------------- main loop


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="train_domain_skill",
        description=(
            "Domain-level skill self-improvement: run all train tasks, "
            "distill ONE merged skill from all PASS episodes, repeat "
            "until every train task PASSes (or max-epochs hit), then "
            "validate on the valid split."
        ),
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", required=True,
                        help="path to registry/splits/<split>.json")
    parser.add_argument("--skill-id", required=True,
                        help="domain-level skill id, e.g. wave_optics_recon_v1")
    parser.add_argument("--out-root", default=None,
                        help="default: <repo>/artifacts/skills")
    parser.add_argument("--model-id", default=None,
                        help="model id from config/llm.yaml; needed for real LLM runs")
    parser.add_argument("--llm-config", default=None)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="--max-rounds passed to run-task / validate-skill")
    parser.add_argument("--budget-seconds", type=int, default=5400)
    parser.add_argument("--max-turns-per-round", type=int, default=60)
    parser.add_argument("--sandbox-root", default=None)
    parser.add_argument("--artifact-model-slug", default=None,
                        help=("override the artifact slug used by run-task and "
                              "validate-skill; the split model_slug should "
                              "match this when isolating a reproduction run"))
    parser.add_argument("--logs-fresh-after", type=float, default=None,
                        help=("Only treat run-* directories newer than this "
                              "UNIX-epoch timestamp as evidence of PASS. "
                              "Use to force a clean retrain."))
    parser.add_argument("--skip-validate", action="store_true",
                        help="don't run validate-skill at the end")
    parser.add_argument("--valid-task-id", default=None,
                        help=("single valid task to evaluate after all train "
                              "tasks PASS (default: first task in split.valid)"))
    parser.add_argument("--require-all-train-pass", action="store_true",
                        help=("explicitly fail the run unless every train task "
                              "has at least one PASS; valid still waits for "
                              "that gate"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    repo_root = Path(args.repo_root).resolve()
    split_path = Path(args.split).resolve()
    if not split_path.is_absolute():
        split_path = (repo_root / args.split).resolve()
    if not split_path.exists():
        print(f"split not found: {split_path}", file=sys.stderr)
        return 2

    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids: list[str] = list(split.get("train") or [])
    valid_ids: list[str] = list(split.get("valid") or [])
    if not train_ids:
        print("split has no train tasks; nothing to do", file=sys.stderr)
        return 2
    if args.valid_task_id and args.valid_task_id not in valid_ids:
        print(
            f"--valid-task-id must be in split.valid; got {args.valid_task_id!r}",
            file=sys.stderr,
        )
        return 2
    selected_valid_task = args.valid_task_id or (valid_ids[0] if valid_ids else None)

    out_root = Path(args.out_root).resolve() if args.out_root else (
        repo_root / "artifacts" / "skills"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    model_slug = _model_slug_for_split(split)
    log_root = repo_root / "artifacts" / "logs" / model_slug

    session_log_root = (
        repo_root
        / "artifacts"
        / "logs"
        / "_domain_train"
        / f"{args.skill_id}_{int(time.time())}"
    )
    session_log_root.mkdir(parents=True, exist_ok=True)
    logger.info("session log root: %s", session_log_root)
    logger.info("split: %s  (train=%d, valid=%d, model_slug=%s)",
                split_path.name, len(train_ids), len(valid_ids), model_slug)

    # initial state from disk
    states: dict[str, TaskState] = {
        tid: _scan_task_state(log_root, tid, args.logs_fresh_after)
        for tid in train_ids
    }
    for tid, st in states.items():
        logger.info("initial: %-35s verdict=%s", tid, st.verdict)

    # epoch loop
    sandbox_root = Path(args.sandbox_root).resolve() if args.sandbox_root else None
    existing_skill_pack = (out_root / _skill_dir_name(args.skill_id)).resolve()
    skill_pack_dir: Path | None = (
        existing_skill_pack
        if args.require_all_train_pass and (existing_skill_pack / "SKILL.md").exists()
        else None
    )
    if skill_pack_dir is not None:
        logger.info("resume skill pack for strict train gate: %s", skill_pack_dir)
    epoch_results: list[EpochResult] = []

    for epoch in range(1, args.max_epochs + 1):
        epoch_log_dir = session_log_root / f"epoch_{epoch:02d}"
        epoch_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 72)
        logger.info("EPOCH %d / %d", epoch, args.max_epochs)
        logger.info("  inject skill_pack: %s", skill_pack_dir or "(none)")
        logger.info("=" * 72)

        # Epoch 1 is normally a full domain sweep before any distillation.
        # For the strict train gate, resume from existing PASS evidence and
        # spend budget only on tasks that still lack a PASS.
        todo = (
            [tid for tid in train_ids if states[tid].verdict != "PASS"]
            if args.require_all_train_pass
            else list(train_ids)
            if epoch == 1
            else [tid for tid in train_ids if states[tid].verdict != "PASS"]
        )
        if not todo:
            logger.info("all train tasks already PASS; epoch loop done")
            break
        logger.info("epoch %d todo (%d): %s", epoch, len(todo), todo)

        for tid in todo:
            rc = _invoke_run_task(
                repo_root=repo_root,
                task_id=tid,
                model_id=args.model_id,
                llm_config=Path(args.llm_config).resolve() if args.llm_config else None,
                max_rounds=args.max_rounds,
                budget_seconds=args.budget_seconds,
                max_turns_per_round=args.max_turns_per_round,
                skill_pack_dir=skill_pack_dir,
                sandbox_root=sandbox_root,
                artifact_model_slug=args.artifact_model_slug,
                epoch_log_dir=epoch_log_dir,
            )
            states[tid].runs_done_this_session += 1
            # Re-scan disk to pick up the new run-*'s verdict.
            new_state = _scan_task_state(log_root, tid, args.logs_fresh_after)
            states[tid].verdict = new_state.verdict
            states[tid].last_run_id = new_state.last_run_id
            logger.info("after run %s: verdict=%s (rc=%d)",
                        tid, states[tid].verdict, rc)

        # Distill a domain-level skill from *all* current PASS episodes.
        rc_d = _invoke_distill_skill(
            repo_root=repo_root,
            split_path=split_path,
            skill_id=args.skill_id,
            out_root=out_root,
            model_id=args.model_id,
            llm_config=Path(args.llm_config).resolve() if args.llm_config else None,
            epoch_log_dir=epoch_log_dir,
        )
        if rc_d == 0:
            skill_pack_dir = (out_root / _skill_dir_name(args.skill_id)).resolve()
            logger.info("epoch %d distilled skill: %s", epoch, skill_pack_dir)
        else:
            logger.warning("epoch %d distill-skill returned rc=%d (no PASS yet?)",
                           epoch, rc_d)
            skill_pack_dir = skill_pack_dir  # keep last one if any

        epoch_results.append(EpochResult(
            epoch=epoch,
            pass_ids=[t for t, s in states.items() if s.verdict == "PASS"],
            fail_ids=[t for t, s in states.items() if s.verdict != "PASS"],
            skill_pack_dir=skill_pack_dir,
            per_task={t: s.verdict for t, s in states.items()},
        ))

        # All passed?
        if all(s.verdict == "PASS" for s in states.values()):
            logger.info("ALL TRAIN PASS at epoch %d", epoch)
            break

    # Final summary
    all_train_pass = all(s.verdict == "PASS" for s in states.values())
    if all_train_pass and skill_pack_dir is not None:
        final_log_dir = session_log_root / "final"
        final_log_dir.mkdir(parents=True, exist_ok=True)
        rc_d = _invoke_distill_skill(
            repo_root=repo_root,
            split_path=split_path,
            skill_id=args.skill_id,
            out_root=out_root,
            model_id=args.model_id,
            llm_config=Path(args.llm_config).resolve() if args.llm_config else None,
            epoch_log_dir=final_log_dir,
        )
        if rc_d == 0:
            skill_pack_dir = (out_root / _skill_dir_name(args.skill_id)).resolve()
            logger.info("final all-train PASS skill distilled: %s", skill_pack_dir)
        else:
            logger.warning("final distill-skill returned rc=%d; keeping previous pack", rc_d)

    final = {
        "split": str(split_path),
        "skill_id": args.skill_id,
        "skill_dir_name": _skill_dir_name(args.skill_id),
        "epochs_run": len(epoch_results),
        "train_final_verdicts": {t: s.verdict for t, s in states.items()},
        "train_pass_count": sum(1 for s in states.values() if s.verdict == "PASS"),
        "all_train_pass": all_train_pass,
        "skill_pack_dir": str(skill_pack_dir) if skill_pack_dir else None,
        "selected_valid_task": selected_valid_task,
        "session_log_root": str(session_log_root),
    }
    (session_log_root / "summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("train summary: %s", final)

    # Optional: validate one valid task using the final domain skill.
    if args.skip_validate:
        logger.info("validation skipped by --skip-validate")
    elif not all_train_pass:
        logger.info("validation skipped: not all train tasks PASS")
    elif skill_pack_dir is None:
        logger.info("validation skipped: no skill pack was distilled")
    elif selected_valid_task is None:
        logger.info("validation skipped: split has no valid tasks")
    else:
        rc_v = _invoke_validate_skill(
            repo_root=repo_root,
            split_path=split_path,
            skill_pack_dir=skill_pack_dir,
            model_id=args.model_id,
            llm_config=Path(args.llm_config).resolve() if args.llm_config else None,
            max_rounds=args.max_rounds,
            budget_seconds=args.budget_seconds,
            max_turns_per_round=args.max_turns_per_round,
            valid_task_id=selected_valid_task,
            artifact_model_slug=args.artifact_model_slug,
            epoch_log_dir=session_log_root,
        )
        logger.info("validate-skill rc=%d (see %s/validate_skill.log)",
                    rc_v, session_log_root)
        final["validate_rc"] = rc_v
        (session_log_root / "summary.json").write_text(
            json.dumps(final, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Exit non-zero if not all train PASS
    return 0 if all_train_pass else 1


if __name__ == "__main__":
    sys.exit(main())
