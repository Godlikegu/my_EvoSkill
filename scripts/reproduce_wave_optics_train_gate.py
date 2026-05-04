"""Chunked isolated runner for the wave-optics train gate.

This wrapper is for operator experiments where the current global log tree
already contains 6/6 PASS and would otherwise make `train_domain_skill.py`
finish immediately. It creates a temporary split whose `model_slug` points at
an isolated artifact namespace, starts from a clean 0/6 train state by default,
and repeatedly invokes `train_domain_skill.py` in small chunks until all train
tasks PASS. Historical 2/6 seeding is still available as an explicit option.

Example:

    conda run -n evoskill python scripts/reproduce_wave_optics_train_gate.py \
      --repo-root . \
      --model-id "Vendor2/Claude-4.6-opus" \
      --epochs-per-call 1 \
      --max-total-epochs 0 \
      --skip-validate
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("reproduce_wave_optics_train_gate")


DEFAULT_SEED_RUNS = {
    "seismic_lsrtm_original": "run-1777365437-b08375",
    "ultrasound_sos_tomography": "run-1777377521-6473ec",
}


def _skill_dir_name(skill_id: str) -> str:
    raw = skill_id.lower().replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", raw)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:64].strip("-") or "untitled-skill"


def _python_exe() -> str:
    return sys.executable


def _copytree_fresh(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _patch_summary_paths(summary_path: Path, *, log_root: Path, workspace_root: Path) -> None:
    if not summary_path.exists():
        return
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return
    data["log_root"] = str(log_root)
    data["workspace_root"] = str(workspace_root)
    policy = data.get("policy")
    if isinstance(policy, dict):
        policy["agent_root"] = str(workspace_root)
    summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _seed_initial_passes(
    *,
    repo_root: Path,
    base_model_slug: str,
    artifact_slug: str,
    seed_runs: dict[str, str],
) -> list[str]:
    copied: list[str] = []
    for task_id, run_id in seed_runs.items():
        src_log = repo_root / "artifacts" / "logs" / base_model_slug / task_id / run_id
        src_ws = repo_root / "artifacts" / "workspaces" / base_model_slug / task_id / run_id
        if not src_log.exists():
            raise FileNotFoundError(f"seed log not found: {src_log}")
        if not src_ws.exists():
            raise FileNotFoundError(f"seed workspace not found: {src_ws}")

        dst_log = repo_root / "artifacts" / "logs" / artifact_slug / task_id / run_id
        dst_ws = repo_root / "artifacts" / "workspaces" / artifact_slug / task_id / run_id
        _copytree_fresh(src_log, dst_log)
        _copytree_fresh(src_ws, dst_ws)
        _patch_summary_paths(dst_log / "run_summary.json", log_root=dst_log, workspace_root=dst_ws)
        _patch_summary_paths(dst_log / "summary.json", log_root=dst_log, workspace_root=dst_ws)
        copied.append(f"{task_id}/{run_id}")
    return copied


def _write_isolated_split(
    *,
    repo_root: Path,
    base_split_path: Path,
    artifact_slug: str,
) -> Path:
    split = json.loads(base_split_path.read_text(encoding="utf-8"))
    split["model_slug"] = artifact_slug
    split["description"] = (
        str(split.get("description") or "")
        + f" Isolated reproduction artifact slug: {artifact_slug}."
    ).strip()
    out = repo_root / "registry" / "splits" / f"wave_optics_v1_repro_{artifact_slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _latest_repro_summary(repo_root: Path, split_path: Path, started_at: float) -> dict | None:
    candidates = []
    root = repo_root / "artifacts" / "logs" / "_domain_train"
    for path in root.glob("wave_optics_recon_v1_*/summary.json"):
        if path.stat().st_mtime < started_at:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(Path(data.get("split", "")).resolve()) == str(split_path.resolve()):
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return None
    _, path, data = sorted(candidates)[-1]
    data["_summary_path"] = str(path)
    return data


def _run_train_call(
    *,
    repo_root: Path,
    split_path: Path,
    skill_id: str,
    out_root: Path,
    model_id: str,
    llm_config: Path | None,
    artifact_slug: str,
    epochs: int,
    max_rounds: int,
    budget_seconds: int,
    max_turns_per_round: int,
    valid_task_id: str | None,
    skip_validate: bool,
    call_index: int,
    outer_log: Path,
) -> int:
    cmd = [
        _python_exe(),
        str(repo_root / "scripts" / "train_domain_skill.py"),
        "--repo-root", str(repo_root),
        "--split", str(split_path),
        "--skill-id", skill_id,
        "--out-root", str(out_root),
        "--model-id", model_id,
        "--artifact-model-slug", artifact_slug,
        "--max-epochs", str(epochs),
        "--max-rounds", str(max_rounds),
        "--budget-seconds", str(budget_seconds),
        "--max-turns-per-round", str(max_turns_per_round),
        "--require-all-train-pass",
    ]
    if llm_config is not None:
        cmd += ["--llm-config", str(llm_config)]
    if valid_task_id:
        cmd += ["--valid-task-id", valid_task_id]
    if skip_validate:
        cmd += ["--skip-validate"]

    outer_log.parent.mkdir(parents=True, exist_ok=True)
    with outer_log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== CALL {call_index} =====\n")
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        fh.write(f"\nCALL {call_index} rc={proc.returncode}\n")
        return proc.returncode


def _run_valid_gate(
    *,
    repo_root: Path,
    split_path: Path,
    skill_pack_dir: Path,
    model_id: str,
    llm_config: Path | None,
    artifact_slug: str,
    max_rounds: int,
    budget_seconds: int,
    max_turns_per_round: int,
    min_valid_pass: int,
    outer_log: Path,
) -> int:
    report_path = (
        repo_root
        / "artifacts"
        / "logs"
        / "_valid_runs"
        / f"{artifact_slug}_e2e"
        / "transfer_report.json"
    )
    valid_slug = f"{artifact_slug}_valid_{time.strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        _python_exe(),
        "-m", "myevoskill.cli",
        "validate-skill",
        "--repo-root", str(repo_root),
        "--split", str(split_path),
        "--skill-pack-dir", str(skill_pack_dir),
        "--model-id", model_id,
        "--artifact-model-slug", valid_slug,
        "--max-rounds", str(max_rounds),
        "--budget-seconds", str(budget_seconds),
        "--max-turns-per-round", str(max_turns_per_round),
        "--report-path", str(report_path),
        "--json",
    ]
    if llm_config is not None:
        cmd += ["--llm-config", str(llm_config)]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with outer_log.open("a", encoding="utf-8") as fh:
        fh.write("\n\n===== E2E VALID GATE =====\n")
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        fh.write(f"\nE2E VALID rc={proc.returncode}\n")

    if not report_path.exists():
        logger.error("valid gate did not write report: %s", report_path)
        return proc.returncode or 1
    diagnosis_path = report_path.parent / "diagnosis.md"
    diag_cmd = [
        _python_exe(),
        str(repo_root / "scripts" / "diagnose_valid_failures.py"),
        "--log-root", str(repo_root / "artifacts" / "logs" / valid_slug),
        "--out", str(diagnosis_path),
    ]
    diag = subprocess.run(diag_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if diag.returncode == 0:
        logger.info("valid failure diagnosis: %s", diagnosis_path)
    else:
        logger.warning("valid failure diagnosis failed: %s", (diag.stderr or diag.stdout).strip())
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("valid gate report is not valid JSON: %s (%s)", report_path, exc)
        return 1
    summary = report.get("summary") or {}
    n_plus = int(summary.get("n_plus_skill_pass") or 0)
    logger.info(
        "E2E valid gate: n_plus_skill_pass=%d min_required=%d report=%s",
        n_plus,
        min_valid_pass,
        report_path,
    )
    if n_plus < min_valid_pass:
        logger.error(
            "E2E transfer failed: valid pass count %d < required %d",
            n_plus,
            min_valid_pass,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isolated chunked runner for wave_optics train 0/6 -> 6/6."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split", default="registry/splits/wave_optics_v1.json")
    parser.add_argument("--skill-id", default="wave_optics_recon_v1")
    parser.add_argument("--model-id", default="Vendor2/Claude-4.6-opus")
    parser.add_argument("--llm-config", default=None)
    parser.add_argument("--artifact-model-slug", default=None,
                        help="default: Vendor2_Claude-4.6-opus_repro_<timestamp>")
    parser.add_argument("--out-root", default=None,
                        help="default: artifacts/skills_repro/<artifact_slug>")
    parser.add_argument("--max-total-epochs", type=int, default=0,
                        help="total epoch cap; 0 means keep running until 6/6 PASS")
    parser.add_argument("--epochs-per-call", type=int, default=1,
                        help="small restartable chunk size for each train_domain_skill.py call")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--budget-seconds", type=int, default=5400)
    parser.add_argument("--max-turns-per-round", type=int, default=60)
    parser.add_argument("--valid-task-id", default="reflection_ODT",
                        help="kept for compatibility; final E2E gate validates the full valid split")
    parser.add_argument("--skip-validate", action="store_true",
                        help="skip the final full-valid E2E transfer gate")
    parser.add_argument("--min-valid-pass", type=int, default=1,
                        help="E2E success requires at least this many valid tasks to PASS with skill")
    parser.add_argument("--seed-initial-passes", action="store_true",
                        help="seed the historical 2/6 checkpoint instead of starting from clean 0/6")
    parser.add_argument("--no-seed-initial-passes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sleep-seconds-between-calls", type=float, default=0.0)
    parser.add_argument("--max-consecutive-crashes", type=int, default=3,
                        help="stop after this many child calls return an abnormal rc with no summary")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.epochs_per_call <= 0:
        parser.error("--epochs-per-call must be positive")
    if args.max_total_epochs < 0:
        parser.error("--max-total-epochs must be >= 0")
    if args.min_valid_pass < 0:
        parser.error("--min-valid-pass must be >= 0")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    repo_root = Path(args.repo_root).resolve()
    base_split_path = Path(args.split)
    if not base_split_path.is_absolute():
        base_split_path = repo_root / base_split_path
    base_split = json.loads(base_split_path.read_text(encoding="utf-8"))
    base_model_slug = str(base_split.get("model_slug") or "Vendor2_Claude-4.6-opus")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_slug = args.artifact_model_slug or f"{base_model_slug}_repro_{stamp}"
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else repo_root / "artifacts" / "skills_repro" / artifact_slug
    )
    out_root.mkdir(parents=True, exist_ok=True)
    split_path = _write_isolated_split(
        repo_root=repo_root,
        base_split_path=base_split_path,
        artifact_slug=artifact_slug,
    )

    seeded: list[str] = []
    if args.seed_initial_passes and not args.no_seed_initial_passes:
        seeded = _seed_initial_passes(
            repo_root=repo_root,
            base_model_slug=base_model_slug,
            artifact_slug=artifact_slug,
            seed_runs=DEFAULT_SEED_RUNS,
        )

    started_at = time.time()
    outer_log = (
        repo_root
        / "artifacts"
        / "logs"
        / "_domain_train"
        / f"{args.skill_id}_{stamp}_repro.outer.log"
    )
    manifest = {
        "artifact_model_slug": artifact_slug,
        "base_split": str(base_split_path),
        "isolated_split": str(split_path),
        "out_root": str(out_root),
        "seeded_pass_runs": seeded,
        "started_at": started_at,
        "outer_log": str(outer_log),
        "epochs_per_call": args.epochs_per_call,
        "max_total_epochs": args.max_total_epochs,
        "clean_start": not bool(seeded),
        "skip_validate": bool(args.skip_validate),
        "min_valid_pass": args.min_valid_pass,
    }
    manifest_path = outer_log.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("isolated artifact slug: %s", artifact_slug)
    logger.info("isolated split: %s", split_path)
    logger.info("skill out root: %s", out_root)
    logger.info("seeded PASS runs: %s", seeded or "(none; clean 0/6 start)")
    logger.info("outer log: %s", outer_log)

    epochs_used = 0
    call_index = 0
    last_rc = 1
    consecutive_crashes = 0
    while args.max_total_epochs == 0 or epochs_used < args.max_total_epochs:
        call_index += 1
        if args.max_total_epochs == 0:
            epochs = args.epochs_per_call
        else:
            epochs = min(args.epochs_per_call, args.max_total_epochs - epochs_used)
        last_rc = _run_train_call(
            repo_root=repo_root,
            split_path=split_path,
            skill_id=args.skill_id,
            out_root=out_root,
            model_id=args.model_id,
            llm_config=Path(args.llm_config).resolve() if args.llm_config else None,
            artifact_slug=artifact_slug,
            epochs=epochs,
            max_rounds=args.max_rounds,
            budget_seconds=args.budget_seconds,
            max_turns_per_round=args.max_turns_per_round,
            valid_task_id=args.valid_task_id,
            # The wrapper owns the E2E gate. Child train calls skip their
            # single-task validation so the final skill is evaluated once
            # against the full valid split after train reaches 6/6.
            skip_validate=True,
            call_index=call_index,
            outer_log=outer_log,
        )
        epochs_used += epochs
        summary = _latest_repro_summary(repo_root, split_path, started_at)
        if summary is not None:
            logger.info(
                "after call %d: pass=%s/%s all=%s summary=%s",
                call_index,
                summary.get("train_pass_count"),
                len(base_split.get("train") or []),
                summary.get("all_train_pass"),
                summary.get("_summary_path"),
            )
            if summary.get("all_train_pass"):
                logger.info("reproduction reached all train PASS")
                if args.skip_validate:
                    logger.info("E2E valid gate skipped by --skip-validate")
                    return 0
                skill_pack_dir = out_root / _skill_dir_name(args.skill_id)
                if not (skill_pack_dir / "SKILL.md").exists():
                    logger.error("final skill pack missing: %s", skill_pack_dir)
                    return 1
                return _run_valid_gate(
                    repo_root=repo_root,
                    split_path=split_path,
                    skill_pack_dir=skill_pack_dir,
                    model_id=args.model_id,
                    llm_config=Path(args.llm_config).resolve() if args.llm_config else None,
                    artifact_slug=artifact_slug,
                    max_rounds=args.max_rounds,
                    budget_seconds=args.budget_seconds,
                    max_turns_per_round=args.max_turns_per_round,
                    min_valid_pass=args.min_valid_pass,
                    outer_log=outer_log,
                )
            consecutive_crashes = 0
        elif last_rc not in (0, 1):
            consecutive_crashes += 1
            logger.warning("call %d returned rc=%d and wrote no summary", call_index, last_rc)
            if consecutive_crashes >= args.max_consecutive_crashes:
                logger.error(
                    "stopping after %d consecutive abnormal child exits with no summary",
                    consecutive_crashes,
                )
                return last_rc or 1

        if args.sleep_seconds_between_calls > 0:
            time.sleep(args.sleep_seconds_between_calls)

    logger.error("runner exhausted max-total-epochs=%d without all train PASS", args.max_total_epochs)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
