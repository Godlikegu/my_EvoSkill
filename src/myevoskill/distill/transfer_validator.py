"""Validation gate for a freshly distilled skill pack.

Given a skill pack on disk and the project's train/valid split, this
module can run the harness in two modes:

* skill-only validation, the default CLI behavior used by the domain train
  gate: one ``--skill-pack-dir`` run per selected valid task.
* compare validation, the original promote-or-reject gate: one baseline run
  and one skill run per selected valid task.

Promotion rule
--------------

Let ``B(t)`` be the baseline verdict on valid task ``t`` and ``S(t)`` the
+skill verdict on the same task. We require:

1. **No regression.**     For every ``t`` with ``B(t) == PASS``,
                          we must also have ``S(t) == PASS``.
                          Otherwise the skill *broke* something that used
                          to work and we reject.
2. **One new PASS.**      There must exist *at least one* ``t`` with
                          ``B(t) != PASS`` and ``S(t) == PASS``.
                          Otherwise the skill is, at best, a no-op and we
                          reject it (we don't promote skills that don't
                          earn their keep).

If both hold, the skill is **PROMOTED**: an audit ``promotion.json`` is
written next to the skill pack so future runs of the validator know it's
already gated. We never *delete* a rejected pack -- the operator can
inspect ``transfer_report.json`` to understand why.

Isolation
---------

This module *does* read valid-split logs (it has to, that's the whole
point), but it never reads valid-split *source code*. It uses the
``DistillUniverse`` only for the train-side check (``is_train`` /
``is_valid``) and for the ``model_slug``. The actual workspace build is
done by the harness, which already enforces public-only data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .universe import DistillUniverse

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- types


# A "runner" takes (task_id, *, with_skill: bool) and returns the verdict
# string ("PASS"/"FAIL"/"INVALID"/...). We use a callable so tests can
# inject a deterministic fake without spinning up Claude.
RunnerFn = Callable[[str, bool], str]


@dataclass(frozen=True)
class TaskComparison:
    task_id: str
    baseline_verdict: str
    plus_skill_verdict: str

    @property
    def is_regression(self) -> bool:
        return self.baseline_verdict == "PASS" and self.plus_skill_verdict != "PASS"

    @property
    def is_new_pass(self) -> bool:
        return self.baseline_verdict != "PASS" and self.plus_skill_verdict == "PASS"


@dataclass
class TransferReport:
    skill_pack_dir: Path
    valid_task_ids: tuple[str, ...]
    comparisons: list[TaskComparison] = field(default_factory=list)
    mode: str = "compare"
    verdict: str = "PENDING"  # PROMOTE | REJECT_REGRESSION | REJECT_NO_NEW_PASS
    rejection_reasons: list[str] = field(default_factory=list)
    started_ts: float = 0.0
    finished_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_pack_dir": str(self.skill_pack_dir),
            "valid_task_ids": list(self.valid_task_ids),
            "mode": self.mode,
            "verdict": self.verdict,
            "rejection_reasons": list(self.rejection_reasons),
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "comparisons": [
                {
                    "task_id": c.task_id,
                    "baseline_verdict": c.baseline_verdict,
                    "plus_skill_verdict": c.plus_skill_verdict,
                    "is_regression": c.is_regression,
                    "is_new_pass": c.is_new_pass,
                }
                for c in self.comparisons
            ],
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, int]:
        return {
            "n_valid": len(self.comparisons),
            "n_baseline_pass": sum(1 for c in self.comparisons if c.baseline_verdict == "PASS"),
            "n_plus_skill_pass": sum(1 for c in self.comparisons if c.plus_skill_verdict == "PASS"),
            "n_new_pass": sum(1 for c in self.comparisons if c.is_new_pass),
            "n_regression": sum(1 for c in self.comparisons if c.is_regression),
        }


# --------------------------------------------------------------------------- core


VERDICT_PROMOTE = "PROMOTE"
VERDICT_REJECT_REGRESSION = "REJECT_REGRESSION"
VERDICT_REJECT_NO_NEW_PASS = "REJECT_NO_NEW_PASS"


def validate_skill(
    *,
    universe: DistillUniverse,
    skill_pack_dir: Path,
    runner: RunnerFn,
    valid_task_ids: Sequence[str] | None = None,
    compare_baseline: bool = True,
) -> TransferReport:
    """Run valid-split validation.

    ``runner(task_id, with_skill)`` is the only side-effect this function
    has. It is expected to:
      * build the workspace,
      * (when ``with_skill`` is True) inject ``skill_pack_dir`` via
        ``HarnessConfig.skill_pack_dir``,
      * run the harness end-to-end,
      * return the final verdict string.

    When ``compare_baseline`` is true, baseline and skill calls share *no*
    state -- the harness already isolates per-run sandboxes -- so order is
    irrelevant. When false, only the skill call is made.
    """

    pack_dir = Path(skill_pack_dir).resolve()
    if not (pack_dir / "SKILL.md").exists():
        raise FileNotFoundError(
            f"skill pack does not look valid (missing SKILL.md): {pack_dir}"
        )

    valid_ids = tuple(valid_task_ids if valid_task_ids is not None else universe.valid_task_ids)
    if not valid_ids:
        raise ValueError(
            "validate_skill requires at least one valid-split task; "
            "the universe has none configured."
        )
    # Guard: we are *not* allowed to validate against train tasks
    # (otherwise we'd be measuring memorisation, not transfer).
    train_overlap = [t for t in valid_ids if universe.is_train(t)]
    if train_overlap:
        raise PermissionError(
            f"validate_skill refuses to evaluate against train tasks: {train_overlap}"
        )

    report = TransferReport(
        skill_pack_dir=pack_dir,
        valid_task_ids=valid_ids,
        mode="compare" if compare_baseline else "skill_only",
        started_ts=time.time(),
    )

    for task_id in valid_ids:
        if compare_baseline:
            logger.info("[validate] %s: baseline run starting", task_id)
            baseline = runner(task_id, False)
            logger.info("[validate] %s: baseline=%s", task_id, baseline)
        else:
            baseline = "SKIPPED"

        logger.info("[validate] %s: +skill run starting", task_id)
        with_skill = runner(task_id, True)
        logger.info("[validate] %s: +skill=%s", task_id, with_skill)

        report.comparisons.append(
            TaskComparison(
                task_id=task_id,
                baseline_verdict=str(baseline),
                plus_skill_verdict=str(with_skill),
            )
        )

    # Decide. Skill-only validation is a strict final gate: every selected
    # valid task must pass with the skill. Compare mode keeps the original
    # transfer-improvement rule.
    if not compare_baseline:
        failed = [c for c in report.comparisons if c.plus_skill_verdict != "PASS"]
        if failed:
            report.verdict = VERDICT_REJECT_NO_NEW_PASS
            report.rejection_reasons.append(
                "skill validation failed on " + ",".join(c.task_id for c in failed)
            )
        else:
            report.verdict = VERDICT_PROMOTE
    else:
        regressions = [c for c in report.comparisons if c.is_regression]
        new_passes = [c for c in report.comparisons if c.is_new_pass]

        if regressions:
            report.verdict = VERDICT_REJECT_REGRESSION
            report.rejection_reasons.append(
                "regression on " + ",".join(c.task_id for c in regressions)
            )
        elif not new_passes:
            report.verdict = VERDICT_REJECT_NO_NEW_PASS
            report.rejection_reasons.append("no previously-failing task was rescued by the skill")
        else:
            report.verdict = VERDICT_PROMOTE

    report.finished_ts = time.time()
    return report


def write_transfer_report(
    report: TransferReport,
    out_path: Path | None = None,
) -> Path:
    """Persist a TransferReport next to the skill pack (or wherever)."""

    target = (
        Path(out_path).resolve()
        if out_path is not None
        else report.skill_pack_dir.parent / f"{report.skill_pack_dir.name}.transfer.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def stamp_promotion(report: TransferReport, *, model_slug: str) -> Path | None:
    """If the report PROMOTEs, drop a ``promotion.json`` inside the pack.

    Returns the path written, or ``None`` if the report did not promote.
    """

    if report.verdict != VERDICT_PROMOTE:
        return None
    promotion = {
        "promoted_ts": time.time(),
        "model_slug": model_slug,
        "summary": report.summary(),
        "valid_task_ids": list(report.valid_task_ids),
    }
    target = report.skill_pack_dir / "promotion.json"
    target.write_text(
        json.dumps(promotion, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


__all__ = [
    "RunnerFn",
    "TaskComparison",
    "TransferReport",
    "VERDICT_PROMOTE",
    "VERDICT_REJECT_NO_NEW_PASS",
    "VERDICT_REJECT_REGRESSION",
    "stamp_promotion",
    "validate_skill",
    "write_transfer_report",
]
