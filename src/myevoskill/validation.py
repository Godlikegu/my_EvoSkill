"""Transfer validation with monotonic non-regression enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from .models import TaskOutcome


@dataclass(frozen=True)
class TransferValidationResult:
    """Validation summary for baseline vs with-skill comparisons."""

    baseline_successes: List[str]
    treatment_successes: List[str]
    new_successes: List[str]
    regressions: List[str]
    monotonic_non_regression: bool
    promotable: bool
    decision: str
    decision_reason: str


class TransferValidator:
    """Validate transfer with a strict subset-based promotion rule."""

    def evaluate(
        self,
        baseline_outcomes: Mapping[str, TaskOutcome],
        treatment_outcomes: Mapping[str, TaskOutcome],
    ) -> TransferValidationResult:
        baseline_ids = set(baseline_outcomes.keys())
        treatment_ids = set(treatment_outcomes.keys())
        if baseline_ids != treatment_ids:
            raise ValueError("baseline and treatment task ids must match exactly")

        baseline_successes = sorted(
            task_id
            for task_id, outcome in baseline_outcomes.items()
            if outcome.all_metrics_passed
        )
        treatment_successes = sorted(
            task_id
            for task_id, outcome in treatment_outcomes.items()
            if outcome.all_metrics_passed
        )

        s0 = set(baseline_successes)
        s1 = set(treatment_successes)
        regressions = sorted(s0 - s1)
        new_successes = sorted(s1 - s0)
        monotonic = not regressions
        promotable = monotonic and bool(new_successes)

        if regressions:
            decision = "rejected"
            reason = f"regressions detected: {', '.join(regressions)}"
        elif new_successes:
            decision = "validated"
            reason = f"strict improvement with new successes: {', '.join(new_successes)}"
        else:
            decision = "draft"
            reason = "non-regressive but no new validation successes"

        return TransferValidationResult(
            baseline_successes=baseline_successes,
            treatment_successes=treatment_successes,
            new_successes=new_successes,
            regressions=regressions,
            monotonic_non_regression=monotonic,
            promotable=promotable,
            decision=decision,
            decision_reason=reason,
        )

