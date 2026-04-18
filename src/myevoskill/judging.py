"""Hidden judge helpers with all-metrics-pass semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .models import JudgeResult


@dataclass(frozen=True)
class MetricRequirement:
    """Single metric threshold specification."""

    name: str
    threshold: float
    operator: str

    def is_satisfied(self, actual: float) -> bool:
        if self.operator == ">=":
            return actual >= self.threshold
        if self.operator == "<=":
            return actual <= self.threshold
        raise ValueError(f"unsupported metric operator: {self.operator}")


class HiddenJudge:
    """Compute the minimal hidden judge result contract."""

    def evaluate(
        self,
        task_id: str,
        metrics_actual: Mapping[str, float],
        requirements: Sequence[MetricRequirement],
        failure_tags: Iterable[str] | None = None,
    ) -> JudgeResult:
        failed_metrics: List[str] = []
        for requirement in requirements:
            if requirement.name not in metrics_actual:
                failed_metrics.append(requirement.name)
                continue
            if not requirement.is_satisfied(metrics_actual[requirement.name]):
                failed_metrics.append(requirement.name)
        return JudgeResult(
            task_id=task_id,
            all_metrics_passed=not failed_metrics,
            metrics_actual=dict(metrics_actual),
            failed_metrics=failed_metrics,
            failure_tags=list(failure_tags or []),
        )

