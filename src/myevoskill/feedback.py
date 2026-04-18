"""Skill distillation and surrogate feedback helpers."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .models import RunRecord, SkillCandidate


class SurrogateFeedbackBuilder:
    """Summarize repeated proxy and runtime issues without leaking private data."""

    def summarize(self, run_records: Sequence[RunRecord]) -> str:
        warnings = Counter()
        for record in run_records:
            if record.proxy_feedback:
                warnings.update(record.proxy_feedback.warnings)
        if not warnings:
            return "No repeated low-leakage proxy issues detected."
        parts = [f"{warning} x{count}" for warning, count in warnings.most_common()]
        return "Repeated proxy issues: " + ", ".join(parts)


class SkillDistiller:
    """Create a candidate skill skeleton from reusable successful runs."""

    def distill(
        self,
        skill_id: str,
        description: str,
        source_runs: Sequence[RunRecord],
        legal_source: bool,
        reusable: bool,
    ) -> SkillCandidate:
        failure_modes = sorted(
            {
                tag
                for record in source_runs
                for tag in (record.judge_result.failure_tags if record.judge_result else [])
            }
        )
        return SkillCandidate(
            skill_id=skill_id,
            version="0.1.0",
            description=description,
            source_run_ids=[record.run_id for record in source_runs],
            legal_source=legal_source,
            reusable=reusable,
            applicability=["scientific_task"],
            known_failure_modes=failure_modes,
            known_bad_triggers=[],
        )
