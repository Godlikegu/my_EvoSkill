"""Skill registry that only permanently stores promotable skills."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .models import SkillCandidate, SkillRegistryEntry
from .validation import TransferValidationResult


class SkillRegistry:
    """Persist skill records according to promotion policy."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def decide(self, candidate: SkillCandidate, validation: TransferValidationResult) -> str:
        if not candidate.legal_source:
            return "rejected"
        if not candidate.reusable:
            return "rejected"
        return validation.decision

    def register(
        self, candidate: SkillCandidate, validation: TransferValidationResult
    ) -> SkillRegistryEntry:
        status = self.decide(candidate, validation)
        if status == "rejected" and candidate.legal_source and candidate.reusable:
            reason = validation.decision_reason
        elif status == "rejected" and not candidate.legal_source:
            reason = "candidate rejected: illegal source"
        elif status == "rejected":
            reason = "candidate rejected: not reusable"
        else:
            reason = validation.decision_reason

        record = SkillRegistryEntry(
            skill_id=candidate.skill_id,
            version=candidate.version,
            status=status,
            description=candidate.description,
            origin_runs=list(candidate.source_run_ids),
            baseline_successes=validation.baseline_successes,
            treatment_successes=validation.treatment_successes,
            new_successes=validation.new_successes,
            regressions=validation.regressions,
            decision_reason=reason,
            validation_result={
                "decision": validation.decision,
                "monotonic_non_regression": validation.monotonic_non_regression,
                "promotable": validation.promotable,
            },
            known_failure_modes=list(candidate.known_failure_modes),
            known_bad_triggers=list(candidate.known_bad_triggers),
            parent_skill_id=candidate.parent_skill_id,
            metadata=dict(candidate.evidence),
            registry_path=self.root / f"{candidate.skill_id}.json",
        )
        payload = asdict(record)
        payload["registry_path"] = str(record.registry_path) if record.registry_path else ""
        record.registry_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return record

    def upsert(
        self,
        existing_skill_id: Optional[str],
        candidate: SkillCandidate,
        validation: TransferValidationResult,
    ) -> SkillRegistryEntry:
        merged_candidate = candidate
        if existing_skill_id:
            merged_candidate = SkillCandidate(
                skill_id=candidate.skill_id,
                version=candidate.version,
                description=candidate.description,
                source_run_ids=candidate.source_run_ids,
                legal_source=candidate.legal_source,
                reusable=candidate.reusable,
                applicability=candidate.applicability,
                known_failure_modes=candidate.known_failure_modes,
                known_bad_triggers=candidate.known_bad_triggers,
                parent_skill_id=existing_skill_id,
                evidence=candidate.evidence,
            )
        return self.register(merged_candidate, validation)
