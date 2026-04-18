import json

from myevoskill.models import SkillCandidate
from myevoskill.registry import SkillRegistry
from myevoskill.validation import TransferValidationResult


def make_candidate(legal=True, reusable=True):
    return SkillCandidate(
        skill_id="skill-alpha",
        version="0.1.0",
        description="Reusable debug workflow",
        source_run_ids=["run-1"],
        legal_source=legal,
        reusable=reusable,
        known_failure_modes=["runtime_budget_exceeded"],
        known_bad_triggers=["single-task-hardcode"],
    )


def make_validation(decision: str) -> TransferValidationResult:
    if decision == "validated":
        return TransferValidationResult(
            baseline_successes=["task-1"],
            treatment_successes=["task-1", "task-2"],
            new_successes=["task-2"],
            regressions=[],
            monotonic_non_regression=True,
            promotable=True,
            decision="validated",
            decision_reason="strict improvement with new successes: task-2",
        )
    if decision == "draft":
        return TransferValidationResult(
            baseline_successes=["task-1"],
            treatment_successes=["task-1"],
            new_successes=[],
            regressions=[],
            monotonic_non_regression=True,
            promotable=False,
            decision="draft",
            decision_reason="non-regressive but no new validation successes",
        )
    return TransferValidationResult(
        baseline_successes=["task-1"],
        treatment_successes=[],
        new_successes=[],
        regressions=["task-1"],
        monotonic_non_regression=False,
        promotable=False,
        decision="rejected",
        decision_reason="regressions detected: task-1",
    )


def test_registry_only_validates_promotable_skill(tmp_path):
    registry = SkillRegistry(tmp_path)
    record = registry.register(make_candidate(), make_validation("validated"))
    assert record.status == "validated"
    payload = json.loads(record.registry_path.read_text(encoding="utf-8"))
    assert payload["status"] == "validated"
    assert payload["new_successes"] == ["task-2"]
    assert payload["known_failure_modes"] == ["runtime_budget_exceeded"]


def test_registry_keeps_tie_as_draft(tmp_path):
    registry = SkillRegistry(tmp_path)
    record = registry.register(make_candidate(), make_validation("draft"))
    assert record.status == "draft"


def test_registry_rejects_illegal_or_regressive_skill(tmp_path):
    registry = SkillRegistry(tmp_path)
    record = registry.register(make_candidate(legal=False), make_validation("validated"))
    assert record.status == "rejected"

    record2 = registry.register(make_candidate(), make_validation("rejected"))
    assert record2.status == "rejected"


def test_registry_upsert_tracks_parent_skill(tmp_path):
    registry = SkillRegistry(tmp_path)
    record = registry.upsert("skill-base", make_candidate(), make_validation("validated"))
    assert record.parent_skill_id == "skill-base"
