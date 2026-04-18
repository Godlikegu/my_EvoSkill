from myevoskill.models import TaskOutcome
from myevoskill.validation import TransferValidator


def make_outcome(task_id: str, passed: bool) -> TaskOutcome:
    return TaskOutcome(task_id=task_id, all_metrics_passed=passed)


def test_transfer_validator_promotes_strict_improvement():
    validator = TransferValidator()
    result = validator.evaluate(
        baseline_outcomes={
            "a": make_outcome("a", False),
            "b": make_outcome("b", True),
        },
        treatment_outcomes={
            "a": make_outcome("a", True),
            "b": make_outcome("b", True),
        },
    )
    assert result.monotonic_non_regression is True
    assert result.promotable is True
    assert result.new_successes == ["a"]
    assert result.regressions == []
    assert result.decision == "validated"


def test_transfer_validator_rejects_regression():
    validator = TransferValidator()
    result = validator.evaluate(
        baseline_outcomes={
            "a": make_outcome("a", True),
            "b": make_outcome("b", False),
        },
        treatment_outcomes={
            "a": make_outcome("a", False),
            "b": make_outcome("b", True),
        },
    )
    assert result.monotonic_non_regression is False
    assert result.promotable is False
    assert result.regressions == ["a"]
    assert result.decision == "rejected"


def test_transfer_validator_keeps_tie_as_draft():
    validator = TransferValidator()
    result = validator.evaluate(
        baseline_outcomes={
            "a": make_outcome("a", True),
            "b": make_outcome("b", False),
        },
        treatment_outcomes={
            "a": make_outcome("a", True),
            "b": make_outcome("b", False),
        },
    )
    assert result.monotonic_non_regression is True
    assert result.promotable is False
    assert result.decision == "draft"

