from myevoskill.judging import HiddenJudge, MetricRequirement


def test_hidden_judge_requires_all_metrics_to_pass():
    judge = HiddenJudge()
    result = judge.evaluate(
        task_id="task-1",
        metrics_actual={"acc": 0.92, "loss": 0.08},
        requirements=[
            MetricRequirement("acc", 0.9, ">="),
            MetricRequirement("loss", 0.1, "<="),
        ],
    )
    assert result.all_metrics_passed is True
    assert result.failed_metrics == []


def test_hidden_judge_fails_if_any_metric_fails():
    judge = HiddenJudge()
    result = judge.evaluate(
        task_id="task-1",
        metrics_actual={"acc": 0.92, "loss": 0.15},
        requirements=[
            MetricRequirement("acc", 0.9, ">="),
            MetricRequirement("loss", 0.1, "<="),
        ],
        failure_tags=["underfit"],
    )
    assert result.all_metrics_passed is False
    assert result.failed_metrics == ["loss"]
    assert result.failure_tags == ["underfit"]

