from myevoskill.feedback import SkillDistiller, SurrogateFeedbackBuilder
from myevoskill.models import ProxyFeedback, RunRecord


def test_surrogate_feedback_builder_summarizes_repeated_warnings(tmp_path):
    run = RunRecord(
        run_id="run-1",
        task_id="task-1",
        provider="fallback",
        env_hash="env-1",
        skills_active=[],
        workspace_root=tmp_path,
        proxy_feedback=ProxyFeedback(
            task_id="task-1",
            output_exists=False,
            warnings=["missing output artifact"],
        ),
    )
    summary = SurrogateFeedbackBuilder().summarize([run, run])
    assert "missing output artifact x2" in summary


def test_skill_distiller_collects_failure_modes(tmp_path):
    run = RunRecord(
        run_id="run-1",
        task_id="task-1",
        provider="fallback",
        env_hash="env-1",
        skills_active=[],
        workspace_root=tmp_path,
    )
    candidate = SkillDistiller().distill(
        skill_id="skill-alpha",
        description="reusable workflow",
        source_runs=[run],
        legal_source=True,
        reusable=True,
    )
    assert candidate.skill_id == "skill-alpha"
    assert candidate.legal_source is True
