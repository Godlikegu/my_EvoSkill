from __future__ import annotations

from pathlib import Path

from myevoskill.distill.episode_miner import TaskEpisode, ToolUseEpisode
from myevoskill.distill.skill_synthesizer import (
    SanitizationError,
    collect_train_gap_evidence,
    synthesize_skill,
    write_skill_pack,
)
from myevoskill.distill.universe import DistillUniverse


def test_polished_ground_truth_phrase_is_reworded_before_write(tmp_path: Path) -> None:
    universe = DistillUniverse(
        repo_root=tmp_path,
        train_task_ids=("train_wave_task",),
        valid_task_ids=("valid_wave_task",),
        model_slug="Vendor2_Claude-4.6-opus",
    )
    episode = TaskEpisode(
        task_id="train_wave_task",
        run_id="run-1",
        family="wave_physics",
        final_verdict="PASS",
        rounds_used=1,
        runtime_seconds=1.0,
        tool_uses=(
            ToolUseEpisode(
                round_index=1,
                tool="Bash",
                input_signature="python work/main.py",
                success=True,
                referenced_files=("work/main.py",),
            ),
        ),
        failure_signals=(),
        primary_output_rel="output/reconstruction.npz",
    )

    def polish(_body: str, _evidence: object) -> str:
        return (
            "## Routes\n"
            "| public signal | algorithm route | required checks |\n"
            "| --- | --- | --- |\n"
            "| inverse problem | public forward model | inspect arrays |\n\n"
            "## Metric Diagnostic\n"
            "| failed metric signal | first diagnosis | next action | give-up signal |\n"
            "| --- | --- | --- | --- |\n"
            "| structure fails | geometry | probe | unchanged output |\n\n"
            "## Anti-Patterns\n"
            "- Do not repeat a failed guard.\n\n"
            "## Self-check\nCompare with ground truth only if it is public.\n"
        )

    spec = synthesize_skill(
        skill_id="wave_optics_recon_v1",
        episodes=(episode,),
        universe=universe,
        llm_polish=polish,
    )
    pack_dir = write_skill_pack(
        spec,
        tmp_path / "skills",
        valid_task_ids=universe.valid_task_ids,
        train_task_ids=universe.train_task_ids,
    )

    text = (pack_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "ground truth" not in text.lower()
    assert "reference solution" in text
    assert pack_dir.name == "wave-optics-recon-v1"


def test_train_gap_evidence_reads_failed_train_source_only(tmp_path: Path) -> None:
    repo = tmp_path / "MyEvoSkill"
    tasks = tmp_path / "tasks"
    repo.mkdir()
    for task_id in ("train_pass", "train_fail", "valid_task"):
        task_dir = tasks / task_id
        (task_dir / "src").mkdir(parents=True)
        (task_dir / "README.md").write_text(
            "# Wave task\nUse a source-grounded inverse solver.\n",
            encoding="utf-8",
        )
        (task_dir / "src" / "main.py").write_text(
            "def solve(measurements):\n    return measurements\n",
            encoding="utf-8",
        )
        (task_dir / "src" / "solvers.py").write_text(
            "def stolt(measurements):\n    return measurements\n",
            encoding="utf-8",
        )

    run = repo / "artifacts" / "logs" / "Model" / "train_fail" / "run-2"
    run.mkdir(parents=True)
    (run / "run_summary.json").write_text(
        '{"run_id":"run-2","verdict":"FAIL","feedback_history":[{"feedback":{"metric_status":{"ncc":false}}}]}',
        encoding="utf-8",
    )
    (run / "trajectory.jsonl").write_text(
        '{"kind":"assistant_text","text":"full optimisation timed out"}\n',
        encoding="utf-8",
    )
    workspace = repo / "artifacts" / "workspaces" / "Model" / "train_fail" / "run-2"
    workspace.mkdir(parents=True)
    (workspace / "plan.md").write_text("Tried an expensive full solver.\n", encoding="utf-8")

    universe = DistillUniverse(
        repo_root=repo,
        train_task_ids=("train_pass", "train_fail"),
        valid_task_ids=("valid_task",),
        model_slug="Model",
    )
    audit = repo / "audit.jsonl"
    universe.bind_audit_log(audit)
    episode = TaskEpisode(
        task_id="train_pass",
        run_id="run-1",
        family="wave_physics",
        final_verdict="PASS",
        rounds_used=1,
        runtime_seconds=1.0,
        tool_uses=(),
        failure_signals=(),
        primary_output_rel="output/reconstruction.npz",
    )

    gaps = collect_train_gap_evidence((episode,), universe)

    assert len(gaps) == 1
    assert gaps[0]["failure_mode"] == "FAIL"
    assert gaps[0]["source_hint"]
    assert any(s["rel_path"] == "src/solvers.py" for s in gaps[0]["source_hint"])
    assert "valid_task" not in audit.read_text(encoding="utf-8")


def test_skill_pack_writes_helper_scripts_and_gap_lessons(tmp_path: Path) -> None:
    repo = tmp_path / "MyEvoSkill"
    (tmp_path / "tasks" / "train_pass").mkdir(parents=True)
    repo.mkdir()
    universe = DistillUniverse(
        repo_root=repo,
        train_task_ids=("train_pass", "train_fail"),
        valid_task_ids=("valid_task",),
        model_slug="Model",
    )
    episode = TaskEpisode(
        task_id="train_pass",
        run_id="run-1",
        family="wave_physics",
        final_verdict="PASS",
        rounds_used=1,
        runtime_seconds=1.0,
        tool_uses=(),
        failure_signals=(),
        primary_output_rel="output/reconstruction.npz",
    )
    spec = synthesize_skill(
        skill_id="wave_optics_recon_v1",
        episodes=(episode,),
        universe=universe,
        gap_evidence=(
            {
                "task_id": "train_fail",
                "failure_mode": "TIMEOUT",
                "source_hint": [],
                "agent_attempt": {"plan": "full solver", "trajectory": "timeout"},
                "transferable_lesson": "Prefer timing probes.",
            },
        ),
    )
    pack_dir = write_skill_pack(
        spec,
        tmp_path / "skills",
        valid_task_ids=universe.valid_task_ids,
        train_task_ids=universe.train_task_ids,
    )

    text = (pack_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "Train-gap lessons" in text
    assert (pack_dir / "scripts" / "inspect_npz.py").exists()
    assert (pack_dir / "scripts" / "npz_array_baseline.py").exists()
    assert (pack_dir / "scripts" / "stolt_mapping_checks.py").exists()
    assert (pack_dir / "scripts" / "confocal_fk_migration.py").exists()
    assert (pack_dir / "scripts" / "ssnp_grid_checks.py").exists()
    assert (pack_dir / "scripts" / "ssnp_idt_reconstruct.py").exists()
    assert (pack_dir / "scripts" / "wave_solver_checks.py").exists()
    assert (pack_dir / "scripts" / "fwi_eager_checks.py").exists()
    assert (pack_dir / "scripts" / "fwi_epoch_budget.py").exists()
    assert (pack_dir / "scripts" / "fwi_main_scan.py").exists()
    assert (pack_dir / "scripts" / "fwi_cpml_reconstruct.py").exists()


def test_sanitizer_rejects_failed_train_task_id_from_gap_polish(tmp_path: Path) -> None:
    repo = tmp_path / "MyEvoSkill"
    (tmp_path / "tasks" / "train_pass").mkdir(parents=True)
    repo.mkdir()
    universe = DistillUniverse(
        repo_root=repo,
        train_task_ids=("train_pass", "train_fail"),
        valid_task_ids=("valid_task",),
        model_slug="Model",
    )
    episode = TaskEpisode(
        task_id="train_pass",
        run_id="run-1",
        family="wave_physics",
        final_verdict="PASS",
        rounds_used=1,
        runtime_seconds=1.0,
        tool_uses=(),
        failure_signals=(),
        primary_output_rel="output/reconstruction.npz",
    )

    def polish(_body: str, _evidence: object) -> str:
        return "## Bad draft\nUse train_fail as a special case.\n"

    spec = synthesize_skill(
        skill_id="wave_optics_recon_v1",
        episodes=(episode,),
        universe=universe,
        llm_polish=polish,
        gap_evidence=(
            {
                "task_id": "train_fail",
                "failure_mode": "TIMEOUT",
                "source_hint": [],
                "agent_attempt": {"plan": "timeout", "trajectory": "timeout"},
                "transferable_lesson": "Prefer generic timing probes.",
            },
        ),
    )

    try:
        write_skill_pack(
            spec,
            tmp_path / "skills",
            valid_task_ids=universe.valid_task_ids,
            train_task_ids=universe.train_task_ids,
        )
    except SanitizationError as exc:
        assert "train_task_literal:train_fail" in exc.report.reason_summary()
    else:
        raise AssertionError("expected sanitizer to reject failed train task id")
