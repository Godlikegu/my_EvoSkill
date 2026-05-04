from pathlib import Path

from myevoskill.workspace.builder import build_workspace


def _minimal_task(tmp_path: Path) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    task = tmp_path / "tasks" / "demo"
    repo.mkdir()
    task.mkdir(parents=True)
    (task / "README.md").write_text("# Demo\n", encoding="utf-8")
    manifest = {
        "task_id": "demo",
        "source_task_dir": str(task),
        "primary_output_path": "output/result.npz",
    }
    return repo, manifest


def test_build_workspace_uses_model_parent_when_provided(tmp_path):
    repo, manifest = _minimal_task(tmp_path)
    parent = repo / "artifacts" / "workspaces" / "model-a" / "demo"

    build = build_workspace(
        repo_root=repo,
        manifest=manifest,
        run_id="run-1",
        workspace_parent=parent,
    )

    assert build.agent_root == parent / "run-1"
    assert (build.agent_root / "README.md").exists()


def test_build_workspace_keeps_legacy_layout_without_model_parent(tmp_path):
    repo, manifest = _minimal_task(tmp_path)

    build = build_workspace(repo_root=repo, manifest=manifest, run_id="run-1")

    assert build.agent_root == repo / "artifacts" / "workspaces" / "demo" / "run-1"


def test_build_workspace_injects_single_skill_pack_under_named_skill_dir(tmp_path):
    repo, manifest = _minimal_task(tmp_path)
    skill_pack = tmp_path / "skills" / "wave-optics-recon-v1"
    skill_pack.mkdir(parents=True)
    (skill_pack / "scripts").mkdir()
    (skill_pack / "SKILL.md").write_text(
        "---\nname: wave-optics-recon-v1\ndescription: test\n---\n# Skill\n",
        encoding="utf-8",
    )
    (skill_pack / "scripts" / "helper.py").write_text("print('ok')\n", encoding="utf-8")

    build = build_workspace(
        repo_root=repo,
        manifest=manifest,
        run_id="run-1",
        skill_pack_dir=skill_pack,
    )

    assert (
        build.agent_root
        / ".claude"
        / "skills"
        / "wave-optics-recon-v1"
        / "SKILL.md"
    ).exists()
    assert (
        build.agent_root
        / ".claude"
        / "skills"
        / "wave-optics-recon-v1"
        / "scripts"
        / "helper.py"
    ).exists()
    assert not (build.agent_root / ".claude" / "skills" / "SKILL.md").exists()
