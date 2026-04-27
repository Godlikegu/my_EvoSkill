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
