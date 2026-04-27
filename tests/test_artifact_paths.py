from pathlib import Path

from myevoskill.artifact_paths import (
    DEFAULT_MODEL_SLUG,
    model_artifact_root,
    model_slug,
    resolve_workspace_output_path,
)


def test_model_slug_sanitizes_common_model_ids():
    assert model_slug("Vendor2/Gemini-3.1-pro") == "Vendor2_Gemini-3.1-pro"
    assert model_slug("  open ai / model @ preview  ") == "open_ai_model_preview"
    assert model_slug("") == DEFAULT_MODEL_SLUG
    assert model_slug("///") == DEFAULT_MODEL_SLUG


def test_model_slug_limits_length():
    assert len(model_slug("x" * 200)) == 120


def test_model_artifact_root_uses_model_task_run_layout(tmp_path):
    assert model_artifact_root(tmp_path, "logs", "Vendor2/GPT", "task-a", "run-1") == (
        tmp_path / "artifacts" / "logs" / "Vendor2_GPT" / "task-a" / "run-1"
    )


def test_resolve_workspace_output_path_prefers_explicit_model(tmp_path):
    new_path = (
        tmp_path
        / "artifacts"
        / "workspaces"
        / "model-a"
        / "task"
        / "run"
        / "output"
        / "x.npz"
    )
    new_path.parent.mkdir(parents=True)
    new_path.write_text("new", encoding="utf-8")

    assert (
        resolve_workspace_output_path(
            repo_root=tmp_path,
            task_id="task",
            run_id="run",
            filename="x.npz",
            model_slug_value="model-a",
        )
        == new_path
    )


def test_resolve_workspace_output_path_falls_back_to_legacy(tmp_path):
    legacy = tmp_path / "artifacts" / "workspaces" / "task" / "run" / "output"
    legacy.mkdir(parents=True)

    assert (
        resolve_workspace_output_path(repo_root=tmp_path, task_id="task", run_id="run")
        == legacy
    )


def test_resolve_workspace_output_path_searches_model_buckets(tmp_path):
    new_path = (
        tmp_path
        / "artifacts"
        / "workspaces"
        / "model-a"
        / "task"
        / "run"
        / "output"
    )
    new_path.mkdir(parents=True)

    assert (
        resolve_workspace_output_path(repo_root=tmp_path, task_id="task", run_id="run")
        == new_path
    )
