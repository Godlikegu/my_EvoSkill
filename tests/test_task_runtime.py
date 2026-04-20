from pathlib import Path

from myevoskill.models import ExecutorSessionConfig, ModelConfig
from myevoskill.task_runtime import (
    DEFAULT_RUNTIME_POLICY,
    ensure_clean_run_directory,
    primary_output_relative_path,
    resolve_run_paths,
    resolve_runtime_policy,
)


def test_resolve_run_paths_uses_project_artifacts_layout(tmp_path):
    paths = resolve_run_paths(tmp_path, "cars_spectroscopy", "run-1")
    assert paths.workspace_root == tmp_path / "artifacts" / "workspaces" / "cars_spectroscopy" / "run-1"
    assert paths.log_root == tmp_path / "artifacts" / "logs" / "cars_spectroscopy" / "run-1"


def test_ensure_clean_run_directory_only_resets_target_run(tmp_path):
    sibling = tmp_path / "task-a" / "run-a"
    target = tmp_path / "task-a" / "run-b"
    sibling.mkdir(parents=True)
    target.mkdir(parents=True)
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")
    (target / "stale.txt").write_text("stale", encoding="utf-8")

    ensure_clean_run_directory(target)

    assert (sibling / "keep.txt").exists()
    assert target.exists()
    assert list(target.iterdir()) == []


def test_resolve_runtime_policy_prefers_session_over_manifest_and_defaults():
    policy = resolve_runtime_policy(
        task_spec={
            "runtime_policy": {
                "model_timeout_seconds": 120,
                "execution_budget_seconds": 300,
            }
        },
        session_config=ExecutorSessionConfig(
            run_id="run-1",
            env_hash="env-1",
            workspace_root=Path("/tmp/run-1"),
            budget_seconds=450,
        ),
        model_config=ModelConfig(
            provider_name="openai-compatible",
            model_name="m1",
            timeout=180,
        ),
    )
    assert policy.model_timeout_seconds == 180
    assert policy.execution_budget_seconds == 450


def test_resolve_runtime_policy_uses_global_defaults_when_missing():
    policy = resolve_runtime_policy(
        task_spec={},
        session_config=ExecutorSessionConfig(
            run_id="run-1",
            env_hash="env-1",
            workspace_root=Path("/tmp/run-1"),
        ),
        model_config=ModelConfig(
            provider_name="openai-compatible",
            model_name="m1",
        ),
    )
    assert policy.model_timeout_seconds == DEFAULT_RUNTIME_POLICY["model_timeout_seconds"]
    assert (
        policy.execution_budget_seconds
        == DEFAULT_RUNTIME_POLICY["execution_budget_seconds"]
    )


def test_primary_output_relative_path_prefers_primary_output_path():
    assert (
        primary_output_relative_path({"primary_output_path": "output/custom_result.npz"})
        == "output/custom_result.npz"
    )
