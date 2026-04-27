from pathlib import Path
from types import SimpleNamespace

import myevoskill.cli as cli


def _base_args(repo_root: Path, **overrides):
    data = {
        "repo_root": str(repo_root),
        "task_id": "demo_task",
        "max_rounds": 1,
        "budget_seconds": 60,
        "max_turns_per_round": 2,
        "model": None,
        "model_id": None,
        "llm_config": None,
        "judge_python": None,
        "show_metric_status": False,
        "keep_workspace": True,
        "sandbox_root": None,
        "keep_sandbox": False,
        "json": True,
        "record_thinking": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _write_manifest(repo_root: Path):
    task_dir = repo_root / "registry" / "tasks"
    task_dir.mkdir(parents=True)
    (task_dir / "demo_task.json").write_text('{"task_id":"demo_task"}', encoding="utf-8")


def test_run_task_plain_model_keeps_existing_claude_model_behavior(tmp_path, monkeypatch):
    _write_manifest(tmp_path)
    captured = {}

    def fake_run_task_once(config):
        captured["config"] = config
        return SimpleNamespace(
            task_id="demo_task",
            run_id="run-1",
            verdict="PASS",
            rounds_used=1,
            runtime_seconds=0.1,
            summary_path=tmp_path / "summary.json",
            trajectory_path=tmp_path / "trajectory.jsonl",
            log_root=tmp_path / "logs",
            workspace_root=tmp_path / "workspace",
            error=None,
        )

    monkeypatch.setattr(cli, "run_task_once", fake_run_task_once)

    rc = cli.cmd_run_task(_base_args(tmp_path, model="existing-claude-name"))

    assert rc == 0
    assert captured["config"].model == "existing-claude-name"
    assert captured["config"].model_provider_env == {}


def test_run_task_model_id_openai_only_fails_before_harness(tmp_path, monkeypatch):
    _write_manifest(tmp_path)
    llm_config = tmp_path / "llm.yaml"
    llm_config.write_text(
        """
models:
  openai-only:
    api_type: openai
    base_url: https://openai.example/v1
    api_key: test-local-secret
    model_name: openai-model
""",
        encoding="utf-8",
    )

    def fail_if_called(_config):
        raise AssertionError("harness should not run for openai-only model_id")

    monkeypatch.setattr(cli, "run_task_once", fail_if_called)

    rc = cli.cmd_run_task(
        _base_args(tmp_path, model_id="openai-only", llm_config=str(llm_config))
    )

    assert rc == 2
