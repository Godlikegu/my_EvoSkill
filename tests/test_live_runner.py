from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from myevoskill.judge_runner import invoke_judge_runner
from myevoskill.live_runner import (
    evaluate_manifest_run,
    load_registered_manifest,
    resolve_registered_task_root,
    run_registered_task_live,
)
from myevoskill.registration_contract import ensure_live_ready_manifest
from myevoskill.models import JudgeResult, ProxyFeedback, RunRecord


def test_load_registered_manifest_reads_registry_task(tmp_path):
    project_root = tmp_path / "project"
    registry_root = project_root / "registry" / "tasks"
    registry_root.mkdir(parents=True, exist_ok=True)
    manifest_path = registry_root / "demo_task.json"
    manifest_path.write_text(
        json.dumps({"task_id": "demo_task", "source_task_dir": "../tasks/demo_task"}),
        encoding="utf-8",
    )

    manifest = load_registered_manifest("demo_task", project_root=project_root)

    assert manifest["task_id"] == "demo_task"


def test_invoke_judge_runner_uses_utf8_stdio_for_payloads(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["encoding"] = kwargs.get("encoding")
        captured["text"] = kwargs.get("text")
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(command, 0, '{"ok": true}', "")

    monkeypatch.setattr("myevoskill.judge_runner.subprocess.run", fake_run)

    payload = invoke_judge_runner(
        Path(sys.executable),
        mode="evaluate",
        payload={"message": "möbius"},
    )

    assert payload["ok"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["text"] is True
    assert "möbius" in str(captured["input"])


def test_resolve_registered_task_root_uses_project_relative_source_dir(tmp_path):
    project_root = tmp_path / "project"
    task_root = tmp_path / "tasks" / "demo_task"
    task_root.mkdir(parents=True, exist_ok=True)

    resolved = resolve_registered_task_root(
        {"task_id": "demo_task", "source_task_dir": "../tasks/demo_task"},
        project_root=project_root,
    )

    assert resolved == task_root.resolve()


def test_evaluate_manifest_run_loads_task_local_judge_adapter(tmp_path):
    task_root = tmp_path / "task_root"
    adapter_dir = task_root / "evaluation"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "judge_adapter.py").write_text(
        "\n".join(
            [
                "from myevoskill.models import JudgeResult",
                "",
                "def evaluate_run(task_root, run_record, manifest):",
                "    return JudgeResult(",
                "        task_id=run_record.task_id,",
                "        all_metrics_passed=True,",
                "        metrics_actual={'demo_metric': 1.0},",
                "        failed_metrics=[],",
                "        failure_tags=[],",
                "    )",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": "demo_task",
        "judge_spec": {
            "adapter_path": "evaluation/judge_adapter.py",
            "callable": "evaluate_run",
        },
        "runtime_env": {
            "python_executable": str(Path(sys.executable).resolve()),
            "ready": True,
            "env_hash": "env-1",
        },
    }
    run_record = RunRecord(
        run_id="run-1",
        task_id="demo_task",
        provider="local",
        env_hash="env-1",
        skills_active=(),
        workspace_root=tmp_path / "workspace",
    )

    result = evaluate_manifest_run(task_root, run_record, manifest)

    assert isinstance(result, JudgeResult)
    assert result.all_metrics_passed is True
    assert result.metrics_actual["demo_metric"] == 1.0


def test_live_runner_gate_rejects_manifest_without_ready_judge(tmp_path):
    task_root = tmp_path / "task_root"
    (task_root / "evaluation").mkdir(parents=True, exist_ok=True)
    contract_path = task_root / "evaluation" / "registration_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "task_id": "demo_task",
                "family": "optics",
                "resources": [
                    {
                        "path": "README.md",
                        "role": "task_description",
                        "visibility": "public",
                        "semantics": "Task description.",
                        "authority": "authoritative",
                    }
                ],
                "output_contract": {
                    "path": "output/reconstruction.npz",
                    "format": "npz",
                    "required_fields": ["signal"],
                    "numeric_fields": ["signal"],
                    "same_shape_fields": ["signal"],
                },
                "judge_contract": {
                    "metrics": [
                        {
                            "name": "score",
                            "kind": "standard",
                            "description": "Demo score.",
                            "mode": "ncc",
                            "output_field": "signal",
                            "reference_resource_path": "data/raw_data.npz",
                            "reference_field": "signal",
                            "pass_condition": {"operator": ">=", "threshold": 0.9},
                        }
                    ]
                },
                "execution_conventions": {
                    "read_first": ["README_public.md"],
                    "readable_paths": ["README_public.md"],
                    "writable_paths": ["work", "output", "checkpoints"],
                    "entrypoint": "work/main.py",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = {
        "task_id": "demo_task",
        "judge_spec": {
            "ready": False,
            "registration_contract_path": "evaluation/registration_contract.json",
        },
    }

    with pytest.raises(RuntimeError, match="judge_spec.ready is false"):
        ensure_live_ready_manifest(manifest, task_root=task_root)


def test_run_registered_task_live_uses_sdk_result_completion_and_30m_defaults(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    task_root = tmp_path / "tasks" / "demo_task"
    task_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task_id": "demo_task",
        "family": "optics",
        "public_policy": {},
        "source_task_dir": "../tasks/demo_task",
    }
    bundle = SimpleNamespace(root_dir=project_root / "artifacts" / "compiled" / "demo_task")
    captured: dict[str, object] = {}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sdk-secret")
    monkeypatch.setattr(
        "myevoskill.live_runner.load_registered_manifest",
        lambda task_id, *, project_root: manifest,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.resolve_registered_task_root",
        lambda manifest, *, project_root: task_root,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ensure_live_ready_manifest",
        lambda manifest, *, task_root: None,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ensure_manifest_runtime_env",
        lambda manifest, *, task_root, output_root: {
            "backend": "venv_pip",
            "env_hash": "env-1",
            "requirements_path": str(task_root / "requirements.txt"),
            "python_executable": str(Path(sys.executable).resolve()),
            "ready": True,
            "build_log_path": str(project_root / "artifacts" / "env_cache" / "build.log"),
            "install_report_path": str(
                project_root / "artifacts" / "env_cache" / "install_report.json"
            ),
        },
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.TaskBundleCompiler.compile",
        lambda self, *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ClaudeSDKAdapter.resolve_model_name",
        lambda self: "",
    )

    class FakeWorkspaceAdapter:
        def run(self, bundle_value, session_config, skills):
            captured["session_config"] = session_config
            return RunRecord(
                run_id=session_config.run_id,
                task_id="demo_task",
                provider="claude_workspace",
                env_hash=session_config.env_hash,
                skills_active=skills,
                workspace_root=session_config.workspace_root,
                model_provider=session_config.model_config.provider_name,
                model_name=session_config.model_config.model_name,
                metadata={
                    "protocol_status": "completed",
                    "sdk_completion_source": "result_message",
                    "num_turns": 3,
                    "message_count": 12,
                },
            )

    monkeypatch.setattr("myevoskill.live_runner.ClaudeWorkspaceAdapter", FakeWorkspaceAdapter)
    monkeypatch.setattr(
        "myevoskill.live_runner.manifest_proxy_spec",
        lambda record, manifest: {},
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ProxyVerifier.evaluate",
        lambda self, record, spec: ProxyFeedback(task_id="demo_task", output_exists=True),
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.evaluate_manifest_run",
        lambda task_root, record, manifest: JudgeResult(
            task_id="demo_task",
            all_metrics_passed=False,
            metrics_actual={"score": 0.1},
            failed_metrics=["score"],
            failure_tags=[],
        ),
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.write_live_run_logs",
        lambda log_dir, **kwargs: Path(log_dir),
    )

    run_registered_task_live("demo_task", project_root=project_root)

    session_config = captured["session_config"]
    assert session_config.budget_seconds == 1800
    assert session_config.model_config.timeout == 1800
    assert session_config.provider_extras["workspace_completion_policy"] == "sdk_result_message"
    assert session_config.provider_extras["max_workspace_iterations"] == 0
    assert session_config.provider_extras["claude_max_turns"] == 0
    assert session_config.tool_policy["network_access"] is False


def test_run_registered_task_live_passes_allow_network_into_tool_policy(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    task_root = tmp_path / "tasks" / "demo_task"
    task_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task_id": "demo_task",
        "family": "optics",
        "public_policy": {},
        "source_task_dir": "../tasks/demo_task",
    }
    bundle = SimpleNamespace(root_dir=project_root / "artifacts" / "compiled" / "demo_task")
    captured: dict[str, object] = {}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sdk-secret")
    monkeypatch.setattr(
        "myevoskill.live_runner.load_registered_manifest",
        lambda task_id, *, project_root: manifest,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.resolve_registered_task_root",
        lambda manifest, *, project_root: task_root,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ensure_live_ready_manifest",
        lambda manifest, *, task_root: None,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ensure_manifest_runtime_env",
        lambda manifest, *, task_root, output_root: {
            "backend": "venv_pip",
            "env_hash": "env-1",
            "requirements_path": str(task_root / "requirements.txt"),
            "python_executable": str(Path(sys.executable).resolve()),
            "ready": True,
            "build_log_path": str(project_root / "artifacts" / "env_cache" / "build.log"),
            "install_report_path": str(
                project_root / "artifacts" / "env_cache" / "install_report.json"
            ),
        },
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.TaskBundleCompiler.compile",
        lambda self, *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ClaudeSDKAdapter.resolve_model_name",
        lambda self: "",
    )

    class FakeWorkspaceAdapter:
        def run(self, bundle_value, session_config, skills):
            captured["session_config"] = session_config
            return RunRecord(
                run_id=session_config.run_id,
                task_id="demo_task",
                provider="claude_workspace",
                env_hash=session_config.env_hash,
                skills_active=skills,
                workspace_root=session_config.workspace_root,
                model_provider=session_config.model_config.provider_name,
                model_name=session_config.model_config.model_name,
                metadata={"protocol_status": "completed"},
            )

    monkeypatch.setattr("myevoskill.live_runner.ClaudeWorkspaceAdapter", FakeWorkspaceAdapter)
    monkeypatch.setattr(
        "myevoskill.live_runner.manifest_proxy_spec",
        lambda record, manifest: {},
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ProxyVerifier.evaluate",
        lambda self, record, spec: ProxyFeedback(task_id="demo_task", output_exists=True),
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.evaluate_manifest_run",
        lambda task_root, record, manifest: JudgeResult(
            task_id="demo_task",
            all_metrics_passed=True,
            metrics_actual={},
            failed_metrics=[],
            failure_tags=[],
        ),
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.write_live_run_logs",
        lambda log_dir, **kwargs: Path(log_dir),
    )

    run_registered_task_live("demo_task", project_root=project_root, allow_network=True)

    session_config = captured["session_config"]
    assert session_config.tool_policy["network_access"] is True


def test_run_registered_task_live_uses_inspect_executor_when_requested(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    task_root = tmp_path / "tasks" / "demo_task"
    task_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task_id": "demo_task",
        "family": "optics",
        "public_policy": {},
        "source_task_dir": "../tasks/demo_task",
    }
    bundle = SimpleNamespace(root_dir=project_root / "artifacts" / "compiled" / "demo_task")
    captured: dict[str, object] = {}

    monkeypatch.setenv("OPENAI_API_KEY", "inspect-secret")
    monkeypatch.setattr(
        "myevoskill.live_runner.load_registered_manifest",
        lambda task_id, *, project_root: manifest,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.resolve_registered_task_root",
        lambda manifest, *, project_root: task_root,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ensure_live_ready_manifest",
        lambda manifest, *, task_root: None,
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ensure_manifest_runtime_env",
        lambda manifest, *, task_root, output_root: {
            "backend": "venv_pip",
            "env_hash": "env-1",
            "requirements_path": str(task_root / "requirements.txt"),
            "python_executable": str(Path(sys.executable).resolve()),
            "ready": True,
            "build_log_path": str(project_root / "artifacts" / "env_cache" / "build.log"),
            "install_report_path": str(
                project_root / "artifacts" / "env_cache" / "install_report.json"
            ),
        },
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.TaskBundleCompiler.compile",
        lambda self, *args, **kwargs: bundle,
    )

    class FakeInspectAdapter:
        def run(self, bundle_value, session_config, skills):
            captured["session_config"] = session_config
            return RunRecord(
                run_id=session_config.run_id,
                task_id="demo_task",
                provider="inspect_bridge",
                env_hash=session_config.env_hash,
                skills_active=skills,
                workspace_root=session_config.workspace_root,
                model_provider=session_config.model_config.provider_name,
                model_name=session_config.model_config.model_name,
                metadata={"protocol_status": "completed"},
            )

    monkeypatch.setattr("myevoskill.live_runner.InspectBridgeAdapter", FakeInspectAdapter)
    monkeypatch.setattr(
        "myevoskill.live_runner.manifest_proxy_spec",
        lambda record, manifest: {},
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.ProxyVerifier.evaluate",
        lambda self, record, spec: ProxyFeedback(task_id="demo_task", output_exists=True),
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.evaluate_manifest_run",
        lambda task_root, record, manifest: JudgeResult(
            task_id="demo_task",
            all_metrics_passed=True,
            metrics_actual={},
            failed_metrics=[],
            failure_tags=[],
        ),
    )
    monkeypatch.setattr(
        "myevoskill.live_runner.write_live_run_logs",
        lambda log_dir, **kwargs: Path(log_dir),
    )

    result = run_registered_task_live(
        "demo_task",
        project_root=project_root,
        executor_name="inspect",
        model_provider="openai-compatible",
        model_base_url="https://example.invalid/v1",
        model_api_key_env="OPENAI_API_KEY",
        model_name="gpt-test",
        allow_network=True,
    )

    session_config = captured["session_config"]
    assert result["executor_name"] == "inspect"
    assert session_config.model_config.provider_name == "openai-compatible"
    assert session_config.model_config.base_url == "https://example.invalid/v1"
    assert session_config.model_config.api_key_env == "OPENAI_API_KEY"
    assert session_config.model_config.model_name == "gpt-test"
    assert session_config.tool_policy["network_access"] is True
