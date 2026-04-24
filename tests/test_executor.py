import asyncio
import json
import os
import subprocess
import sys
import urllib.error
import uuid

import numpy as np
import pytest

from myevoskill.executor import (
    ClaudeAdapter,
    ClaudeSDKExecutionError,
    ClaudeWorkspaceAdapter,
    InspectBridgeAdapter,
    LocalRunnerAdapter,
    OpenHandsAdapter,
    _prepare_runtime_workspace,
    _run_subprocess,
    _runtime_environment,
)
from myevoskill.models import ExecutorSessionConfig, ModelConfig, TaskBundle


def make_bundle(tmp_path, task_spec=None):
    bundle_root = tmp_path / "bundle"
    public_root = bundle_root / "public_bundle"
    hidden_root = bundle_root / "hidden_bundle"
    public_root.mkdir(parents=True)
    hidden_root.mkdir(parents=True)
    readme = public_root / "README_public.md"
    readme.write_text("public instructions", encoding="utf-8")
    task_spec_path = bundle_root / "task.yaml"
    compile_report_path = bundle_root / "compile_report.json"
    spec = {
        "task_id": "task-1",
        "family": "optics",
        "runtime_layout": {
            "data_dir": "data",
            "work_dir": "work",
            "output_dir": "output",
            "checkpoints_dir": "checkpoints",
            "public_bundle_dir": "public_bundle",
        },
        "output_contract": {
            "required_outputs": [{"path": "output/reconstruction.npz", "format": "npz"}]
        },
        "runtime_policy": {
            "model_timeout_seconds": 120,
            "execution_budget_seconds": 120,
        },
        "proxy_spec": {"primary_output": "output/reconstruction.npz", "output_dtype": "npz"},
        "judge_spec": {},
    }
    spec.update(task_spec or {})
    task_spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    compile_report_path.write_text(
        json.dumps(
            {
                "final_public_contract": spec.get("output_contract", {}),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TaskBundle(
        task_id="task-1",
        family="optics",
        root_dir=bundle_root,
        public_bundle_dir=public_root,
        hidden_bundle_dir=hidden_root,
        task_spec_path=task_spec_path,
        compile_report_path=compile_report_path,
        readme_public_path=readme,
    )


def test_local_runner_adapter_executes_real_command(tmp_path):
    bundle = make_bundle(tmp_path)
    script = bundle.public_bundle_dir / "runner.py"
    script.write_text("print('hello from local runner')\n", encoding="utf-8")
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        command=("python3", "runner.py"),
    )
    record = LocalRunnerAdapter().run(bundle, config, ["skill-a"])
    assert record.provider == "local_runner"
    assert "hello from local runner" in record.stdout
    assert record.skills_active == ["skill-a"]


def test_runtime_environment_forces_utf8_encoding(tmp_path):
    bundle = make_bundle(tmp_path)
    session = ExecutorSessionConfig(run_id="run-1", env_hash="env-1")
    _, runtime_paths = _prepare_runtime_workspace(bundle, tmp_path / "workspace")
    env = _runtime_environment(session, bundle, runtime_paths)
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"


def test_runtime_environment_uses_task_runtime_python_and_path(tmp_path):
    fake_env_root = tmp_path / "task_env" / "venv"
    task_python = fake_env_root / "Scripts" / "python.exe"
    for relative in (
        "Scripts/python.exe",
        "Library/bin/python.dll",
        "Library/usr/bin/libopenblas.dll",
        "DLLs/_ctypes.pyd",
    ):
        path = fake_env_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    bundle = make_bundle(
        tmp_path,
        task_spec={
            "runtime_env": {
                "backend": "venv_pip",
                "env_hash": "env-task",
                "python_executable": str(task_python.resolve()),
                "ready": True,
            }
        },
    )
    session = ExecutorSessionConfig(run_id="run-1", env_hash="env-session")
    _, runtime_paths = _prepare_runtime_workspace(bundle, tmp_path / "workspace")

    env = _runtime_environment(session, bundle, runtime_paths)

    path_entries = env["PATH"].split(os.pathsep)
    assert env["MYEVOSKILL_TASK_PYTHON"] == str(task_python.resolve())
    assert env["MYEVOSKILL_TASK_ENV_HASH"] == "env-task"
    assert env["MYEVOSKILL_TASK_ENV_BACKEND"] == "venv_pip"
    assert env["VIRTUAL_ENV"] == str(fake_env_root.resolve())
    assert path_entries[:5] == [
        str((fake_env_root / "Scripts").resolve()),
        str(fake_env_root.resolve()),
        str((fake_env_root / "Library" / "bin").resolve()),
        str((fake_env_root / "Library" / "usr" / "bin").resolve()),
        str((fake_env_root / "DLLs").resolve()),
    ]


def test_run_subprocess_uses_task_python_launcher(tmp_path):
    env = {
        **os.environ,
        "MYEVOSKILL_TASK_PYTHON": str(os.path.abspath(sys.executable)),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }

    completed, timed_out = _run_subprocess(
        ("python", "-c", "import sys; print(sys.executable)"),
        cwd=tmp_path,
        env=env,
        timeout_seconds=30,
    )

    assert timed_out is False
    assert completed.returncode == 0
    assert os.path.normcase(completed.args[0]) == os.path.normcase(os.path.abspath(sys.executable))
    assert os.path.normcase(completed.stdout.strip()) == os.path.normcase(
        os.path.abspath(sys.executable)
    )


def test_inspect_bridge_adapter_falls_back_when_dependency_forced_missing(tmp_path):
    bundle = make_bundle(tmp_path)
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        provider_extras={"allow_fallback": True},
    )
    record = InspectBridgeAdapter(inspect_available=False).run(bundle, config, [])
    assert record.provider == "inspect_unavailable_fallback"
    assert record.metadata["fallback_reason"] == "inspect_ai_missing"
    assert record.metadata["requested_provider"] == "inspect_bridge"


def test_inspect_bridge_adapter_raises_when_dependency_forced_missing_and_fallback_disabled(
    tmp_path,
):
    bundle = make_bundle(tmp_path)
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        provider_extras={"allow_fallback": False},
    )
    with pytest.raises(RuntimeError, match="inspect_ai is unavailable"):
        InspectBridgeAdapter(inspect_available=False).run(bundle, config, [])


def test_inspect_bridge_adapter_uses_bridge_provider_when_dependency_available(tmp_path):
    bundle = make_bundle(tmp_path)
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        provider_extras={"allow_fallback": True},
    )
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert record.provider == "inspect_bridge"
    assert record.metadata["bridge_mode"] == "placeholder"


def test_inspect_bridge_adapter_requires_env_api_key_for_model_backed_runs(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="MISSING_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": "print('hi')"},
    )
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="environment variable 'MISSING_KEY'"):
        InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])


def test_inspect_bridge_adapter_runs_mock_model_response_and_records_safe_metadata(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    generated_code = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "workspace = Path(os.environ['MYEVOSKILL_WORKSPACE'])",
            "(workspace / 'output').mkdir(parents=True, exist_ok=True)",
            "print(Path(os.environ['MYEVOSKILL_PUBLIC_BUNDLE']).name)",
        ]
    )
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": generated_code},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, ["skill-a"])
    assert record.provider == "inspect_bridge"
    assert record.model_provider == "openai-compatible"
    assert record.metadata["bridge_mode"] == "single_shot_mock_llm"
    assert record.metadata["model_provider_kind"] == "openai_compatible"
    assert record.metadata["api_key_env"] == "TEST_API_KEY"
    assert record.metadata["response_format"] == "plain_text_code"
    assert record.metadata["prompt_contract_version"] == "v2_structured_json"
    assert record.metadata["response_candidate_count"] >= 1
    assert record.metadata["response_selected_source"] == "plain_text_fallback"
    assert "public_bundle" in record.stdout
    assert "secret-value" not in record.stdout
    transcript = (tmp_path / "workspace" / "transcript.txt").read_text(encoding="utf-8")
    assert "TEST_API_KEY" in transcript
    assert "secret-value" not in transcript


def test_inspect_workspace_response_normalizes_root_level_paths_into_work():
    adapter = InspectBridgeAdapter(inspect_available=True)

    payload, metadata = adapter._parse_inspect_workspace_model_response(
        raw_content=json.dumps(
            {
                "files": {
                    "plan.md": "# Task Understanding\n",
                    "main.py": "print('hi')\n",
                    "solver/utils.py": "VALUE = 1\n",
                },
                "entrypoint": "main.py",
                "declared_outputs": ["output/reconstruction.npz"],
                "assumptions": [],
                "solver_summary": "demo",
                "implementation_notes": ["using a small helper module"],
                "validation_plan": ["run python work/main.py"],
            }
        ),
        bridge_mode="inspect_agent_mock",
        model_provider_kind="openai-compatible",
    )

    assert payload["entrypoint"] == "work/main.py"
    assert payload["files"] == {
        "output/plan.md": "# Task Understanding\n",
        "work/main.py": "print('hi')\n",
        "work/solver/utils.py": "VALUE = 1\n",
    }
    assert metadata["parsed_response"]["entrypoint"] == "work/main.py"
    assert metadata["parsed_response"]["files"][0] == "output/plan.md"


def test_inspect_workspace_response_requires_plan_and_protocol_fields():
    adapter = InspectBridgeAdapter(inspect_available=True)

    with pytest.raises(Exception, match="output/plan.md"):
        adapter._parse_inspect_workspace_model_response(
            raw_content=json.dumps(
                {
                    "files": {
                        "main.py": "print('hi')\n",
                    },
                    "entrypoint": "main.py",
                    "declared_outputs": ["output/reconstruction.npz"],
                    "assumptions": [],
                    "solver_summary": "demo",
                    "implementation_notes": ["note"],
                    "validation_plan": ["validate"],
                }
            ),
            bridge_mode="inspect_agent_mock",
            model_provider_kind="openai-compatible",
        )

    with pytest.raises(Exception, match="implementation_notes"):
        adapter._parse_inspect_workspace_model_response(
            raw_content=json.dumps(
                {
                    "files": {
                        "output/plan.md": "# plan\n",
                        "main.py": "print('hi')\n",
                    },
                    "entrypoint": "main.py",
                    "declared_outputs": ["output/reconstruction.npz"],
                    "assumptions": [],
                    "solver_summary": "demo",
                    "validation_plan": ["validate"],
                }
            ),
            bridge_mode="inspect_agent_mock",
            model_provider_kind="openai-compatible",
        )

    with pytest.raises(Exception, match="validation_plan"):
        adapter._parse_inspect_workspace_model_response(
            raw_content=json.dumps(
                {
                    "files": {
                        "output/plan.md": "# plan\n",
                        "main.py": "print('hi')\n",
                    },
                    "entrypoint": "main.py",
                    "declared_outputs": ["output/reconstruction.npz"],
                    "assumptions": [],
                    "solver_summary": "demo",
                    "implementation_notes": ["note"],
                }
            ),
            bridge_mode="inspect_agent_mock",
            model_provider_kind="openai-compatible",
        )


def test_inspect_bridge_adapter_workspace_mode_runs_self_eval_and_writes_trajectory(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    np.savez(
        bundle.public_bundle_dir / "data" / "raw_data.npz",
        measurements=np.array([[0.1, 0.2]], dtype=float),
        nu_axis=np.array([[1.0, 2.0]], dtype=float),
    )
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="gpt-test",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    response_payload = {
        "files": {
            "output/plan.md": "\n".join(
                [
                    "# Task Understanding",
                    "",
                    "# Input/Output Contract",
                    "",
                    "# Module Layout",
                    "",
                    "# Algorithm Choice",
                    "",
                    "# Implementation Steps",
                    "",
                    "# Validation Plan",
                    "",
                    "# Assumptions",
                ]
            )
            + "\n",
            "work/main.py": "\n".join(
                [
                    "from src.preprocessing import load_inputs",
                    "from src.physics_model import build_model",
                    "from src.solvers import solve",
                    "from src.visualization import summarize",
                    "from pathlib import Path",
                    "import numpy as np",
                    "Path('output').mkdir(parents=True, exist_ok=True)",
                    "raw = load_inputs()",
                    "model = build_model(raw)",
                    "signal = solve(model)",
                    "summarize(signal)",
                    "np.savez('output/reconstruction.npz', signal=signal)",
                    "print('inspect-workspace-pass')",
                ]
            )
            + "\n",
            "work/src/__init__.py": "",
            "work/src/preprocessing.py": "\n".join(
                [
                    "import numpy as np",
                    "",
                    "def load_inputs():",
                    "    raw = np.load('data/raw_data.npz')",
                    "    return raw['measurements']",
                ]
            )
            + "\n",
            "work/src/physics_model.py": "\n".join(
                [
                    "def build_model(raw):",
                    "    return raw",
                ]
            )
            + "\n",
            "work/src/solvers.py": "\n".join(
                [
                    "def solve(model):",
                    "    return model",
                ]
            )
            + "\n",
            "work/src/visualization.py": "\n".join(
                [
                    "def summarize(signal):",
                    "    return {'shape': getattr(signal, 'shape', None)}",
                ]
            )
            + "\n",
        },
        "entrypoint": "work/main.py",
        "declared_outputs": ["output/reconstruction.npz"],
        "assumptions": ["public inputs are sufficient"],
        "solver_summary": "inspect workspace test",
        "implementation_notes": [
            "Use the default scientific module split to keep preprocessing, modeling, solving, and reporting separate."
        ],
        "validation_plan": [
            "Run python work/main.py",
            "Run python evaluation/self_eval.py",
        ],
    }
    config = ExecutorSessionConfig(
        run_id="run-inspect-workspace",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_inspect_response": json.dumps(response_payload)},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")

    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, ["skill-a"])

    assert record.provider == "inspect_bridge"
    assert record.metadata["bridge_mode"] == "inspect_agent"
    assert record.metadata["sdk_backend"] == "inspect_ai_compatible"
    assert record.metadata["prompt_contract_version"] == "v9_inspect_plan_scaffold_self_eval"
    assert record.metadata["public_self_eval_seen_in_trace"] is True
    assert record.metadata["public_self_eval_passed_post_run"] is True
    assert record.metadata["output_contract_satisfied_post_run"] is True
    assert record.metadata["network_access"] is False
    assert record.metadata["plan_artifact_required"] is True
    assert record.metadata["plan_artifact_present"] is True
    assert record.metadata["plan_artifact_written_before_entrypoint"] is True
    assert record.metadata["default_scientific_layout_expected"] is True
    assert record.metadata["default_scientific_layout_present"] is True
    assert record.metadata["implementation_notes_present"] is True
    assert record.metadata["validation_plan_present"] is True
    assert record.metadata["declared_outputs_match_public_contract"] is True
    assert record.metadata["protocol_warnings"] == []
    assert record.metadata["tool_call_count"] >= 2
    assert "inspect-workspace-pass" in record.stdout
    assert (tmp_path / "workspace" / "inspect_sandbox.json").exists()
    assert (tmp_path / "workspace" / "trajectory_normalized.json").exists()
    assert (tmp_path / "workspace" / "trajectory_summary.json").exists()
    assert (tmp_path / "workspace" / "vendor_session_ref.json").exists()
    assert (tmp_path / "workspace" / "public_self_eval_round_1.json").exists()
    assert (tmp_path / "workspace" / "output" / "plan.md").exists()


def test_inspect_bridge_adapter_workspace_mode_records_network_opt_in(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="gpt-test",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    response_payload = {
        "files": {
            "output/plan.md": "\n".join(
                [
                    "# Task Understanding",
                    "",
                    "# Input/Output Contract",
                    "",
                    "# Module Layout",
                    "",
                    "Small smoke test: deviation from the default scientific layout is acceptable here.",
                    "",
                    "# Algorithm Choice",
                    "",
                    "# Implementation Steps",
                    "",
                    "# Validation Plan",
                    "",
                    "# Assumptions",
                ]
            )
            + "\n",
            "work/main.py": "\n".join(
                [
                    "from pathlib import Path",
                    "import numpy as np",
                    "Path('output').mkdir(parents=True, exist_ok=True)",
                    "np.savez('output/reconstruction.npz', signal=np.array([1.0]))",
                    "print('network-opt-in')",
                ]
            )
            + "\n"
        },
        "entrypoint": "work/main.py",
        "declared_outputs": ["output/reconstruction.npz"],
        "assumptions": [],
        "solver_summary": "network metadata test",
        "implementation_notes": [
            "This is a simple task smoke test, so the implementation stays in main.py as an explained deviation."
        ],
        "validation_plan": ["run the entrypoint and public self eval"],
    }
    config = ExecutorSessionConfig(
        run_id="run-inspect-network",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        tool_policy={"network_access": True},
        provider_extras={"mock_inspect_response": json.dumps(response_payload)},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")

    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])

    sandbox_payload = json.loads(
        (tmp_path / "workspace" / "inspect_sandbox.json").read_text(encoding="utf-8")
    )
    assert record.metadata["network_access"] is True
    assert sandbox_payload["network_access"] is True
    assert record.metadata["default_scientific_layout_present"] is False
    assert record.metadata["layout_deviation_explained"] is True
    assert "default scientific multi-file layout not fully adopted" in record.metadata["protocol_warnings"]


def test_provider_placeholders_accept_model_config(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    claude_model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    openai_model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    claude_config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=claude_model,
        provider_extras={
            "mock_claude_sdk_response": {
                "files": {
                    "work/main.py": "print('claude-workspace')\n",
                    "work/src/__init__.py": "",
                    "work/src/preprocessing.py": "def prepare():\n    return None\n",
                    "work/src/physics_model.py": "def build_model():\n    return None\n",
                    "work/src/solvers.py": "def solve():\n    return None\n",
                    "work/src/visualization.py": "def summarize():\n    return None\n",
                },
                "solver_summary": "claude sdk mock",
                "declared_outputs": [],
                "assumptions": [],
                "files_written": ["work/main.py"],
                "commands_run": ["python work/main.py"],
            }
        },
    )
    openai_config = ExecutorSessionConfig(
        run_id="run-1-openai",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace_openai",
        model_config=openai_model,
        provider_extras={
            "mock_llm_response": json.dumps(
                {
                    "files": {
                        "work/main.py": "print('claude-workspace')\n",
                        "work/src/__init__.py": "",
                        "work/src/preprocessing.py": "def prepare():\n    return None\n",
                        "work/src/physics_model.py": "def build_model():\n    return None\n",
                        "work/src/solvers.py": "def solve():\n    return None\n",
                        "work/src/visualization.py": "def summarize():\n    return None\n",
                    },
                    "entrypoint": "work/main.py",
                }
            )
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    claude = ClaudeAdapter().run(bundle, claude_config, [])
    openhands = OpenHandsAdapter().run(bundle, openai_config, [])
    assert claude.model_name == "claude-test"
    assert "claude-workspace" in claude.stdout
    assert openhands.model_provider == "openai-compatible"


def test_claude_workspace_adapter_writes_multifile_workspace_and_runs(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez(
        bundle.public_bundle_dir / "data" / "raw_data.npz",
        measurements=np.array([[0.1, 0.2]], dtype=float),
        nu_axis=np.array([[1.0, 2.0]], dtype=float),
    )
    config = ExecutorSessionConfig(
        run_id="run-workspace",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "mock_claude_sdk_response": {
                "files": {
                    "work/main.py": "\n".join(
                        [
                            "import numpy as np",
                            "from pathlib import Path",
                            "Path('output').mkdir(parents=True, exist_ok=True)",
                            "raw = np.load('data/raw_data.npz')",
                            "np.savez('output/reconstruction.npz', estimated_temperature_K=np.array([2400.0]), reconstructed_spectrum=raw['measurements'], nu_axis=raw['nu_axis'])",
                            "print('workspace-pass')",
                        ]
                    )
                    + "\n",
                    "work/src/__init__.py": "",
                    "work/src/preprocessing.py": "def prepare():\n    return 'ok'\n",
                    "work/src/physics_model.py": "def build_model():\n    return 'ok'\n",
                    "work/src/solvers.py": "def solve():\n    return 'ok'\n",
                    "work/src/visualization.py": "def summarize():\n    return 'ok'\n",
                },
                "solver_summary": "workspace multifile test",
                "declared_outputs": ["output/reconstruction.npz"],
                "assumptions": ["public inputs are staged"],
                "files_written": [
                    "work/main.py",
                    "work/src/__init__.py",
                ],
                "commands_run": ["python work/main.py", "python evaluation/self_eval.py"],
                "sdk_messages": [
                    {"name": "Bash", "input": {"command": "cd . && python work/main.py"}},
                    {"name": "Bash", "input": {"command": "cd . && python evaluation/self_eval.py"}},
                ],
            }
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    record = ClaudeWorkspaceAdapter().run(bundle, config, ["skill-a"])
    assert record.provider == "claude_workspace"
    assert record.metadata["agent_mode"] == "workspace_edit"
    assert record.metadata["prompt_contract_version"] == "v7_claude_sdk_public_self_eval"
    assert record.metadata["sdk_backend"] == "claude_sdk"
    assert record.metadata["agent_stop_policy"] == "run_self_eval_then_summary"
    assert record.metadata["stop_oracle"] == "public_self_eval"
    assert record.metadata["harness_feedback_mode"] == "none"
    assert record.metadata["run_status"] == "succeeded"
    assert record.metadata["run_failure_reason"] == ""
    assert record.metadata["output_contract_satisfied_post_run"] is True
    assert record.metadata["public_self_eval_seen_in_trace"] is True
    assert record.metadata["public_self_eval_passed_post_run"] is True
    assert "workspace-pass" in record.stdout
    assert (tmp_path / "workspace" / "work" / "main.py").exists()
    assert record.metadata["files_written"]
    assert record.metadata["public_self_check_status"]["self_check_passed"] is True
    assert (tmp_path / "workspace" / "post_run_audit.json").exists()
    assert (tmp_path / "workspace" / "public_self_eval_round_1.json").exists()
    assert (tmp_path / "workspace" / "public_self_eval_stdout_round_1.log").exists()
    assert (tmp_path / "workspace" / "public_self_eval_stderr_round_1.log").exists()
    assert (tmp_path / "workspace" / "evaluation" / "self_eval.py").exists()
    assert (tmp_path / "workspace" / "evaluation" / "self_eval_spec.json").exists()
    assert not (tmp_path / "workspace" / "check_ready_calls_round_1.json").exists()
    assert (tmp_path / "workspace" / "trajectory_native.jsonl").exists()
    assert (tmp_path / "workspace" / "trajectory_normalized.json").exists()
    assert (tmp_path / "workspace" / "trajectory_summary.json").exists()
    assert (tmp_path / "workspace" / "trajectory_redaction_report.json").exists()
    assert (tmp_path / "workspace" / "vendor_session_ref.json").exists()


def test_claude_workspace_adapter_files_written_excludes_removed_scaffold_paths(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    np.savez(
        bundle.public_bundle_dir / "data" / "raw_data.npz",
        measurements=np.array([[0.1, 0.2]], dtype=float),
        nu_axis=np.array([[1.0, 2.0]], dtype=float),
    )
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-files-written",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "mock_claude_sdk_response": {
                "files": {
                    "work/main.py": "\n".join(
                        [
                            "import numpy as np",
                            "from pathlib import Path",
                            "Path('output').mkdir(parents=True, exist_ok=True)",
                            "raw = np.load('data/raw_data.npz')",
                            "np.savez('output/reconstruction.npz', estimated_temperature_K=np.array([2400.0]), reconstructed_spectrum=raw['measurements'], nu_axis=raw['nu_axis'])",
                            "print('workspace-pass')",
                        ]
                    )
                    + "\n",
                    "work/src/__init__.py": "",
                    "work/src/forward_model.py": "def predict(measurements):\n    return measurements\n",
                    "work/src/molecular.py": "def constants():\n    return {'ok': True}\n",
                    "work/src/optimizer.py": "def solve():\n    return 2400.0\n",
                },
                "solver_summary": "workspace files written test",
                "declared_outputs": ["output/reconstruction.npz"],
                "assumptions": [],
                "files_written": ["work/main.py"],
                "commands_run": [],
            }
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    record = ClaudeWorkspaceAdapter().run(bundle, config, [])
    files_written = json.loads(
        (tmp_path / "workspace" / "files_written_round_1.json").read_text(encoding="utf-8")
    )
    assert record.metadata["run_status"] == "succeeded"
    assert sorted(files_written) == [
        "work/main.py",
        "work/src/__init__.py",
        "work/src/forward_model.py",
        "work/src/molecular.py",
        "work/src/optimizer.py",
    ]
    assert "work/src/preprocessing.py" not in files_written
    assert "work/src/physics_model.py" not in files_written
    assert "work/src/solvers.py" not in files_written
    assert "work/src/visualization.py" not in files_written


def test_claude_workspace_adapter_rejects_write_outside_workdir(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-bad-path",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "mock_claude_sdk_response": {
                "files": {"data/overwrite.py": "print('bad')\n"},
                "solver_summary": "bad write",
                "declared_outputs": [],
                "assumptions": [],
                "files_written": ["data/overwrite.py"],
                "commands_run": [],
            }
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    with pytest.raises(RuntimeError, match="modified read-only paths"):
        ClaudeWorkspaceAdapter().run(bundle, config, [])


def test_claude_workspace_adapter_does_not_reprompt_after_failed_post_run_audit(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez(
        bundle.public_bundle_dir / "data" / "raw_data.npz",
        measurements=np.array([[0.1, 0.2]], dtype=float),
        nu_axis=np.array([[1.0, 2.0]], dtype=float),
    )
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-single-query",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "mock_claude_sdk_response": [
                {
                    "files": {
                        "work/main.py": "print('missing-output')\n",
                        "work/src/__init__.py": "",
                        "work/src/preprocessing.py": "",
                        "work/src/physics_model.py": "",
                        "work/src/solvers.py": "",
                        "work/src/visualization.py": "",
                    },
                    "solver_summary": "first attempt",
                    "declared_outputs": [],
                    "assumptions": [],
                    "files_written": ["work/main.py"],
                    "commands_run": [],
                },
                {
                    "files": {
                        "work/main.py": "\n".join(
                            [
                                "import numpy as np",
                                "from pathlib import Path",
                                "raw = np.load('data/raw_data.npz')",
                                "Path('output').mkdir(parents=True, exist_ok=True)",
                                "np.savez('output/reconstruction.npz', estimated_temperature_K=np.array([2400.0]), reconstructed_spectrum=raw['measurements'], nu_axis=raw['nu_axis'])",
                                "print('repaired-output')",
                            ]
                        )
                        + "\n",
                        "work/src/__init__.py": "",
                        "work/src/preprocessing.py": "",
                        "work/src/physics_model.py": "",
                        "work/src/solvers.py": "",
                        "work/src/visualization.py": "",
                    },
                    "solver_summary": "second attempt",
                    "declared_outputs": ["output/reconstruction.npz"],
                    "assumptions": [],
                    "files_written": ["work/main.py"],
                    "commands_run": ["python work/main.py"],
                },
            ]
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    record = ClaudeWorkspaceAdapter().run(bundle, config, [])
    assert "missing-output" in record.stdout
    assert "repaired-output" not in record.stdout
    assert record.metadata["iteration_count"] == 1
    assert record.metadata["run_status"] == "failed"
    assert record.metadata["run_failure_reason"] == "missing_output_artifact"
    assert record.metadata["public_self_eval_passed_post_run"] is False
    assert record.metadata["public_self_check_status"]["self_check_passed"] is False
    assert not (tmp_path / "workspace" / "agent_summary_round_2.json").exists()


def test_claude_workspace_adapter_blocks_runtime_write_to_readonly_public_paths(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    (bundle.public_bundle_dir / "data" / "meta_data.json").write_text("{}", encoding="utf-8")
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-readonly",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "mock_claude_sdk_response": {
                "files": {
                    "work/main.py": "\n".join(
                        [
                            "from pathlib import Path",
                            "Path('data/meta_data.json').write_text('mutated', encoding='utf-8')",
                        ]
                    )
                    + "\n",
                    "work/src/__init__.py": "",
                    "work/src/preprocessing.py": "",
                    "work/src/physics_model.py": "",
                    "work/src/solvers.py": "",
                    "work/src/visualization.py": "",
                },
                "solver_summary": "bad readonly write",
                "declared_outputs": [],
                "assumptions": [],
                "files_written": ["work/main.py"],
                "commands_run": ["python work/main.py"],
            }
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    record = ClaudeWorkspaceAdapter().run(bundle, config, [])
    assert record.metadata["public_self_check_status"]["self_check_passed"] is False
    assert record.metadata["returncode"] != 0
    assert "Permission" in record.stderr or "permission" in record.stderr.lower()


def test_claude_workspace_adapter_prompt_marks_readme_as_authoritative(tmp_path):
    bundle = make_bundle(
        tmp_path,
        {
            "output_contract": {
                "required_outputs": [
                    {
                        "path": "output/reconstruction.npz",
                        "format": "npz",
                        "required_fields": [
                            "estimated_temperature_K",
                            "reconstructed_spectrum",
                            "nu_axis",
                        ],
                    }
                ]
            },
            "registration_contract": {
                "task_id": "task-1",
                "family": "optics",
                "resources": [
                    {
                        "path": "README.md",
                        "role": "task_description",
                        "visibility": "public",
                        "semantics": "Task description and public constraints for the solver.",
                        "authority": "authoritative",
                    },
                    {
                        "path": "data/meta_data.json",
                        "role": "public_metadata",
                        "visibility": "public",
                        "semantics": "Physical parameters, constants, and experiment configuration.",
                        "authority": "authoritative",
                    },
                ],
                "execution_conventions": {
                    "read_first": ["README_public.md"],
                    "readable_paths": [
                        "README_public.md",
                        "data/raw_data.npz",
                        "data/meta_data.json",
                        "requirements.txt",
                    ],
                    "writable_paths": ["work", "output", "checkpoints"],
                    "entrypoint": "work/main.py",
                },
            },
        },
    )
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    (bundle.public_bundle_dir / "data" / "meta_data.json").write_text('{"alpha": 1}', encoding="utf-8")
    (bundle.public_bundle_dir / "requirements.txt").write_text("numpy\nscipy\n", encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    (workspace_root / "evaluation").mkdir(parents=True, exist_ok=True)
    (workspace_root / "evaluation" / "self_eval.py").write_text("print('ok')\n", encoding="utf-8")
    adapter = ClaudeWorkspaceAdapter()
    prompt = adapter._build_workspace_agent_prompt(
        bundle,
        ["skill-a"],
        workspace_root=workspace_root,
        tool_policy=adapter._coerce_tool_policy(
            ExecutorSessionConfig(run_id="run-1", env_hash="env-1")
        ),
        stop_oracle="public_self_eval",
        prompt_mode="semantic_only",
        completion_policy="main_success_output_contract",
    )
    assert "README_public.md is the authoritative task specification" in prompt
    assert "workspace_root=" in prompt
    assert "cwd=" in prompt
    assert "Use relative paths first" in prompt
    assert "Your job is to complete the workspace contract and stop cleanly" in prompt
    assert "evaluation/self_eval.py" in prompt
    assert "## Contract" in prompt
    assert "## Public Files" in prompt
    assert "## IMPORTANT_RULES" in prompt
    assert "## Recommended Workflow" in prompt
    assert "output/reconstruction.npz" in prompt
    assert "estimated_temperature_K" in prompt
    assert "reconstructed_spectrum" in prompt
    assert "nu_axis" in prompt
    assert "public instructions" not in prompt
    assert '{"alpha": 1}' not in prompt
    assert "numpy\nscipy\n" not in prompt
    assert "Read first: README_public.md" in prompt
    assert "Entrypoint: python work/main.py" in prompt
    assert (
        "`README_public.md`: authoritative source: Task description and public constraints for the solver."
        in prompt
    )
    assert (
        "`data/meta_data.json`: authoritative source: Physical parameters, constants, and experiment configuration."
        in prompt
    )
    assert "`requirements.txt`: available dependency set for the workspace." in prompt
    assert "then iterate" not in prompt
    assert "STOP CONDITION" in prompt
    assert "check_ready()" not in prompt
    assert "submit_result(...)" not in prompt
    assert "If the run fails, debug locally in this same session and rerun `python work/main.py`." in prompt
    assert "immediately return the final structured summary and stop all further tool use." in prompt
    assert "metrics.json threshold" not in prompt
    assert "forcibly interrupt" not in prompt
    assert "Prefer the `Write` tool to create or modify `work/` source files" in prompt
    assert "Use Bash mainly to run programs, inspect files, debug, and perform workspace-local" in prompt
    assert "Any Bash command is acceptable if it stays inside the workspace" in prompt
    assert "network access" in prompt
    assert "Absolute or relative paths are both acceptable when they resolve inside the workspace" in prompt
    assert "preferably with the `Write` tool" in prompt
    assert "WebSearch" not in prompt
    assert "WebFetch" not in prompt
    assert "paper-first" not in prompt


def test_claude_workspace_prompt_encourages_network_research_when_enabled(tmp_path):
    bundle = make_bundle(
        tmp_path,
        {
            "registration_contract": {
                "files": [
                    {
                        "path": "README_public.md",
                        "visibility": "public",
                        "semantics": "Task description and public constraints for the solver.",
                    },
                    {
                        "path": "data/meta_data.json",
                        "visibility": "public",
                        "semantics": "Physical parameters, constants, and experiment configuration.",
                    },
                    {
                        "path": "requirements.txt",
                        "visibility": "public",
                        "semantics": "Available dependency set for the workspace.",
                    },
                ],
                "execution": {
                    "read_first": [
                        "README_public.md",
                        "data/meta_data.json",
                        "requirements.txt",
                    ],
                    "readable_paths": [
                        "README_public.md",
                        "requirements.txt",
                        "data/meta_data.json",
                    ],
                    "writable_paths": ["work/", "output/", "checkpoints/"],
                    "entrypoint": "work/main.py",
                },
            },
        },
    )
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    (bundle.public_bundle_dir / "data" / "meta_data.json").write_text('{"alpha": 1}', encoding="utf-8")
    (bundle.public_bundle_dir / "requirements.txt").write_text("numpy\nscipy\n", encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    (workspace_root / "evaluation").mkdir(parents=True, exist_ok=True)
    (workspace_root / "evaluation" / "self_eval.py").write_text("print('ok')\n", encoding="utf-8")
    adapter = ClaudeWorkspaceAdapter()
    prompt = adapter._build_workspace_agent_prompt(
        bundle,
        ["skill-a"],
        workspace_root=workspace_root,
        tool_policy=adapter._coerce_tool_policy(
            ExecutorSessionConfig(
                run_id="run-1",
                env_hash="env-1",
                tool_policy={"network_access": True},
            )
        ),
        stop_oracle="public_self_eval",
        prompt_mode="semantic_only",
        completion_policy="main_success_output_contract",
    )

    assert "WebSearch" in prompt
    assert "WebFetch" in prompt
    assert "paper-first" in prompt
    assert "do at least one brief external search" in prompt
    assert "Before writing code, use `WebSearch` and `WebFetch`" in prompt
    assert "consult official or author implementations only when the papers do not provide enough implementation detail" in prompt
    assert "Keep external research bounded" in prompt
    assert "such as network access" not in prompt


def test_claude_workspace_prompt_requires_manifest_output_contract(tmp_path):
    bundle = make_bundle(tmp_path, {"output_contract": {}})
    adapter = ClaudeWorkspaceAdapter()
    with pytest.raises(RuntimeError, match="missing manifest output_contract"):
        adapter._build_workspace_agent_prompt(
            bundle,
            [],
            workspace_root=tmp_path / "workspace",
            tool_policy=adapter._coerce_tool_policy(
                ExecutorSessionConfig(run_id="run-1", env_hash="env-1")
            ),
            stop_oracle="public_self_eval",
        )


def test_claude_workspace_adapter_consumes_result_message_and_stops():
    adapter = ClaudeWorkspaceAdapter()

    class TaskNotificationMessage:
        status = "completed"

    class FakeResultMessage:
        def __init__(
            self,
            structured_output=None,
            result="",
            *,
            is_error=False,
            subtype="success",
            num_turns=2,
            session_id="session-1",
            stop_reason="end_turn",
        ):
            self.structured_output = structured_output
            self.result = result
            self.is_error = is_error
            self.subtype = subtype
            self.num_turns = num_turns
            self.session_id = session_id
            self.stop_reason = stop_reason

    async def stream():
        yield TaskNotificationMessage()
        yield FakeResultMessage(
            structured_output={
                "solver_summary": "good",
                "declared_outputs": ["output/reconstruction.npz"],
                "assumptions": ["public inputs only"],
                "files_written": ["work/main.py"],
                "commands_run": ["python work/main.py"],
            }
        )
        await asyncio.sleep(0.2)

    result = asyncio.run(
        adapter._consume_claude_response(
            stream(),
            total_timeout_seconds=1,
            result_message_type=FakeResultMessage,
        )
    )
    assert result["summary"]["solver_summary"] == "good"
    assert result["sdk_diagnostics"]["sdk_result_seen"] is True
    assert result["sdk_diagnostics"]["sdk_result"]["session_id"] == "session-1"
    assert result["sdk_diagnostics"]["task_notification_statuses"] == ["completed"]


def test_claude_workspace_adapter_marks_protocol_incomplete_after_stop_when_no_result_message():
    adapter = ClaudeWorkspaceAdapter()

    async def stream():
        if False:
            yield None

    with pytest.raises(ClaudeSDKExecutionError, match="without a ResultMessage") as exc_info:
        asyncio.run(
            adapter._consume_claude_response(
                stream(),
                total_timeout_seconds=0.1,
                hook_events=[{"hook_event_name": "Stop"}],
            )
        )
    assert exc_info.value.error_type == "protocol_incomplete_after_stop"


def test_claude_workspace_adapter_ignores_stale_task_notification_when_later_messages_arrive():
    adapter = ClaudeWorkspaceAdapter()

    class TaskNotificationMessage:
        def __init__(self):
            self.status = "completed"

    class AssistantMessage:
        pass

    async def stream():
        yield TaskNotificationMessage()
        yield AssistantMessage()
        await asyncio.sleep(0.2)
        if False:
            yield None

    with pytest.raises(ClaudeSDKExecutionError, match="timed out after 0.05 seconds") as exc_info:
        asyncio.run(
            adapter._consume_claude_response(
                stream(),
                total_timeout_seconds=0.05,
            )
        )
    assert exc_info.value.error_type == "request_timeout"
    assert exc_info.value.diagnostics["last_message_type"] == "AssistantMessage"
    assert exc_info.value.diagnostics["last_task_notification_status"] == "completed"


def test_claude_workspace_adapter_times_out_without_result_message():
    adapter = ClaudeWorkspaceAdapter()

    async def stream():
        await asyncio.sleep(0.2)
        if False:
            yield None

    with pytest.raises(ClaudeSDKExecutionError, match="timed out after 0.05 seconds") as exc_info:
        asyncio.run(
            adapter._consume_claude_response(
                stream(),
                total_timeout_seconds=0.05,
            )
        )
    assert exc_info.value.error_type == "request_timeout"
    assert exc_info.value.diagnostics["timeout_occurred"] is True


def test_claude_workspace_adapter_builds_sdk_options_with_default_max_turns(tmp_path):
    adapter = ClaudeWorkspaceAdapter()
    options = adapter._build_claude_sdk_options_kwargs(
        session_config=ExecutorSessionConfig(run_id="run-1", env_hash="env-1"),
        workspace=tmp_path / "workspace",
        system_prompt=adapter._build_workspace_system_prompt(
            adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
        ),
        stop_oracle="public_self_eval",
    )
    assert options["max_turns"] == 50
    assert options["cwd"] == str(tmp_path / "workspace")
    assert options["add_dirs"] == [str(tmp_path / "workspace")]
    assert options["allowed_tools"] == [
        "Read",
        "Write",
        "Bash",
        "Glob",
        "Grep",
    ]
    assert options["disallowed_tools"] == ["WebFetch", "WebSearch", "TodoWrite"]
    assert options["mcp_servers"] == {}
    assert options["env"] == {}


def test_claude_workspace_adapter_builds_sdk_options_with_web_tools_when_network_enabled(
    tmp_path,
):
    adapter = ClaudeWorkspaceAdapter()
    session = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        tool_policy={"network_access": True},
    )
    policy = adapter._coerce_tool_policy(session)
    options = adapter._build_claude_sdk_options_kwargs(
        session_config=session,
        workspace=tmp_path / "workspace",
        system_prompt=adapter._build_workspace_system_prompt(policy),
        stop_oracle="public_self_eval",
        tool_policy=policy,
    )

    assert options["allowed_tools"] == [
        "Read",
        "Write",
        "Bash",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
    ]
    assert options["disallowed_tools"] == ["TodoWrite"]


def test_claude_workspace_adapter_omits_sdk_max_turns_when_unbounded(tmp_path):
    adapter = ClaudeWorkspaceAdapter()
    options = adapter._build_claude_sdk_options_kwargs(
        session_config=ExecutorSessionConfig(
            run_id="run-1",
            env_hash="env-1",
            provider_extras={"claude_max_turns": 0},
        ),
        workspace=tmp_path / "workspace",
        system_prompt=adapter._build_workspace_system_prompt(
            adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
        ),
        stop_oracle="public_self_eval",
    )
    assert options["max_turns"] is None


def test_claude_workspace_adapter_generates_ephemeral_session_ids():
    adapter = ClaudeWorkspaceAdapter()
    session_id = adapter._new_claude_session_id()
    assert str(uuid.UUID(session_id)) == session_id


def test_claude_workspace_adapter_strips_native_trace_after_session_cleanup():
    adapter = ClaudeWorkspaceAdapter()
    finalized = adapter._attach_session_cleanup(
        {
            "sdk_backend": "claude_sdk",
            "session_id": str(uuid.uuid4()),
            "matched_native_path": "C:/Users/admin/.claude/projects/demo/session.jsonl",
            "matched_native_exists": True,
        },
        session_cleanup={"requested": True, "deleted": True, "error": "", "directory": "C:/demo"},
    )
    assert finalized["matched_native_path"] == ""
    assert finalized["matched_native_exists"] is False
    assert finalized["session_cleanup"]["deleted"] is True


def test_claude_workspace_adapter_builds_sdk_env_with_task_python_first(
    tmp_path, monkeypatch
):
    adapter = ClaudeWorkspaceAdapter()
    control_env_root = tmp_path / "control_env"
    control_python = control_env_root / "python.exe"
    control_python.parent.mkdir(parents=True, exist_ok=True)
    control_python.write_text("", encoding="utf-8")
    fake_env_root = tmp_path / "task_env"
    for relative in (
        "python.exe",
        "Scripts/python.exe",
        "Library/bin/python.dll",
        "Library/usr/bin/libopenblas.dll",
        "DLLs/_ctypes.pyd",
        "conda-meta/history",
    ):
        path = fake_env_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(control_python.resolve()))
    monkeypatch.setattr(sys, "prefix", str(control_env_root.resolve()))

    sdk_env = adapter._build_workspace_sdk_env(
        {
            "PATH": os.pathsep.join(["C:\\existing\\bin", "D:\\tools"]),
            "MYEVOSKILL_RUNTIME_ROOT": "D:\\workspace",
            "MYEVOSKILL_PUBLIC_BUNDLE": "D:\\workspace\\public_bundle",
            "MYEVOSKILL_WORK_DIR": "D:\\workspace\\work",
            "MYEVOSKILL_OUTPUT_DIR": "D:\\workspace\\output",
            "MYEVOSKILL_CHECKPOINT_DIR": "D:\\workspace\\checkpoints",
            "MYEVOSKILL_WORKSPACE": "D:\\workspace",
            "MYEVOSKILL_TASK_ID": "demo-task",
            "MYEVOSKILL_TASK_PYTHON": str((fake_env_root / "python.exe").resolve()),
            "MYEVOSKILL_TASK_ENV_HASH": "env-demo-task",
            "MYEVOSKILL_TASK_ENV_BACKEND": "venv_pip",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "VIRTUAL_ENV": str(fake_env_root.resolve()),
        },
        {"ANTHROPIC_API_KEY": "sdk-secret"},
    )

    path_entries = sdk_env["PATH"].split(os.pathsep)
    assert path_entries[:5] == [
        str(fake_env_root.resolve()),
        str((fake_env_root / "Scripts").resolve()),
        str((fake_env_root / "Library" / "bin").resolve()),
        str((fake_env_root / "Library" / "usr" / "bin").resolve()),
        str((fake_env_root / "DLLs").resolve()),
    ]
    assert sdk_env["MYEVOSKILL_PYTHON_EXE"] == str((fake_env_root / "python.exe").resolve())
    assert sdk_env["CONDA_PREFIX"] == str(fake_env_root.resolve())
    assert sdk_env["ANTHROPIC_API_KEY"] == "sdk-secret"
    assert sdk_env["MYEVOSKILL_TASK_ID"] == "demo-task"
    assert sdk_env["MYEVOSKILL_TASK_PYTHON"] == str((fake_env_root / "python.exe").resolve())


def test_claude_workspace_adapter_resolves_model_name_from_env_for_logging(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    np.savez(
        bundle.public_bundle_dir / "data" / "raw_data.npz",
        measurements=np.array([[0.1, 0.2]], dtype=float),
        nu_axis=np.array([[1.0, 2.0]], dtype=float),
    )
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-workspace-env-model",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "mock_claude_sdk_response": {
                "files": {
                    "work/main.py": "\n".join(
                        [
                            "import numpy as np",
                            "from pathlib import Path",
                            "Path('output').mkdir(parents=True, exist_ok=True)",
                            "raw = np.load('data/raw_data.npz')",
                            "np.savez('output/reconstruction.npz', estimated_temperature_K=np.array([2400.0]), reconstructed_spectrum=raw['measurements'], nu_axis=raw['nu_axis'])",
                            "print('workspace-pass')",
                        ]
                    )
                    + "\n",
                    "work/src/__init__.py": "",
                    "work/src/preprocessing.py": "",
                    "work/src/physics_model.py": "",
                    "work/src/solvers.py": "",
                    "work/src/visualization.py": "",
                },
                "solver_summary": "workspace env model test",
                "declared_outputs": ["output/reconstruction.npz"],
                "assumptions": [],
                "files_written": ["work/main.py"],
                "commands_run": ["python work/main.py", "python evaluation/self_eval.py"],
                "sdk_messages": [
                    {"name": "Bash", "input": {"command": "cd . && python work/main.py"}},
                    {"name": "Bash", "input": {"command": "cd . && python evaluation/self_eval.py"}},
                ],
            }
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    monkeypatch.setenv("MYEVOSKILL_CLAUDE_MODEL", "sonnet")
    record = ClaudeWorkspaceAdapter().run(bundle, config, [])
    transcript = (tmp_path / "workspace" / "transcript.txt").read_text(encoding="utf-8")
    assert record.model_name == "sonnet"
    assert "model_name=sonnet" in transcript


def test_claude_workspace_adapter_extracts_sdk_session_id_from_messages():
    adapter = ClaudeWorkspaceAdapter()
    session_id = adapter._extract_sdk_session_id(
        [
            {"data": {"session_id": "nested-session"}},
            {"session_id": "top-level-session"},
        ]
    )
    assert session_id == "nested-session"


def test_claude_workspace_adapter_locate_native_trace_handles_missing_directory(tmp_path):
    adapter = ClaudeWorkspaceAdapter()
    native_ref = adapter._locate_claude_native_trace(tmp_path / "workspace", session_id="")
    assert native_ref["matched_native_exists"] is False
    assert native_ref["matched_native_path"] == ""


def test_claude_workspace_adapter_project_key_normalizes_underscores(tmp_path):
    adapter = ClaudeWorkspaceAdapter()
    key = adapter._claude_project_key(tmp_path / "cars_spectroscopy" / "run-real-live")
    assert "cars-spectroscopy" in key
    assert "cars_spectroscopy" not in key


def test_claude_workspace_adapter_surfaces_result_error_text():
    adapter = ClaudeWorkspaceAdapter()

    class FakeMessage:
        structured_output = None
        result = "selected model is unavailable"
        is_error = True
        subtype = "error_model_unavailable"

    with pytest.raises(RuntimeError, match="selected model is unavailable"):
        adapter._extract_sdk_summary(FakeMessage())



def test_claude_workspace_adapter_classifies_max_turns_error():
    adapter = ClaudeWorkspaceAdapter()

    class FakeMessage:
        structured_output = None
        result = ""
        is_error = True
        subtype = "error_max_turns"

    with pytest.raises(RuntimeError, match="max_turns"):
        adapter._extract_sdk_summary(FakeMessage(), [])


def test_claude_workspace_adapter_returns_failed_record_without_result_message(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-no-result-message",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    adapter = ClaudeWorkspaceAdapter()

    def fake_generate_workspace_sdk_response(**kwargs):
        workspace = kwargs["workspace"]
        (workspace / "work" / "src").mkdir(parents=True, exist_ok=True)
        (workspace / "work" / "main.py").write_text("print('no-result-message')\n", encoding="utf-8")
        raise ClaudeSDKExecutionError(
            "Claude SDK response stream ended without a ResultMessage",
            error_type="missing_result_message",
            sdk_messages=[{"name": "Bash", "input": {"command": "python work/main.py"}}],
            diagnostics={
                "timeout_occurred": False,
                "sdk_result": {
                    "subtype": "",
                    "stop_reason": "",
                    "is_error": False,
                    "num_turns": 0,
                    "session_id": "",
                },
            },
        )

    monkeypatch.setattr(adapter, "_generate_workspace_sdk_response", fake_generate_workspace_sdk_response)
    record = adapter.run(bundle, config, [])
    assert record.metadata["protocol_status"] == "failed"
    assert record.metadata["protocol_failure_reason"] == "missing_result_message"
    assert record.metadata["sdk_completion_source"] == "result_message"
    assert (tmp_path / "workspace" / "trajectory_normalized.json").exists()
    assert "Claude SDK protocol failure" in record.stderr


def test_claude_workspace_adapter_accepts_external_output_contract_without_result_message(
    tmp_path, monkeypatch
):
    bundle = make_bundle(
        tmp_path,
        {
            "output_contract": {
                "required_outputs": [
                    {
                        "path": "output/reconstruction.npz",
                        "format": "npz",
                        "required_fields": [
                            "estimated_temperature_K",
                            "reconstructed_spectrum",
                            "nu_axis",
                        ],
                    }
                ]
            },
            "proxy_spec": {
                "primary_output": "output/reconstruction.npz",
                "output_dtype": "npz",
                "required_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "numeric_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "same_shape_fields": [
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
            },
        },
    )
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-external-output-contract",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"workspace_completion_policy": "main_success_output_contract"},
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    adapter = ClaudeWorkspaceAdapter()

    def fake_generate_workspace_sdk_response(**kwargs):
        workspace = kwargs["workspace"]
        (workspace / "work" / "src").mkdir(parents=True, exist_ok=True)
        (workspace / "work" / "main.py").write_text("print('trace-success')\n", encoding="utf-8")
        (workspace / "output").mkdir(parents=True, exist_ok=True)
        np.savez(
            workspace / "output" / "reconstruction.npz",
            estimated_temperature_K=np.asarray([1200.0]),
            reconstructed_spectrum=np.zeros((1, 200), dtype=float),
            nu_axis=np.linspace(2280.0, 2330.0, 200, dtype=float).reshape(1, 200),
        )
        raise ClaudeSDKExecutionError(
            "Claude SDK response stream ended without a ResultMessage",
            error_type="missing_result_message",
            sdk_messages=[
                {
                    "content": [
                        {
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "cd . && python work/main.py"},
                        }
                    ]
                },
                {
                    "content": [
                        {
                            "tool_use_id": "tool-1",
                            "content": "trace-success",
                            "is_error": False,
                        }
                    ],
                    "tool_use_result": {
                        "stdout": "trace-success\n",
                        "stderr": "",
                        "interrupted": False,
                        "noOutputExpected": False,
                    },
                },
            ],
            diagnostics={
                "timeout_occurred": True,
                "sdk_result": {
                    "subtype": "",
                    "stop_reason": "",
                    "is_error": False,
                    "num_turns": 0,
                    "session_id": "",
                },
            },
        )

    monkeypatch.setattr(adapter, "_generate_workspace_sdk_response", fake_generate_workspace_sdk_response)
    record = adapter.run(bundle, config, [])
    assert record.metadata["protocol_status"] == "completed"
    assert record.metadata["sdk_completion_source"] == "external_output_contract"
    assert record.metadata["run_status"] == "succeeded"
    assert record.metadata["workspace_completion_policy"] == "main_success_output_contract"
    assert record.metadata["output_contract_satisfied_post_run"] is True
    assert record.metadata["public_self_eval_passed_post_run"] is False
    assert record.metadata["returncode"] == 0
    assert "python work/main.py" in " ".join(record.metadata["commands_run"])


def test_claude_workspace_adapter_requires_complete_output_contract_for_external_completion(
    tmp_path, monkeypatch
):
    bundle = make_bundle(
        tmp_path,
        {
            "output_contract": {
                "required_outputs": [
                    {
                        "path": "output/reconstruction.npz",
                        "format": "npz",
                        "required_fields": [
                            "estimated_temperature_K",
                            "reconstructed_spectrum",
                            "nu_axis",
                        ],
                    }
                ]
            },
            "proxy_spec": {
                "primary_output": "output/reconstruction.npz",
                "output_dtype": "npz",
                "required_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "numeric_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "same_shape_fields": [
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
            },
        },
    )
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-external-output-contract-missing-field",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"workspace_completion_policy": "main_success_output_contract"},
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    adapter = ClaudeWorkspaceAdapter()

    def fake_generate_workspace_sdk_response(**kwargs):
        workspace = kwargs["workspace"]
        (workspace / "work" / "src").mkdir(parents=True, exist_ok=True)
        (workspace / "work" / "main.py").write_text("print('trace-success')\n", encoding="utf-8")
        (workspace / "output").mkdir(parents=True, exist_ok=True)
        np.savez(
            workspace / "output" / "reconstruction.npz",
            estimated_temperature_K=np.asarray([1200.0]),
            reconstructed_spectrum=np.zeros((1, 200), dtype=float),
        )
        raise ClaudeSDKExecutionError(
            "Claude SDK response stream ended without a ResultMessage",
            error_type="missing_result_message",
            sdk_messages=[
                {
                    "content": [
                        {
                            "id": "tool-1",
                            "name": "Bash",
                            "input": {"command": "cd . && python work/main.py"},
                        }
                    ]
                },
                {
                    "content": [
                        {
                            "tool_use_id": "tool-1",
                            "content": "trace-success",
                            "is_error": False,
                        }
                    ],
                    "tool_use_result": {
                        "stdout": "trace-success\n",
                        "stderr": "",
                        "interrupted": False,
                        "noOutputExpected": False,
                    },
                },
            ],
            diagnostics={
                "timeout_occurred": False,
                "sdk_result": {
                    "subtype": "",
                    "stop_reason": "",
                    "is_error": False,
                    "num_turns": 0,
                    "session_id": "",
                },
            },
        )

    monkeypatch.setattr(adapter, "_generate_workspace_sdk_response", fake_generate_workspace_sdk_response)
    record = adapter.run(bundle, config, [])
    assert record.metadata["protocol_status"] == "failed"
    assert record.metadata["protocol_failure_reason"] == "missing_result_message"
    assert record.metadata["run_status"] == "failed"
    assert record.metadata["output_contract_satisfied_post_run"] is False


def test_claude_workspace_adapter_stops_after_first_sdk_result_message(tmp_path, monkeypatch):
    bundle = make_bundle(
        tmp_path,
        {
            "output_contract": {
                "required_outputs": [
                    {
                        "path": "output/reconstruction.npz",
                        "format": "npz",
                        "required_fields": [
                            "estimated_temperature_K",
                            "reconstructed_spectrum",
                            "nu_axis",
                        ],
                    }
                ]
            },
            "proxy_spec": {
                "primary_output": "output/reconstruction.npz",
                "output_dtype": "npz",
                "required_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "numeric_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "same_shape_fields": [
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
            },
        },
    )
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-multi-sdk-round",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "max_workspace_iterations": 2,
            "mock_claude_sdk_response": [
                {
                    "files": {
                        "work/main.py": "\n".join(
                            [
                                "import numpy as np",
                                "from pathlib import Path",
                                "Path('output').mkdir(parents=True, exist_ok=True)",
                                "np.savez(",
                                "    'output/reconstruction.npz',",
                                "    estimated_temperature_K=np.asarray([1200.0]),",
                                "    reconstructed_spectrum=np.zeros((1, 200), dtype=float),",
                                ")",
                                "print('round-1')",
                            ]
                        )
                        + "\n",
                    },
                    "solver_summary": "round-1",
                    "declared_outputs": ["output/reconstruction.npz"],
                    "assumptions": [],
                    "files_written": ["work/main.py"],
                    "commands_run": ["python work/main.py"],
                },
                {
                    "files": {
                        "work/main.py": "\n".join(
                            [
                                "import numpy as np",
                                "from pathlib import Path",
                                "Path('output').mkdir(parents=True, exist_ok=True)",
                                "np.savez(",
                                "    'output/reconstruction.npz',",
                                "    estimated_temperature_K=np.asarray([1200.0]),",
                                "    reconstructed_spectrum=np.zeros((1, 200), dtype=float),",
                                "    nu_axis=np.linspace(2280.0, 2330.0, 200, dtype=float).reshape(1, 200),",
                                ")",
                                "print('round-2')",
                            ]
                        )
                        + "\n",
                    },
                    "solver_summary": "round-2",
                    "declared_outputs": ["output/reconstruction.npz"],
                    "assumptions": [],
                    "files_written": ["work/main.py"],
                    "commands_run": ["python work/main.py"],
                },
            ],
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    record = ClaudeWorkspaceAdapter().run(bundle, config, [])
    assert record.metadata["iteration_count"] == 1
    assert record.metadata["run_status"] == "failed"
    assert record.metadata["output_contract_satisfied_post_run"] is False
    assert not (tmp_path / "workspace" / "agent_summary_round_2.json").exists()
    assert not (tmp_path / "workspace" / "public_self_eval_round_2.json").exists()


def test_claude_workspace_adapter_returns_specific_failure_when_submission_accepted_but_no_result_message(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-submitted-no-result",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"stop_oracle": "submit_tool"},
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    adapter = ClaudeWorkspaceAdapter()

    def fake_generate_workspace_sdk_response(**kwargs):
        workspace = kwargs["workspace"]
        (workspace / "work" / "src").mkdir(parents=True, exist_ok=True)
        (workspace / "work" / "main.py").write_text("print('submitted-no-result')\n", encoding="utf-8")
        raise ClaudeSDKExecutionError(
            "Claude SDK accepted submit_result(...) but did not return a ResultMessage",
            error_type="accepted_submission_missing_result_message",
            sdk_messages=[{"name": "submit_result", "input": {"solver_summary": "done"}}],
            diagnostics={
                "timeout_occurred": True,
                "submission_state": {
                    "submission_attempted": True,
                    "submission_accepted": True,
                    "submission_id": "round-1-submit-1",
                    "submission_rejection_reasons": [],
                    "check_ready_calls": [{"tool": "check_ready"}],
                    "submit_result_calls": [{"tool": "submit_result"}],
                    "submission_events": [
                        {"tool": "check_ready"},
                        {"tool": "submit_result"},
                    ],
                },
                "sdk_result": {
                    "subtype": "",
                    "stop_reason": "",
                    "is_error": False,
                    "num_turns": 0,
                    "session_id": "",
                },
            },
        )

    monkeypatch.setattr(adapter, "_generate_workspace_sdk_response", fake_generate_workspace_sdk_response)
    record = adapter.run(bundle, config, [])
    assert record.metadata["protocol_status"] == "failed"
    assert (
        record.metadata["protocol_failure_reason"]
        == "accepted_submission_missing_result_message"
    )
    assert record.metadata["submission_attempted"] is True
    assert record.metadata["submission_accepted"] is True
    assert record.metadata["submission_id"] == "round-1-submit-1"


def test_claude_workspace_adapter_installs_public_self_eval_runtime(tmp_path):
    bundle = make_bundle(
        tmp_path,
        {
            "output_contract": {
                "required_outputs": [
                    {
                        "path": "output/reconstruction.npz",
                        "format": "npz",
                        "required_fields": [
                            "estimated_temperature_K",
                            "reconstructed_spectrum",
                            "nu_axis",
                        ],
                    }
                ]
            },
            "proxy_spec": {
                "primary_output": "output/reconstruction.npz",
                "output_dtype": "npz",
                "required_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "numeric_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "same_shape_fields": [
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
            },
        },
    )
    _, runtime_paths = _prepare_runtime_workspace(bundle, tmp_path / "workspace")
    adapter = ClaudeWorkspaceAdapter()
    tool_policy = adapter._coerce_tool_policy(
        ExecutorSessionConfig(run_id="run-1", env_hash="env-1")
    )
    spec = adapter._install_public_self_eval_runtime(
        task_spec=json.loads(bundle.task_spec_path.read_text(encoding="utf-8")),
        runtime_paths=runtime_paths,
        tool_policy=tool_policy,
    )
    self_eval_script = runtime_paths["runtime_root"] / "evaluation" / "self_eval.py"
    self_eval_spec = runtime_paths["runtime_root"] / "evaluation" / "self_eval_spec.json"
    assert self_eval_script.exists()
    assert self_eval_spec.exists()
    assert spec["required_outputs"][0]["path"] == "output/reconstruction.npz"
    assert "evaluation" not in spec["readonly_roots"]
    serialized = json.dumps(spec, sort_keys=True) + self_eval_script.read_text(encoding="utf-8")
    assert not any(metric in serialized for metric in ("ncc", "nrmse", "temperature_error"))


def test_claude_workspace_adapter_public_self_eval_detects_public_failures(tmp_path):
    bundle = make_bundle(
        tmp_path,
        {
            "output_contract": {
                "required_outputs": [
                    {
                        "path": "output/reconstruction.npz",
                        "format": "npz",
                        "required_fields": [
                            "estimated_temperature_K",
                            "reconstructed_spectrum",
                            "nu_axis",
                        ],
                    }
                ]
            },
            "proxy_spec": {
                "primary_output": "output/reconstruction.npz",
                "output_dtype": "npz",
                "required_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "numeric_fields": [
                    "estimated_temperature_K",
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
                "same_shape_fields": [
                    "reconstructed_spectrum",
                    "nu_axis",
                ],
            },
            "public_eval_spec": {
                "alignments": [
                    {
                        "output_path": "output/reconstruction.npz",
                        "field": "nu_axis",
                        "source_path": "data/raw_data.npz",
                        "source_field": "nu_axis",
                        "mode": "allclose",
                    }
                ]
            },
        },
    )
    (bundle.public_bundle_dir / "data").mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez(
        bundle.public_bundle_dir / "data" / "raw_data.npz",
        measurements=np.array([[0.1, 0.2]], dtype=float),
        nu_axis=np.array([[1.0, 2.0]], dtype=float),
    )
    _, runtime_paths = _prepare_runtime_workspace(bundle, tmp_path / "workspace")
    adapter = ClaudeWorkspaceAdapter()
    session = ExecutorSessionConfig(run_id="run-1", env_hash="env-1")
    tool_policy = adapter._coerce_tool_policy(
        session
    )
    adapter._install_public_self_eval_runtime(
        task_spec=json.loads(bundle.task_spec_path.read_text(encoding="utf-8")),
        runtime_paths=runtime_paths,
        tool_policy=tool_policy,
    )

    np.savez(
        runtime_paths["runtime_root"] / "output" / "reconstruction.npz",
        estimated_temperature_K=np.array(["hot"]),
        reconstructed_spectrum=np.array([[0.1, np.nan]], dtype=float),
        nu_axis=np.array([[5.0, 6.0, 7.0]], dtype=float),
    )
    runtime_paths["runtime_root"].joinpath("README_public.md").write_text(
        "mutated readme",
        encoding="utf-8",
    )
    audit = adapter._public_self_check(
        task_spec=json.loads(bundle.task_spec_path.read_text(encoding="utf-8")),
        runtime_paths=runtime_paths,
        completed=subprocess.CompletedProcess(["python", "work/main.py"], 0, "", ""),
        timed_out=False,
        entrypoint="work/main.py",
        env=_runtime_environment(session, bundle, runtime_paths),
        timeout_seconds=30,
    )
    assert audit["self_check_passed"] is False
    assert audit["public_self_eval_passed"] is False
    serialized = json.dumps(audit, sort_keys=True)
    assert "missing required field" not in serialized
    assert "non-numeric field: estimated_temperature_K" in serialized
    assert "nan_or_inf field: reconstructed_spectrum" in serialized
    assert "required fields have inconsistent shapes" in serialized
    assert "public alignment failed: output/reconstruction.npz:nu_axis != data/raw_data.npz:nu_axis" in serialized
    assert "read-only path modified: README_public.md" in serialized

def test_claude_workspace_adapter_allows_low_risk_workspace_bash_commands(tmp_path):
    adapter = ClaudeWorkspaceAdapter()
    policy = adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
    violations = adapter._validate_bash_commands(
        ["mkdir -p work/src && ls work/", f"cd \"{(tmp_path / 'workspace').resolve()}\" && python -c \"print(1)\"", "pwd", "echo ready"],
        tmp_path / "workspace",
        policy,
    )
    assert violations == []


def test_claude_workspace_adapter_allows_empty_file_creation_inside_write_roots(tmp_path):
    adapter = ClaudeWorkspaceAdapter()
    policy = adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    violations = adapter._validate_bash_commands(
        [
            "touch work/src/__init__.py",
            f"cd \"{workspace.resolve()}\" && touch work/src/__init__.py",
            f"touch \"{(workspace / 'work' / 'src' / '__init__.py').resolve()}\"",
            r"type nul > work\src\__init__.py",
            'powershell -Command "New-Item -ItemType File -Force work/src/__init__.py | Out-Null"',
            'pwsh -Command "New-Item -ItemType File -Force work/src/__init__.py | Out-Null"',
        ],
        workspace,
        policy,
    )
    assert violations == []


@pytest.mark.parametrize(
    "command",
    [
        "touch ../outside.txt",
        r"touch C:\abs\path.txt",
        "touch data/raw_data.npz",
        'pwsh -Command "New-Item -ItemType File -Force data/raw_data.npz | Out-Null"',
        "mkdir data/cache",
        r"copy C:\external.txt work\x.txt",
    ],
)
def test_claude_workspace_adapter_rejects_out_of_policy_file_creation_commands(
    tmp_path,
    command,
):
    adapter = ClaudeWorkspaceAdapter()
    policy = adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    violations = adapter._validate_bash_commands([command], workspace, policy)

    assert len(violations) == 1
    assert violations[0]["command"] == command
    assert violations[0]["category"] in {"outside_workspace_path", "outside_write_roots"}


def test_claude_workspace_adapter_allows_workspace_absolute_paths_timeout_and_readonly_diagnostics(
    tmp_path,
):
    adapter = ClaudeWorkspaceAdapter()
    policy = adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    work_src = workspace / "work" / "src"
    output_dir = workspace / "output"
    violations = adapter._validate_bash_commands(
        [
            f'mkdir -p "{work_src.resolve()}" "{output_dir.resolve()}"',
            f'cd "{workspace.resolve()}" && timeout 60 python work/main.py',
            f'cd "{workspace.resolve()}" && timeout 10 python -c "print(1)"',
            'ps aux | grep -i "python work/main.py" | grep -v grep',
            'powershell -Command "Get-ChildItem work"',
            'tasklist | findstr python',
        ],
        workspace,
        policy,
    )
    assert violations == []


def test_claude_workspace_adapter_allows_workspace_heredoc_file_creation(tmp_path):
    adapter = ClaudeWorkspaceAdapter()
    policy = adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    command = "\n".join(
        [
            f'cd "{workspace.resolve()}" && cat > work/src/example.py << \'EOFPYTHON\'',
            "if value > 10:",
            "    print(value)",
            "EOFPYTHON",
        ]
    )

    violations = adapter._validate_bash_commands([command], workspace, policy)

    assert violations == []


@pytest.mark.parametrize(
    ("command", "expected_detail"),
    [
        ("curl https://example.invalid", "network access"),
        ('Invoke-WebRequest https://example.invalid', "network access"),
        ("pip install numpy", "package or environment installation"),
        ("conda install numpy", "package or environment installation"),
        ("git push", "version control"),
        ("taskkill /F /IM python.exe", "killing processes"),
        ("Remove-Service demo", "system or service control"),
    ],
)
def test_claude_workspace_adapter_rejects_denied_categories(tmp_path, command, expected_detail):
    adapter = ClaudeWorkspaceAdapter()
    policy = adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    violations = adapter._validate_bash_commands([command], workspace, policy)

    assert len(violations) == 1
    assert violations[0]["command"] == command
    assert violations[0]["category"] == "denied_category"
    assert expected_detail in violations[0]["detail"]


def test_claude_workspace_tool_policy_allows_network_fetch_tokens_when_enabled():
    adapter = ClaudeWorkspaceAdapter()
    policy = adapter._coerce_tool_policy(
        ExecutorSessionConfig(
            run_id="run-1",
            env_hash="env-1",
            tool_policy={"network_access": True},
        )
    )

    assert policy["network_access"] is True
    assert "curl" not in [str(item).lower() for item in policy["bash_denied_tokens"]]
    assert "wget" not in [str(item).lower() for item in policy["bash_denied_tokens"]]
    assert "ssh" in [str(item).lower() for item in policy["bash_denied_tokens"]]
    assert "scp" in [str(item).lower() for item in policy["bash_denied_tokens"]]


def test_claude_workspace_adapter_allows_curl_when_network_enabled(tmp_path):
    adapter = ClaudeWorkspaceAdapter()
    policy = adapter._coerce_tool_policy(
        ExecutorSessionConfig(
            run_id="run-1",
            env_hash="env-1",
            tool_policy={"network_access": True},
        )
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    violations = adapter._validate_bash_commands(["curl https://example.invalid"], workspace, policy)

    assert violations == []


def test_claude_workspace_system_prompt_describes_boundary_policy_when_network_disabled():
    adapter = ClaudeWorkspaceAdapter()
    prompt = adapter._build_workspace_system_prompt(
        adapter._coerce_tool_policy(ExecutorSessionConfig(run_id="run-1", env_hash="env-1"))
    )

    assert prompt["type"] == "preset"
    assert "Allowed Bash prefixes" not in prompt["append"]
    assert "Denied Bash tokens" not in prompt["append"]
    assert "Any Bash command that stays inside the workspace" in prompt["append"]
    assert "Prohibited external side effects include network access" in prompt["append"]
    assert "Network access is disabled by the harness." in prompt["append"]
    assert "WebSearch" not in prompt["append"]
    assert "paper-first" not in prompt["append"]


def test_claude_workspace_system_prompt_encourages_network_research_when_enabled():
    adapter = ClaudeWorkspaceAdapter()
    prompt = adapter._build_workspace_system_prompt(
        adapter._coerce_tool_policy(
            ExecutorSessionConfig(
                run_id="run-1",
                env_hash="env-1",
                tool_policy={"network_access": True},
            )
        )
    )

    assert "Network access is enabled only because the harness allowed it" in prompt["append"]
    assert "Before writing code, do one brief paper-first external search" in prompt["append"]
    assert "Prioritize papers, project pages, and paper abstract pages" in prompt["append"]
    assert "WebSearch" in prompt["append"]
    assert "WebFetch" in prompt["append"]
    assert "Prohibited external side effects still include package or environment installation" in prompt["append"]
    assert "Prohibited external side effects include network access" not in prompt["append"]


def test_claude_workspace_adapter_rejects_disallowed_bash_commands(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="TEST_CLAUDE_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-bad-bash",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "mock_claude_sdk_response": {
                "files": {
                    "work/main.py": "print('noop')\n",
                    "work/src/__init__.py": "",
                    "work/src/preprocessing.py": "",
                    "work/src/physics_model.py": "",
                    "work/src/solvers.py": "",
                    "work/src/visualization.py": "",
                },
                "solver_summary": "tries disallowed bash",
                "declared_outputs": [],
                "assumptions": [],
                "files_written": ["work/main.py"],
                "commands_run": ["curl https://example.invalid"],
            }
        },
    )
    monkeypatch.setenv("TEST_CLAUDE_API_KEY", "sdk-secret")
    with pytest.raises(RuntimeError, match="disallowed bash commands"):
        ClaudeWorkspaceAdapter().run(bundle, config, [])


def test_inspect_bridge_adapter_sanitizes_fenced_python_response(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    generated_code = "```python\nprint('hello from fenced code')\n```"
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": generated_code},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert "hello from fenced code" in record.stdout
    script = (tmp_path / "workspace" / "work" / "agent_solution.py").read_text(encoding="utf-8")
    assert script.strip() == "print('hello from fenced code')"


def test_inspect_bridge_adapter_rejects_non_code_model_response(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": "I cannot help with that request."},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    with pytest.raises(RuntimeError, match="executable Python code") as exc_info:
        InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert "executable Python code" in str(exc_info.value)
    parse_error = (tmp_path / "workspace" / "response_parse_error.txt").read_text(encoding="utf-8")
    raw_response = (tmp_path / "workspace" / "raw_response.txt").read_text(encoding="utf-8")
    assert "candidate_count=" in parse_error
    assert "I cannot help with that request." in raw_response


def test_inspect_bridge_adapter_extracts_code_after_explanatory_prefix(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    generated_code = "\n".join(
        [
            "Here is the solution:",
            "import os",
            "print('trimmed-prefix')",
        ]
    )
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": generated_code},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert "trimmed-prefix" in record.stdout
    script = (tmp_path / "workspace" / "work" / "agent_solution.py").read_text(encoding="utf-8")
    assert script.startswith("import os")
    assert record.metadata["response_format"] == "plain_text_code"
    assert "Expecting value" in record.metadata["response_parse_error"]
    assert record.metadata["response_selected_source"] == "plain_text_fallback"


def test_inspect_bridge_adapter_accepts_structured_json_response(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    structured_response = {
        "python_code": "print('json-protocol')",
        "declared_outputs": ["output/reconstruction.npz"],
        "assumptions": ["public data is sufficient"],
        "solver_summary": "direct json protocol test",
    }
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": json.dumps(structured_response)},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert "json-protocol" in record.stdout
    assert record.metadata["response_format"] == "structured_json"
    assert record.metadata["response_selected_source"] in {"full_text", "balanced_json_slice"}
    assert record.metadata["parsed_response"]["declared_outputs"] == ["output/reconstruction.npz"]
    assert record.metadata["parsed_response"]["assumptions"] == ["public data is sufficient"]
    assert record.metadata["parsed_response"]["solver_summary"] == "direct json protocol test"


def test_inspect_bridge_adapter_accepts_fenced_structured_json_response(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    structured_response = """```json
{"python_code": "print('fenced-json')", "declared_outputs": ["output/reconstruction.npz"]}
```"""
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": structured_response},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert "fenced-json" in record.stdout
    assert record.metadata["response_format"] == "structured_json"
    assert record.metadata["response_selected_source"] == "fenced_json_block"
    assert record.metadata["parsed_response"]["solver_summary"] == ""


def test_inspect_bridge_adapter_structured_json_defaults_optional_fields(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": json.dumps({"python_code": "print('defaults')"})},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert "defaults" in record.stdout
    assert record.metadata["parsed_response"]["declared_outputs"] == []
    assert record.metadata["parsed_response"]["assumptions"] == []
    assert record.metadata["parsed_response"]["solver_summary"] == ""


def test_inspect_bridge_adapter_structured_json_requires_python_code(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": json.dumps({"solver_summary": "missing code"})},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    with pytest.raises(RuntimeError, match="missing required field 'python_code'"):
        InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    raw_response = (tmp_path / "workspace" / "raw_response.txt").read_text(encoding="utf-8")
    assert "missing code" in raw_response


def test_inspect_bridge_adapter_falls_back_to_plain_text_when_json_is_malformed(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    malformed_then_code = "\n".join(
        [
            "```json",
            '{"python_code": "print("broken")"}',
            "```",
            "import os",
            "print('fallback-after-bad-json')",
        ]
    )
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": malformed_then_code},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert "fallback-after-bad-json" in record.stdout
    assert record.metadata["response_format"] == "plain_text_code"
    assert "Expecting ',' delimiter" in record.metadata["response_parse_error"]


def test_inspect_bridge_adapter_handles_realistic_prefixed_fenced_json_response(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    python_code = "\n".join(
        [
            "import json",
            "payload = {'alpha': 1, 'beta': {'gamma': 2}}",
            "name = 'demo'",
            "print(f\"solver:{name}:{payload['alpha']}\")",
        ]
    )
    structured = {
        "python_code": python_code,
        "declared_outputs": ["output/reconstruction.npz"],
        "assumptions": ["dictionary literals and f-strings are allowed"],
        "solver_summary": "realistic long fenced json payload",
    }
    raw_response = "Here is my solution:\n\n```json\n" + json.dumps(structured) + "\n```"
    config = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={"mock_llm_response": raw_response},
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, [])
    assert "solver:demo:1" in record.stdout
    assert record.metadata["response_format"] == "structured_json"
    assert record.metadata["response_selected_source"] == "fenced_json_block"
    assert record.metadata["parsed_response"]["solver_summary"] == "realistic long fenced json payload"


def test_local_runner_stages_runtime_root_for_relative_data_access(tmp_path):
    bundle = make_bundle(tmp_path)
    data_dir = bundle.public_bundle_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "input.txt").write_text("runtime-staged-data", encoding="utf-8")
    script = bundle.public_bundle_dir / "runner.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "raw = Path('data/input.txt').read_text(encoding='utf-8')",
                "Path('output').mkdir(parents=True, exist_ok=True)",
                "Path('output/result.txt').write_text(raw.upper(), encoding='utf-8')",
                "print(raw)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = ExecutorSessionConfig(
        run_id="run-relative",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        command=("python3", "runner.py"),
    )
    record = LocalRunnerAdapter().run(bundle, config, [])
    assert "runtime-staged-data" in record.stdout
    assert (tmp_path / "workspace" / "output" / "result.txt").read_text(encoding="utf-8") == (
        "RUNTIME-STAGED-DATA"
    )
    assert (tmp_path / "workspace" / "public_bundle" / "data" / "input.txt").exists()
    assert (bundle.public_bundle_dir / "output").exists() is False


def test_local_runner_rerun_clears_work_and_output_without_mutating_public_bundle(tmp_path):
    bundle = make_bundle(tmp_path)
    script = bundle.public_bundle_dir / "runner.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "Path('work').mkdir(parents=True, exist_ok=True)",
                "Path('output').mkdir(parents=True, exist_ok=True)",
                "Path('work/current.txt').write_text('fresh', encoding='utf-8')",
                "Path('output/current.txt').write_text('fresh', encoding='utf-8')",
                "print('rerun-ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = ExecutorSessionConfig(
        run_id="run-rerun",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        command=("python3", "runner.py"),
    )
    LocalRunnerAdapter().run(bundle, config, [])
    (tmp_path / "workspace" / "work" / "stale.txt").write_text("stale", encoding="utf-8")
    (tmp_path / "workspace" / "output" / "stale.txt").write_text("stale", encoding="utf-8")

    record = LocalRunnerAdapter().run(bundle, config, [])
    assert "rerun-ok" in record.stdout
    assert not (tmp_path / "workspace" / "work" / "stale.txt").exists()
    assert not (tmp_path / "workspace" / "output" / "stale.txt").exists()
    assert (bundle.public_bundle_dir / "runner.py").read_text(encoding="utf-8").startswith(
        "from pathlib import Path"
    )


def test_local_runner_marks_timeout_and_preserves_timeout_metadata(tmp_path):
    bundle = make_bundle(
        tmp_path,
        {
            "runtime_policy": {
                "model_timeout_seconds": 120,
                "execution_budget_seconds": 1,
            }
        },
    )
    script = bundle.public_bundle_dir / "runner.py"
    script.write_text(
        "\n".join(
            [
                "import time",
                "print('start')",
                "time.sleep(5)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    record = LocalRunnerAdapter().run(
        bundle,
        ExecutorSessionConfig(
            run_id="run-timeout",
            env_hash="env-1",
            workspace_root=tmp_path / "workspace",
            command=("python3", "runner.py"),
        ),
        [],
    )
    assert record.metadata["timed_out"] is True
    assert record.metadata["timeout_scope"] == "solver_execution"
    assert record.metadata["effective_execution_budget_seconds"] == 1
    assert "timed out after 1 seconds" in record.stderr



def test_local_runner_normalizes_python_launcher_on_windows_style_hosts(tmp_path):
    bundle = make_bundle(tmp_path)
    script = bundle.public_bundle_dir / "runner.py"
    script.write_text("import sys\nprint(sys.executable)\n", encoding="utf-8")
    record = LocalRunnerAdapter().run(
        bundle,
        ExecutorSessionConfig(
            run_id="run-python-launcher",
            env_hash="env-1",
            workspace_root=tmp_path / "workspace",
            command=("python3", "runner.py"),
        ),
        [],
    )
    assert sys.executable.lower() in record.stdout.lower()


def test_inspect_bridge_adapter_returns_timeout_record_for_model_request_timeout(
    tmp_path, monkeypatch
):
    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )

    def raising_urlopen(*args, **kwargs):
        raise urllib.error.URLError(reason=TimeoutError("request timed out"))

    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    monkeypatch.setattr("urllib.request.urlopen", raising_urlopen)
    record = InspectBridgeAdapter(inspect_available=True).run(
        bundle,
        ExecutorSessionConfig(
            run_id="run-request-timeout",
            env_hash="env-1",
            workspace_root=tmp_path / "workspace",
            model_config=model,
            provider_extras={"inspect_legacy_bridge": True},
        ),
        [],
    )
    assert record.metadata["timed_out"] is True
    assert record.metadata["timeout_scope"] == "model_request"
    assert record.metadata["effective_model_timeout_seconds"] == 120
    assert "timed out after 120 seconds" in record.stderr


def test_inspect_bridge_adapter_runs_native_agent_with_inspect_tools(tmp_path, monkeypatch):
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import ModelOutput

    bundle = make_bundle(tmp_path)
    model = ModelConfig(
        provider_name="openai-compatible",
        model_name="Vendor2/Claude-4.5-Sonnet",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        temperature=0.0,
    )
    create_workspace = """python - <<'PY'
from pathlib import Path

files = {
    "output/plan.md": "# Task Understanding\\n\\n## Input/Output Contract\\n\\n## Module Layout\\n\\n## Algorithm Choice\\n\\n## Implementation Steps\\n\\n## Validation Plan\\n\\n## Assumptions\\n",
    "work/src/__init__.py": "",
    "work/src/preprocessing.py": "def load_inputs():\\n    return None\\n",
    "work/src/physics_model.py": "def forward_model(value):\\n    return value\\n",
    "work/src/solvers.py": "def solve(value):\\n    return value\\n",
    "work/src/visualization.py": "def summarize(value):\\n    return value\\n",
    "work/main.py": "from pathlib import Path\\nimport numpy as np\\n\\n\\ndef main():\\n    Path('output').mkdir(parents=True, exist_ok=True)\\n    np.savez('output/reconstruction.npz', reconstruction=np.array([1.0], dtype=np.float32))\\n    print('native-main-ran')\\n\\n\\nif __name__ == '__main__':\\n    main()\\n",
}

for relative_path, content in files.items():
    target = Path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
PY"""
    config = ExecutorSessionConfig(
        run_id="run-native-inspect",
        env_hash="env-1",
        workspace_root=tmp_path / "workspace",
        model_config=model,
        provider_extras={
            "mock_native_inspect_outputs": [
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="bash",
                    tool_arguments={"command": create_workspace},
                ),
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="bash",
                    tool_arguments={"command": "python work/main.py"},
                ),
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="bash",
                    tool_arguments={"command": "python evaluation/self_eval.py"},
                ),
                ModelOutput.for_tool_call(
                    model="mockllm/model",
                    tool_name="submit",
                    tool_arguments={"answer": "native inspect complete"},
                ),
            ]
        },
    )

    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    record = InspectBridgeAdapter(inspect_available=True).run(bundle, config, ["skill-a"])

    assert record.metadata["native_agent_used"] is True
    assert record.metadata["bridge_mode"] == "inspect_native_agent"
    assert record.metadata["sdk_backend"] == "inspect_ai_native_local"
    assert record.metadata["native_tool_trace_present"] is True
    assert record.metadata["native_command_trace_present"] is True
    assert record.metadata["public_self_eval_passed_post_run"] is True
    assert record.metadata["plan_artifact_present"] is True
    assert record.metadata["default_scientific_layout_present"] is True
    assert "python work/main.py" in record.metadata["commands_run"]
    assert "python evaluation/self_eval.py" in record.metadata["commands_run"]
    assert "work/main.py" in record.metadata["files_written"]
    assert "output/plan.md" in record.metadata["files_written"]
    assert (tmp_path / "workspace" / "output" / "plan.md").exists()
    assert (tmp_path / "workspace" / "output" / "reconstruction.npz").exists()
    assert "native inspect complete" in record.stdout
