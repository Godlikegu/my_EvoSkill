import json
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from myevoskill.compile_audit import HeuristicCompileAuditAdapter
from myevoskill.compiler import TaskBundleCompiler
from myevoskill.executor import ClaudeWorkspaceAdapter, InspectBridgeAdapter, LocalRunnerAdapter
from myevoskill.logging_utils import RunLogger
from myevoskill.model_provider import ClaudeSDKAdapter
from myevoskill.models import ExecutorSessionConfig, ModelConfig, SkillCandidate, TaskOutcome
from myevoskill.proxy import ProxyVerifier
from myevoskill.registry import SkillRegistry
from myevoskill.task_adapters import (
    cars_spectroscopy_proxy_spec,
    cars_spectroscopy_public_policy,
    evaluate_cars_spectroscopy_run,
    manifest_proxy_spec,
)
from myevoskill.task_runtime import ensure_clean_run_directory, resolve_run_paths
from myevoskill.validation import TransferValidator


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_real_cars_manifest():
    manifest = json.loads(
        (REPO_ROOT / "MyEvoSkill" / "registry" / "tasks" / "cars_spectroscopy.json").read_text(
            encoding="utf-8"
        )
    )
    source_task_dir = Path(manifest.get("source_task_dir", ""))
    if source_task_dir and not source_task_dir.is_absolute():
        manifest["source_task_dir"] = str((PROJECT_ROOT / source_task_dir).resolve())
    return manifest


def _write_live_run_logs(log_root, record, proxy=None, judge=None, validation=None):
    log_root = Path(log_root)
    logger = RunLogger(log_root.parent)
    log_dir = logger.create_run_dir(log_root.name)
    logger.write_summary(
        log_dir,
        {
            "provider": record.provider,
            "returncode": record.metadata.get("returncode"),
            "bridge_mode": record.metadata.get("bridge_mode"),
            "timed_out": record.metadata.get("timed_out", False),
            "timeout_scope": record.metadata.get("timeout_scope", ""),
            "raw_response_preview": record.metadata.get("raw_response_preview", ""),
            "validation_reason": getattr(validation, "decision_reason", ""),
        },
    )
    logger.write_json_artifact(
        log_dir,
        "executor_config.json",
        {
            "provider": record.provider,
            "model_provider_kind": record.metadata.get("model_provider_kind", ""),
            "api_key_env": record.metadata.get("api_key_env", ""),
            "command_history_summary": record.metadata.get("command_history_summary", []),
        },
    )
    logger.write_json_artifact(
        log_dir,
        "model_config.json",
        {
            "model_provider": record.model_provider,
            "model_name": record.model_name,
            "api_key_env": record.metadata.get("api_key_env", ""),
        },
    )
    logger.append_text_log(log_dir, "stdout.log", record.stdout)
    logger.append_text_log(log_dir, "stderr.log", record.stderr)
    if "raw_response_text" in record.metadata:
        logger.append_text_log(log_dir, "raw_response.txt", record.metadata["raw_response_text"])
    if "parsed_response" in record.metadata:
        logger.write_json_artifact(log_dir, "parsed_response.json", record.metadata["parsed_response"])
    if "response_parse_error" in record.metadata:
        logger.append_text_log(
            log_dir,
            "response_parse_error.txt",
            str(record.metadata["response_parse_error"]),
        )
    transcript = Path(record.transcript_uri)
    if transcript.exists():
        logger.append_text_log(log_dir, "transcript.txt", transcript.read_text(encoding="utf-8"))
    agent_script = Path(record.workspace_root) / "work" / "agent_solution.py"
    if agent_script.exists():
        logger.append_text_log(log_dir, "agent_solution.py", agent_script.read_text(encoding="utf-8"))
    workspace_main = Path(record.workspace_root) / "work" / "main.py"
    if workspace_main.exists():
        logger.append_text_log(log_dir, "work_main.py", workspace_main.read_text(encoding="utf-8"))
    for round_file in sorted(Path(record.workspace_root).glob("*_round_*")):
        if round_file.suffix == ".json":
            logger.append_text_log(log_dir, round_file.name, round_file.read_text(encoding="utf-8"))
        elif round_file.suffix in {".txt", ".log"}:
            logger.append_text_log(log_dir, round_file.name, round_file.read_text(encoding="utf-8"))
    for extra_name in [
        "trajectory_native.jsonl",
        "trajectory_normalized.json",
        "trajectory_summary.json",
        "trajectory_redaction_report.json",
        "vendor_session_ref.json",
    ]:
        extra_path = Path(record.workspace_root) / extra_name
        if extra_path.exists():
            logger.append_text_log(log_dir, extra_name, extra_path.read_text(encoding="utf-8"))
    if proxy is not None:
        logger.write_json_artifact(log_dir, "proxy_feedback.json", proxy)
    if judge is not None:
        logger.write_json_artifact(log_dir, "judge_result.json", judge)
    if validation is not None:
        logger.write_json_artifact(log_dir, "validation_result.json", validation)
    return log_dir


def test_cars_spectroscopy_compile_local_run_and_validation_minimal_flow(tmp_path):
    task_root = tmp_path / "cars_spectroscopy"
    (task_root / "data").mkdir(parents=True)
    (task_root / "evaluation").mkdir(parents=True)
    (task_root / "README.md").write_text(
        "Implementation note: keep this.\nReference output note: keep this.\nDo not expose data/ground_truth.npz.\n",
        encoding="utf-8",
    )
    (task_root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    np.savez(
        task_root / "data" / "raw_data.npz",
        measurements=np.array([[0.1, 0.5, 0.9]], dtype=float),
        nu_axis=np.array([[2280.0, 2290.0, 2300.0]], dtype=float),
    )
    (task_root / "data" / "meta_data.json").write_text("{}", encoding="utf-8")
    np.savez(
        task_root / "data" / "ground_truth.npz",
        spectrum=np.array([[0.1, 0.5, 0.9]], dtype=float),
        temperature=np.array([2400.0], dtype=float),
        x_mol=np.array([0.79], dtype=float),
    )
    (task_root / "evaluation" / "metrics.json").write_text(
        json.dumps(
            {
                "ncc_boundary": 0.8,
                "nrmse_boundary": 0.1,
                "temperature_error_K_boundary": 50.0,
            }
        ),
        encoding="utf-8",
    )

    compiler = TaskBundleCompiler(tmp_path / "compiled")
    bundle = compiler.compile(
        task_root,
        task_id="cars_spectroscopy",
        family="chemistry",
        public_policy={
            "readme_policy": {
                "preserve_user_eval_notes": True,
                "remove_path_patterns": ["(?i)data/ground_truth\\.npz"],
            },
            "public_data_allowlist": list(cars_spectroscopy_public_policy().public_data_allowlist),
            "public_data_denylist": list(cars_spectroscopy_public_policy().public_data_denylist),
        },
    )
    assert "Implementation note" in bundle.readme_public_path.read_text(encoding="utf-8")

    workspace = tmp_path / "run"
    script = bundle.public_bundle_dir / "emit_result.py"
    script.write_text(
        "\n".join(
            [
                "import os",
                "import numpy as np",
                "output_dir = os.path.join(os.getcwd(), os.environ['MYEVOSKILL_WORKSPACE'], 'output')",
                "os.makedirs(output_dir, exist_ok=True)",
                "raw = np.load('data/raw_data.npz')",
                "np.savez(",
                "    os.path.join(output_dir, 'reconstruction.npz'),",
                "    estimated_temperature_K=np.array([2400.0]),",
                "    reconstructed_spectrum=raw['measurements'],",
                "    nu_axis=raw['nu_axis'],",
                ")",
                "print('done')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    session = ExecutorSessionConfig(
        run_id="run-1",
        env_hash="env-1",
        workspace_root=workspace,
        command=("python3", "emit_result.py"),
        env={"MYEVOSKILL_WORKSPACE": str(workspace)},
    )
    record = LocalRunnerAdapter().run(bundle, session, ["skill-alpha"])
    assert "done" in record.stdout

    proxy = ProxyVerifier().evaluate(record, cars_spectroscopy_proxy_spec(record))
    assert proxy.output_exists is True

    judge = evaluate_cars_spectroscopy_run(task_root, record)
    assert judge.all_metrics_passed is True

    logger = RunLogger(tmp_path / "logs")
    log_dir = logger.create_run_dir(record.run_id)
    logger.write_summary(log_dir, {"provider": record.provider, "warnings": proxy.warnings})
    logger.write_json_artifact(log_dir, "proxy_feedback.json", proxy)
    logger.write_json_artifact(log_dir, "judge_result.json", judge)

    validator = TransferValidator()
    validation = validator.evaluate(
        baseline_outcomes={
            "cars_spectroscopy": TaskOutcome(task_id="cars_spectroscopy", all_metrics_passed=False),
            "task-2": TaskOutcome(task_id="task-2", all_metrics_passed=True),
        },
        treatment_outcomes={
            "cars_spectroscopy": TaskOutcome(
                task_id="cars_spectroscopy",
                all_metrics_passed=judge.all_metrics_passed,
                metrics_actual=judge.metrics_actual,
                failed_metrics=judge.failed_metrics,
                failure_tags=judge.failure_tags,
            ),
            "task-2": TaskOutcome(task_id="task-2", all_metrics_passed=True),
        },
    )
    registry = SkillRegistry(tmp_path / "registry")
    entry = registry.register(
        SkillCandidate(
            skill_id="skill-alpha",
            version="0.1.0",
            description="strictly better skill",
            source_run_ids=[record.run_id],
            legal_source=True,
            reusable=True,
        ),
        validation,
    )
    assert entry.status == "validated"


def test_inspect_bridge_forced_missing_dependency_still_supports_minimal_pipeline(
    tmp_path,
):
    task_root = tmp_path / "cars_spectroscopy_fallback"
    (task_root / "data").mkdir(parents=True)
    (task_root / "evaluation").mkdir(parents=True)
    (task_root / "README.md").write_text(
        "Implementation note: keep this.\nReference output note: keep this.\n",
        encoding="utf-8",
    )
    (task_root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (task_root / "data" / "raw_data.npz").write_text("raw", encoding="utf-8")
    (task_root / "data" / "meta_data.json").write_text("{}", encoding="utf-8")
    (task_root / "evaluation" / "metrics.json").write_text(
        json.dumps(
            {
                "ncc_boundary": 0.8,
                "nrmse_boundary": 0.1,
                "temperature_error_K_boundary": 50.0,
            }
        ),
        encoding="utf-8",
    )

    compiler = TaskBundleCompiler(tmp_path / "compiled")
    bundle = compiler.compile(
        task_root,
        task_id="cars_spectroscopy_fallback",
        family="chemistry",
        public_policy={
            "readme_policy": {
                "preserve_user_eval_notes": True,
                "remove_path_patterns": [],
            },
            "public_data_allowlist": list(cars_spectroscopy_public_policy().public_data_allowlist),
            "public_data_denylist": list(cars_spectroscopy_public_policy().public_data_denylist),
        },
    )

    workspace = tmp_path / "run_fallback"
    session = ExecutorSessionConfig(
        run_id="run-fallback",
        env_hash="env-1",
        workspace_root=workspace,
        provider_extras={"allow_fallback": True},
    )
    record = InspectBridgeAdapter(inspect_available=False).run(bundle, session, ["skill-alpha"])

    assert record.provider == "inspect_unavailable_fallback"
    assert record.metadata["fallback_reason"] == "inspect_ai_missing"

    proxy = ProxyVerifier().evaluate(record, cars_spectroscopy_proxy_spec(record))
    assert proxy.task_id == "cars_spectroscopy_fallback"
    assert proxy.output_exists is False


def test_two_manifests_share_same_local_runner_flow(tmp_path):
    compiler = TaskBundleCompiler(tmp_path / "compiled")
    tasks = [
        {
            "task_id": "task_a",
            "output_name": "result_a.npz",
            "token": "alpha",
        },
        {
            "task_id": "task_b",
            "output_name": "result_b.npz",
            "token": "beta",
        },
    ]

    for item in tasks:
        task_root = tmp_path / item["task_id"]
        (task_root / "data").mkdir(parents=True)
        (task_root / "evaluation").mkdir(parents=True)
        (task_root / "README.md").write_text(
            f"{item['task_id']} public task\n",
            encoding="utf-8",
        )
        (task_root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        (task_root / "data" / "raw_data.txt").write_text(item["token"], encoding="utf-8")
        manifest = {
            "task_id": item["task_id"],
            "family": "synthetic",
            "source_task_dir": str(task_root),
            "public_policy": {
                "readme_policy": {"preserve_user_eval_notes": True},
                "public_data_allowlist": ["data/raw_data.txt"],
                "public_data_denylist": [],
            },
            "runtime_layout": {
                "data_dir": "data",
                "work_dir": "work",
                "output_dir": "output",
                "checkpoints_dir": "checkpoints",
                "public_bundle_dir": "public_bundle",
            },
            "output_contract": {
                "required_outputs": [
                    {"path": f"output/{item['output_name']}", "format": "npz"}
                ]
            },
            "proxy_spec": {
                "primary_output": f"output/{item['output_name']}",
                "output_dtype": "npz",
            },
            "judge_spec": {"metrics": ["dummy_metric"], "all_metrics_required": True},
        }
        bundle = compiler.compile(
            task_root,
            task_id=item["task_id"],
            family="synthetic",
            public_policy=manifest["public_policy"],
            manifest=manifest,
        )
        (bundle.public_bundle_dir / "solver.py").write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "import numpy as np",
                    "token = Path('data/raw_data.txt').read_text(encoding='utf-8').strip()",
                    "Path('output').mkdir(parents=True, exist_ok=True)",
                    f"np.savez('output/{item['output_name']}', score=np.array(1.0), token=np.array(token))",
                    "print(token)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        record = LocalRunnerAdapter().run(
            bundle,
            ExecutorSessionConfig(
                run_id=f"run-{item['task_id']}",
                env_hash="env-1",
                workspace_root=tmp_path / f"workspace_{item['task_id']}",
                command=("python3", "solver.py"),
            ),
            [],
        )
        assert item["token"] in record.stdout
        proxy = ProxyVerifier().evaluate(record, manifest_proxy_spec(record, manifest))
        assert proxy.output_exists is True


def test_cars_spectroscopy_real_manifest_compile_and_agent_flow_passes(tmp_path, monkeypatch):
    manifest = load_real_cars_manifest()
    task_root = Path(manifest["source_task_dir"])
    compiler = TaskBundleCompiler(tmp_path / "compiled")
    bundle = compiler.compile(
        task_root,
        task_id=manifest["task_id"],
        family=manifest["family"],
        public_policy=manifest["public_policy"],
        manifest=manifest,
        audit_adapter=HeuristicCompileAuditAdapter(),
    )

    public_readme = bundle.readme_public_path.read_text(encoding="utf-8")
    assert "Method Hints" in public_readme
    assert "ground_truth.npz" not in public_readme
    assert not (bundle.public_bundle_dir / "src").exists()
    assert not (bundle.public_bundle_dir / "main.py").exists()
    assert not (bundle.public_bundle_dir / "notebooks").exists()
    assert not (bundle.public_bundle_dir / "plan").exists()
    assert not (bundle.public_bundle_dir / "data" / "ground_truth.npz").exists()
    assert not (bundle.public_bundle_dir / "evaluation" / "reference_outputs").exists()

    report = json.loads(bundle.compile_report_path.read_text(encoding="utf-8"))
    assert "main.py" in report["rule_blocked_paths"]
    assert "plan/approach.md" in report["rule_blocked_paths"]
    assert report["llm_audit_warnings"]
    assert report["llm_suggested_public_contract"]["required_outputs"][0]["path"] == "output/reconstruction.npz"
    assert report["final_public_contract"]["judge_metrics"] == manifest["judge_metrics"]
    assert report["final_public_contract"]["required_outputs"][0]["required_fields"] == [
        "estimated_temperature_K",
        "reconstructed_spectrum",
        "nu_axis",
    ]
    assert report["runtime_policy"]["model_timeout_seconds"] == 600
    assert report["runtime_policy"]["execution_budget_seconds"] == 240

    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="MYEVOSKILL_CLAUDE_API_KEY",
    )
    generated_code = {
        "files": {
            "work/main.py": "\n".join(
                [
                    "import os",
                    "from pathlib import Path",
                    "import numpy as np",
                    "public = Path(os.environ['MYEVOSKILL_PUBLIC_BUNDLE'])",
                    "raw = np.load('data/raw_data.npz')",
                    "assert raw['measurements'].shape[1] == 200",
                    "assert not (public / 'data' / 'ground_truth.npz').exists()",
                    "assert not (public / 'plan').exists()",
                    "Path('output').mkdir(parents=True, exist_ok=True)",
                    "np.savez(",
                    "    'output/reconstruction.npz',",
                    "    estimated_temperature_K=np.array([2400.0]),",
                    "    reconstructed_spectrum=raw['measurements'],",
                    "    nu_axis=raw['nu_axis'],",
                    ")",
                    "print('agent-done')",
                ]
            )
            + "\n",
            "work/src/__init__.py": "",
            "work/src/preprocessing.py": "def prepare():\n    return 'ok'\n",
            "work/src/physics_model.py": "def build_model():\n    return 'ok'\n",
            "work/src/solvers.py": "def solve():\n    return 'ok'\n",
            "work/src/visualization.py": "def summarize():\n    return 'ok'\n",
        },
        "declared_outputs": ["output/reconstruction.npz"],
        "assumptions": ["public data contains measurements and nu axis"],
        "solver_summary": "writes a contract-compliant reconstruction artifact",
        "files_written": ["work/main.py", "work/src/solvers.py"],
        "commands_run": ["python work/main.py", "python evaluation/self_eval.py"],
        "sdk_messages": [
            {"name": "Bash", "input": {"command": "python work/main.py"}},
            {"name": "Bash", "input": {"command": "python evaluation/self_eval.py"}},
        ],
    }
    monkeypatch.setenv("MYEVOSKILL_CLAUDE_API_KEY", "rotated-test-key")
    workspace = tmp_path / "run_real"
    session = ExecutorSessionConfig(
        run_id="run-real-pass",
        env_hash="env-real",
        workspace_root=workspace,
        model_config=model,
        provider_extras={"mock_claude_sdk_response": generated_code},
    )
    record = ClaudeWorkspaceAdapter().run(bundle, session, ["skill-alpha"])
    assert record.provider == "claude_workspace"
    assert record.metadata["api_key_env"] == "MYEVOSKILL_CLAUDE_API_KEY"
    assert record.metadata["model_provider_kind"] == "claude_sdk"
    assert record.metadata["response_format"] == "claude_sdk_workspace"
    assert record.metadata["sdk_backend"] == "claude_sdk"
    assert record.metadata["effective_model_timeout_seconds"] == 600
    assert record.metadata["effective_execution_budget_seconds"] == 240
    assert record.metadata["stop_oracle"] == "public_self_eval"
    assert record.metadata["public_self_eval_seen_in_trace"] is True
    assert record.metadata["public_self_eval_passed_post_run"] is True
    assert record.metadata["parsed_response"]["declared_outputs"] == ["output/reconstruction.npz"]
    assert "python work/main.py" in record.metadata["parsed_response"]["commands_run"]
    assert "rotated-test-key" not in record.stdout
    prompt_text = (workspace / "agent_prompt_round_1.txt").read_text(encoding="utf-8")
    assert "workspace_root=" in prompt_text
    assert "cwd=" in prompt_text
    assert "README_public.md is the authoritative task specification" in prompt_text
    assert "Use relative paths first" in prompt_text
    assert "Your job is to complete the workspace contract and stop cleanly" in prompt_text
    assert "evaluation/self_eval.py" in prompt_text
    assert (workspace / "trajectory_normalized.json").exists()
    assert (workspace / "public_self_eval_round_1.json").exists()
    assert (workspace / "public_self_eval_stdout_round_1.log").exists()
    assert (workspace / "public_self_eval_stderr_round_1.log").exists()

    proxy = ProxyVerifier().evaluate(record, cars_spectroscopy_proxy_spec(record))
    assert proxy.output_exists is True

    judge = evaluate_cars_spectroscopy_run(task_root, record)
    assert judge.all_metrics_passed is True

    logger = RunLogger(tmp_path / "logs")
    log_dir = logger.create_run_dir(record.run_id)
    logger.write_json_artifact(
        log_dir,
        "executor_config.json",
        {
            "provider": record.provider,
            "model_provider_kind": record.metadata["model_provider_kind"],
            "api_key_env": record.metadata["api_key_env"],
        },
    )
    logger.write_json_artifact(
        log_dir,
        "model_config.json",
        {
            "provider_name": model.provider_name,
            "model_name": model.model_name,
            "api_key_env": model.api_key_env,
        },
    )
    logger.append_text_log(log_dir, "stdout.log", record.stdout)
    logger.append_text_log(log_dir, "stderr.log", record.stderr)
    logger.append_text_log(log_dir, "raw_response.txt", record.metadata["raw_response_text"])
    logger.write_json_artifact(log_dir, "parsed_response.json", record.metadata["parsed_response"])
    logger.write_json_artifact(log_dir, "proxy_feedback.json", proxy)
    logger.write_json_artifact(log_dir, "judge_result.json", judge)

    validator = TransferValidator()
    validation = validator.evaluate(
        baseline_outcomes={
            "cars_spectroscopy": TaskOutcome(task_id="cars_spectroscopy", all_metrics_passed=False),
        },
        treatment_outcomes={
            "cars_spectroscopy": TaskOutcome(
                task_id="cars_spectroscopy",
                all_metrics_passed=judge.all_metrics_passed,
                metrics_actual=judge.metrics_actual,
                failed_metrics=judge.failed_metrics,
                failure_tags=judge.failure_tags,
            ),
        },
    )
    entry = SkillRegistry(tmp_path / "registry_real").register(
        SkillCandidate(
            skill_id="skill-real-pass",
            version="0.1.0",
            description="real manifest pass",
            source_run_ids=[record.run_id],
            legal_source=True,
            reusable=True,
        ),
        validation,
    )
    assert entry.status == "validated"
    model_log = (log_dir / "model_config.json").read_text(encoding="utf-8")
    assert "rotated-test-key" not in model_log
    assert "MYEVOSKILL_CLAUDE_API_KEY" in model_log


def test_cars_spectroscopy_real_manifest_agent_flow_fails_when_any_metric_fails(
    tmp_path, monkeypatch
):
    manifest = load_real_cars_manifest()
    task_root = Path(manifest["source_task_dir"])
    compiler = TaskBundleCompiler(tmp_path / "compiled")
    bundle = compiler.compile(
        task_root,
        task_id=manifest["task_id"],
        family=manifest["family"],
        public_policy=manifest["public_policy"],
        manifest=manifest,
        audit_adapter=HeuristicCompileAuditAdapter(),
    )
    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="MYEVOSKILL_CLAUDE_API_KEY",
    )
    generated_code = {
        "files": {
            "work/main.py": "\n".join(
                [
                    "import os",
                    "from pathlib import Path",
                    "import numpy as np",
                    "workspace = Path(os.environ['MYEVOSKILL_WORKSPACE'])",
                    "raw = np.load('data/raw_data.npz')",
                    "(workspace / 'output').mkdir(parents=True, exist_ok=True)",
                    "np.savez(",
                    "    workspace / 'output' / 'reconstruction.npz',",
                    "    estimated_temperature_K=np.array([2000.0]),",
                    "    reconstructed_spectrum=np.zeros_like(raw['measurements']),",
                    "    nu_axis=raw['nu_axis'],",
                    ")",
                    "print('agent-bad-metric')",
                ]
            )
            + "\n",
            "work/src/__init__.py": "",
            "work/src/preprocessing.py": "",
            "work/src/physics_model.py": "",
            "work/src/solvers.py": "",
            "work/src/visualization.py": "",
        },
        "declared_outputs": ["output/reconstruction.npz"],
        "solver_summary": "intentionally fails one metric",
        "assumptions": [],
        "files_written": ["work/main.py"],
        "commands_run": ["python work/main.py"],
    }
    monkeypatch.setenv("MYEVOSKILL_CLAUDE_API_KEY", "rotated-test-key")
    session = ExecutorSessionConfig(
        run_id="run-real-fail",
        env_hash="env-real",
        workspace_root=tmp_path / "run_real_fail",
        model_config=model,
        provider_extras={"mock_claude_sdk_response": generated_code},
    )
    record = ClaudeWorkspaceAdapter().run(bundle, session, [])
    proxy = ProxyVerifier().evaluate(record, cars_spectroscopy_proxy_spec(record))
    assert proxy.output_exists is True
    assert record.metadata["response_format"] == "claude_sdk_workspace"
    judge = evaluate_cars_spectroscopy_run(task_root, record)
    assert judge.all_metrics_passed is False
    assert "nrmse_vs_ref" in judge.failed_metrics


@pytest.mark.external_network
def test_cars_spectroscopy_real_manifest_live_external_network_flow(tmp_path):
    if importlib.util.find_spec("claude_agent_sdk") is None:
        pytest.skip("claude_agent_sdk is required for the live Claude SDK smoke test")
    live_model = ModelConfig(
        provider_name="claude-sdk",
        model_name="",
        api_key_env="MYEVOSKILL_CLAUDE_API_KEY",
        temperature=0.0,
    )
    try:
        ClaudeSDKAdapter(live_model).resolve_api_key()
    except RuntimeError:
        pytest.skip(
            "A Claude API key is required for the live Claude SDK smoke test "
            "(MYEVOSKILL_CLAUDE_API_KEY, CLAUDE_API_KEY, or ANTHROPIC_API_KEY)"
        )

    manifest = load_real_cars_manifest()
    task_root = Path(manifest["source_task_dir"])
    compiler = TaskBundleCompiler(tmp_path / "compiled")
    bundle = compiler.compile(
        task_root,
        task_id=manifest["task_id"],
        family=manifest["family"],
        public_policy=manifest["public_policy"],
        manifest=manifest,
        audit_adapter=HeuristicCompileAuditAdapter(),
    )

    public_readme = bundle.readme_public_path.read_text(encoding="utf-8")
    assert "Method Hints" in public_readme
    assert "ground_truth.npz" not in public_readme
    assert not (bundle.public_bundle_dir / "src").exists()
    assert not (bundle.public_bundle_dir / "main.py").exists()
    assert not (bundle.public_bundle_dir / "notebooks").exists()
    assert not (bundle.public_bundle_dir / "plan").exists()
    assert not (bundle.public_bundle_dir / "data" / "ground_truth.npz").exists()
    assert not (bundle.public_bundle_dir / "evaluation" / "reference_outputs").exists()

    model = live_model
    run_paths = resolve_run_paths(PROJECT_ROOT, manifest["task_id"], "run-real-live")
    ensure_clean_run_directory(run_paths.workspace_root)
    ensure_clean_run_directory(run_paths.log_root)
    session = ExecutorSessionConfig(
        run_id="run-real-live",
        env_hash="env-real-live",
        workspace_root=run_paths.workspace_root,
        model_config=model,
        provider_extras={
            "repo_root": str(PROJECT_ROOT),
            "workspace_prompt_mode": "semantic_only",
            "workspace_completion_policy": "main_success_output_contract",
        },
    )
    record = ClaudeWorkspaceAdapter().run(bundle, session, ["skill-alpha"])

    proxy = ProxyVerifier().evaluate(record, cars_spectroscopy_proxy_spec(record))
    judge = evaluate_cars_spectroscopy_run(task_root, record)
    validator = TransferValidator()
    validation = validator.evaluate(
        baseline_outcomes={
            "cars_spectroscopy": TaskOutcome(task_id="cars_spectroscopy", all_metrics_passed=False),
        },
        treatment_outcomes={
            "cars_spectroscopy": TaskOutcome(
                task_id="cars_spectroscopy",
                all_metrics_passed=judge.all_metrics_passed,
                metrics_actual=judge.metrics_actual,
                failed_metrics=judge.failed_metrics,
                failure_tags=judge.failure_tags,
            ),
        },
    )
    log_dir = _write_live_run_logs(run_paths.log_root, record, proxy, judge, validation)

    assert record.metadata["bridge_mode"] == "workspace_claude_sdk", (
        f"unexpected bridge mode; see logs in {log_dir}"
    )
    assert record.metadata["protocol_status"] == "completed", (
        f"protocol failure {record.metadata.get('protocol_failure_reason')}; see logs in {log_dir}"
    )
    assert record.metadata["sdk_completion_source"] in {
        "result_message",
        "external_output_contract",
    }
    assert record.metadata["api_key_env"] == "MYEVOSKILL_CLAUDE_API_KEY"
    assert record.metadata["effective_model_timeout_seconds"] == 600
    assert record.metadata["effective_execution_budget_seconds"] == 240
    assert record.metadata["workspace_prompt_mode"] == "semantic_only"
    assert record.metadata["workspace_completion_policy"] == "main_success_output_contract"
    assert Path(record.workspace_root) == run_paths.workspace_root
    assert "Bearer " not in (log_dir / "model_config.json").read_text(encoding="utf-8")

    assert isinstance(judge.metrics_actual, dict), f"judge did not execute; see logs in {log_dir}"
    assert (run_paths.workspace_root / "trajectory_normalized.json").exists(), (
        f"trajectory_normalized.json missing; see logs in {log_dir}"
    )
    assert (run_paths.workspace_root / "vendor_session_ref.json").exists(), (
        f"vendor_session_ref.json missing; see logs in {log_dir}"
    )
    assert (run_paths.workspace_root / "public_self_eval_round_1.json").exists(), (
        f"public_self_eval_round_1.json missing; see logs in {log_dir}"
    )
    assert (run_paths.workspace_root / "public_self_eval_stdout_round_1.log").exists(), (
        f"public_self_eval_stdout_round_1.log missing; see logs in {log_dir}"
    )
    assert (run_paths.workspace_root / "public_self_eval_stderr_round_1.log").exists(), (
        f"public_self_eval_stderr_round_1.log missing; see logs in {log_dir}"
    )
    assert (run_paths.workspace_root / "evaluation" / "self_eval.py").exists(), (
        f"evaluation/self_eval.py missing; see logs in {log_dir}"
    )
    assert record.metadata["protocol_failure_reason"] == "", (
        f"unexpected protocol failure reason {record.metadata.get('protocol_failure_reason')}; see logs in {log_dir}"
    )

    entry = SkillRegistry(tmp_path / "registry_live").register(
        SkillCandidate(
            skill_id="skill-real-live",
            version="0.1.0",
            description="live external-network manifest smoke",
            source_run_ids=[record.run_id],
            legal_source=True,
            reusable=True,
        ),
        validation,
    )
    assert entry.status in {"validated", "rejected"}, (
        f"validation did not complete cleanly; see logs in {log_dir}"
    )

