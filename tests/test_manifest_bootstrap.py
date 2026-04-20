from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from myevoskill import (
    BootstrapResult,
    TaskBundleCompiler,
    bootstrap_task,
    load_task_bootstrap_notes,
)
from myevoskill.manifest_bootstrap import main as bootstrap_main
from myevoskill.registration_agent import (
    build_registration_agent_prompt,
    build_registration_agent_system_prompt,
    normalize_registration_input,
    validate_registration_input,
)
import myevoskill.registration_contract as registration_contract_mod
from myevoskill.task_registration import (
    contract_draft_main,
    draft_task_contract,
    load_task_registration_notes,
    main as register_main,
    register_task,
)
from myevoskill.models import ContractDraftResult, EnvCacheRecord, RunRecord, TaskRegistrationResult

MYEVOSKILL_ROOT = Path(__file__).resolve().parents[1]
AUTOSKILL_ROOT = MYEVOSKILL_ROOT.parent
TASKS_ROOT = AUTOSKILL_ROOT / "tasks"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _stub_registered_task_runtime_env(monkeypatch, tmp_path):
    def _fake_ensure_task_runtime_env(task_root, *, output_root, task_id, family):
        requirements_path = Path(task_root) / "requirements.txt"
        if not requirements_path.exists():
            raise FileNotFoundError(f"requirements.txt not found: {requirements_path}")
        env_root = tmp_path / "env_cache" / task_id
        env_root.mkdir(parents=True, exist_ok=True)
        install_report_path = env_root / "install_report.json"
        build_log_path = env_root / "build.log"
        install_report_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "env_hash": f"env-{task_id}",
                    "python_executable": str(Path(sys.executable).resolve()),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        build_log_path.write_text("stubbed task runtime env\n", encoding="utf-8")
        return EnvCacheRecord(
            env_hash=f"env-{task_id}",
            env_dir=env_root,
            python_executable=Path(sys.executable).resolve(),
            install_report_path=install_report_path,
            build_log_path=build_log_path,
            ready=True,
            base_image_cache_dir=env_root / "base",
            task_env_cache_dir=env_root,
            dataset_cache_dir=env_root / "datasets",
            artifact_cache_dir=env_root / "artifacts",
            checkpoint_cache_dir=env_root / "checkpoints",
        )

    monkeypatch.setattr(
        registration_contract_mod,
        "ensure_task_runtime_env",
        _fake_ensure_task_runtime_env,
    )


def _write_minimal_task(task_root: Path, *, include_metrics: bool = True) -> Path:
    (task_root / "data").mkdir(parents=True, exist_ok=True)
    (task_root / "evaluation").mkdir(parents=True, exist_ok=True)
    (task_root / "src").mkdir(parents=True, exist_ok=True)

    (task_root / "README.md").write_text(
        "\n".join(
            [
                "# Minimal Reconstruction Task",
                "",
                "> Domain: optics | Difficulty: Easy",
                "",
                "## Problem Description",
                "Recover a small array from synthetic data.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (task_root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (task_root / "main.py").write_text(
        "from pathlib import Path\n\n"
        "def main():\n"
        "    Path('output').mkdir(exist_ok=True)\n"
        "    print('ok')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    (task_root / "src" / "__init__.py").write_text("", encoding="utf-8")
    np.savez(task_root / "data" / "raw_data.npz", signal=np.arange(4, dtype=float))
    (task_root / "data" / "meta_data.json").write_text(
        json.dumps({"grid_size": 4}, indent=2),
        encoding="utf-8",
    )
    if include_metrics:
        (task_root / "evaluation" / "metrics.json").write_text(
            json.dumps({"score_boundary": 0.9}, indent=2),
            encoding="utf-8",
        )
    return task_root


def _write_registration_input(task_root: Path) -> Path:
    payload = {
        "task_id": task_root.name,
        "family": "optics",
        "task_description_resources": [
            {
                "path": "README.md",
                "authority": "authoritative",
                "notes": "Task description and public constraints.",
            }
        ],
        "public_input_resources": [
            {
                "path": "data/raw_data.npz",
                "authority": "authoritative",
                "notes": "Observed public input data.",
            }
        ],
        "public_metadata_resources": [
            {
                "path": "data/meta_data.json",
                "authority": "authoritative",
                "notes": "Physical parameters and metadata.",
            }
        ],
        "public_eval_resources": [],
        "evaluation_logic_resources": [
            {
                "path": "main.py",
                "authority": "supplementary",
                "notes": "Reference implementation and metric context.",
            }
        ],
        "hidden_reference_resources": [],
        "hidden_metric_config_resources": [
            {
                "path": "evaluation/metrics.json",
                "authority": "authoritative",
                "notes": "Hidden threshold configuration.",
            }
        ],
        "pass_metrics": [
            {
                "name": "score",
                "description": "Cosine similarity between the reconstructed signal and reference signal.",
                "kind": "standard",
                "operator": ">=",
                "threshold": 0.9,
            }
        ],
        "execution_hints": {
            "read_first": ["README.md", "main.py"],
            "suggested_entrypoint": "main.py",
            "suggested_output_path": "output/reconstruction.npz",
        },
    }
    path = task_root / "evaluation" / "registration_input.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _mock_registration_agent_result(task_root: Path) -> dict:
    return {
        "summary": {
            "registration_contract_draft": {
                "task_id": task_root.name,
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
                        "path": "data/raw_data.npz",
                        "role": "public_input_data",
                        "visibility": "public",
                        "semantics": "Public observation data available to the execution agent.",
                        "authority": "authoritative",
                    },
                    {
                        "path": "data/meta_data.json",
                        "role": "public_metadata",
                        "visibility": "public",
                        "semantics": "Public physical parameters and experiment configuration.",
                        "authority": "authoritative",
                    },
                    {
                        "path": "evaluation/metrics.json",
                        "role": "hidden_metric_config",
                        "visibility": "private",
                        "semantics": "Hidden metric thresholds for the judge.",
                        "authority": "authoritative",
                    },
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
                            "description": "Cosine similarity between output signal and hidden reference signal.",
                            "mode": "ncc",
                            "output_field": "signal",
                            "reference_resource_path": "data/raw_data.npz",
                            "reference_field": "signal",
                            "pass_condition": {"operator": ">=", "threshold": 0.9},
                            "source_hint": "main.py",
                        }
                    ]
                },
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
            "contract_generation_notes": {
                "agent_summary": "Validated the declared resources and mapped them into the contract schema.",
                "warnings": [],
                "open_questions": [],
                "declared_inputs_used": ["README.md", "main.py", "data/raw_data.npz"],
                "resource_validation": [
                    {"path": "README.md", "status": "read"},
                    {"path": "main.py", "status": "read"},
                ],
            },
            "judge_recommendations": {
                "metrics": [
                    {
                        "name": "score",
                        "kind": "standard",
                        "source_path": "main.py",
                        "callable": "",
                        "result_key": "",
                        "rationale": "main.py demonstrates the reference evaluation flow.",
                    }
                ],
                "generation_hints": ["Prefer the raw_data signal as the hidden comparison target."],
            },
        },
        "completion_source": "external_registration_schema",
        "vendor_session_ref": {"sdk_backend": "claude_sdk", "session_id": "test-session"},
    }


def _write_confirmed_minimal_contract(task_root: Path) -> Path:
    evaluation_dir = task_root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "task_id": task_root.name,
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
                "path": "data/raw_data.npz",
                "role": "public_input_data",
                "visibility": "public",
                "semantics": "Public observation data available to the execution agent.",
                "authority": "authoritative",
            },
            {
                "path": "data/meta_data.json",
                "role": "public_metadata",
                "visibility": "public",
                "semantics": "Public physical parameters, constants, and experiment configuration.",
                "authority": "authoritative",
            },
            {
                "path": "evaluation/metrics.json",
                "role": "hidden_metric_config",
                "visibility": "private",
                "semantics": "Metric threshold configuration used by the hidden judge.",
                "authority": "authoritative",
            },
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
                    "description": "Cosine similarity between output signal and hidden reference signal.",
                    "mode": "ncc",
                    "output_field": "signal",
                    "reference_resource_path": "data/raw_data.npz",
                    "reference_field": "signal",
                    "pass_condition": {"operator": ">=", "threshold": 0.9},
                    "source_hint": "main.py",
                }
            ]
        },
        "execution_conventions": {
            "read_first": ["README_public.md"],
            "readable_paths": ["README_public.md", "data/raw_data.npz", "data/meta_data.json", "requirements.txt"],
            "writable_paths": ["work", "output", "checkpoints"],
            "entrypoint": "work/main.py",
        },
    }
    confirmed_path = evaluation_dir / "registration_contract.json"
    confirmed_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return confirmed_path


def _write_cars_like_task(task_root: Path) -> Path:
    (task_root / "data").mkdir(parents=True, exist_ok=True)
    (task_root / "evaluation" / "reference_outputs").mkdir(parents=True, exist_ok=True)
    (task_root / "src").mkdir(parents=True, exist_ok=True)

    shutil.copy2(TASKS_ROOT / "cars_spectroscopy" / "README.md", task_root / "README.md")
    shutil.copy2(TASKS_ROOT / "cars_spectroscopy" / "main.py", task_root / "main.py")
    shutil.copy2(
        TASKS_ROOT / "cars_spectroscopy" / "requirements.txt",
        task_root / "requirements.txt",
    )
    shutil.copy2(
        TASKS_ROOT / "cars_spectroscopy" / "evaluation" / "metrics.json",
        task_root / "evaluation" / "metrics.json",
    )
    np.savez(
        task_root / "data" / "raw_data.npz",
        measurements=np.ones((1, 8), dtype=float),
        nu_axis=np.linspace(2300.0, 2400.0, 8, dtype=float)[None, :],
    )
    np.savez(
        task_root / "data" / "ground_truth.npz",
        spectrum=np.ones((1, 8), dtype=float),
        temperature=np.array([1800.0], dtype=float),
        x_mol=np.array([0.79], dtype=float),
    )
    (task_root / "data" / "meta_data.json").write_text(
        json.dumps({"pressure_bar": 10.0}, indent=2),
        encoding="utf-8",
    )
    np.savez(
        task_root / "evaluation" / "reference_outputs" / "reconstruction.npz",
        y_pred=np.ones((1, 8), dtype=float),
    )
    return task_root


def _write_conventional_ptychography_like_task(task_root: Path) -> Path:
    (task_root / "data").mkdir(parents=True, exist_ok=True)
    (task_root / "evaluation" / "reference_outputs").mkdir(parents=True, exist_ok=True)
    (task_root / "src").mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        TASKS_ROOT / "conventional_ptychography" / "README.md",
        task_root / "README.md",
    )
    shutil.copy2(
        TASKS_ROOT / "conventional_ptychography" / "main.py",
        task_root / "main.py",
    )
    shutil.copy2(
        TASKS_ROOT / "conventional_ptychography" / "requirements.txt",
        task_root / "requirements.txt",
    )
    shutil.copy2(
        TASKS_ROOT / "conventional_ptychography" / "evaluation" / "metrics.json",
        task_root / "evaluation" / "metrics.json",
    )
    np.savez(
        task_root / "data" / "raw_data.npz",
        ptychogram=np.ones((3, 4, 4), dtype=np.float32),
        encoder=np.zeros((3, 2), dtype=np.float32),
    )
    np.savez(
        task_root / "data" / "ground_truth.npz",
        object=np.ones((4, 4), dtype=np.complex64),
    )
    (task_root / "data" / "meta_data.json").write_text(
        json.dumps({"wavelength_m": 6.328e-7, "No": 4}, indent=2),
        encoding="utf-8",
    )
    (task_root / "data" / "simu.hdf5").write_bytes(b"placeholder")
    (task_root / "evaluation" / "reference_outputs" / "recon.hdf5").write_bytes(
        b"placeholder"
    )
    return task_root


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_minimal_task_generates_manifest_stub_and_notes(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "minimal_task")

    result = bootstrap_task(task_root, output_root=project_root)

    assert isinstance(result, BootstrapResult)
    assert result.manifest_path.exists()
    assert result.judge_stub_path.exists()
    assert result.notes_path.exists()

    manifest = _load_json(result.manifest_path)
    notes = load_task_bootstrap_notes("minimal_task", output_root=project_root)

    assert manifest["task_id"] == "minimal_task"
    assert manifest["family"] == "optics"
    assert manifest["output_contract"]["required_outputs"][0]["path"] == "output/reconstruction.npz"
    assert manifest["judge_spec"]["adapter_path"] == "evaluation/judge_adapter.py"
    assert notes["manifest_path"] == str(result.manifest_path)

    bundle = TaskBundleCompiler(project_root / "compiled").compile(
        source_task_dir=task_root,
        task_id=manifest["task_id"],
        family=manifest["family"],
        public_policy=manifest["public_policy"],
        manifest=manifest,
    )
    task_spec = _load_json(bundle.task_spec_path)
    assert task_spec["judge_spec"]["adapter_path"] == "evaluation/judge_adapter.py"


def test_generated_judge_stub_never_succeeds_without_hidden_metric_impl(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(
        project_root / "tasks" / "judge_stub_task",
        include_metrics=False,
    )

    result = bootstrap_task(task_root, output_root=project_root)
    manifest = _load_json(result.manifest_path)
    workspace_root = project_root / "runs" / "judge_stub_task"
    (workspace_root / "output").mkdir(parents=True, exist_ok=True)
    np.savez(
        workspace_root / "output" / "reconstruction.npz",
        TODO_field_name=np.ones((2, 2), dtype=float),
    )

    module = _load_module(result.judge_stub_path, "generated_judge_stub")
    judge_result = module.evaluate_run(
        task_root,
        RunRecord(
            run_id="run-1",
            task_id="judge_stub_task",
            provider="local",
            env_hash="env-1",
            skills_active=(),
            workspace_root=workspace_root,
        ),
        manifest,
    )

    assert not judge_result.all_metrics_passed
    assert "todo_hidden_metric" in judge_result.failed_metrics
    assert "judge_not_implemented" in judge_result.failure_tags


def test_bootstrap_cars_respects_existing_manifest_and_extracts_cars_fields(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_cars_like_task(project_root / "tasks" / "cars_spectroscopy")
    registry_root = project_root / "registry" / "tasks"
    registry_root.mkdir(parents=True, exist_ok=True)

    source_manifest_path = MYEVOSKILL_ROOT / "registry" / "tasks" / "cars_spectroscopy.json"
    original_manifest = _load_json(source_manifest_path)
    copied_manifest_path = registry_root / "cars_spectroscopy.json"
    copied_manifest_path.write_text(
        json.dumps(original_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = bootstrap_task(task_root, output_root=project_root)
    notes = load_task_bootstrap_notes("cars_spectroscopy", output_root=project_root)

    assert _load_json(copied_manifest_path) == original_manifest
    assert notes["output_detection"]["output_path"] == "output/reconstruction.npz"
    assert notes["output_detection"]["candidate_output_fields"] == [
        "estimated_temperature_K",
        "reconstructed_spectrum",
        "nu_axis",
    ]
    assert result.missing_items == []
    assert any("manifest already exists" in warning for warning in result.warnings)


def test_bootstrap_conventional_ptychography_recommends_npz_contract_and_hidden_paths(
    tmp_path,
):
    project_root = tmp_path / "project"
    task_root = _write_conventional_ptychography_like_task(
        project_root / "tasks" / "conventional_ptychography"
    )

    result = bootstrap_task(task_root, output_root=project_root)
    manifest = _load_json(result.manifest_path)
    notes = load_task_bootstrap_notes("conventional_ptychography", output_root=project_root)
    judge_stub_text = result.judge_stub_path.read_text(encoding="utf-8")

    required_output = manifest["output_contract"]["required_outputs"][0]
    assert required_output["path"] == "output/reconstruction.npz"
    assert required_output["format"] == "npz"
    assert required_output["required_fields"] == ["object"]
    assert manifest["judge_spec"]["metrics"] == ["phase_ncc", "phase_nrmse"]
    assert "metric_names = list(judge_spec.get('metrics') or ['phase_ncc', 'phase_nrmse'])" in judge_stub_text
    assert notes["output_detection"]["needs_public_npz_export"] is True
    assert "data/ground_truth.npz" in notes["recommended_hidden_paths"]
    assert "evaluation/reference_outputs/recon.hdf5" in notes["recommended_hidden_paths"]
    assert result.missing_items == []
    assert any("public npz export" in warning for warning in notes["warnings"])


def test_bootstrap_use_llm_writes_suggestion_file_without_overwriting_manifest(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "llm_task")
    registry_root = project_root / "registry" / "tasks"
    registry_root.mkdir(parents=True, exist_ok=True)

    custom_manifest = {
        "task_id": "llm_task",
        "family": "kept_family",
        "source_task_dir": "tasks/llm_task",
    }
    manifest_path = registry_root / "llm_task.json"
    manifest_path.write_text(
        json.dumps(custom_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = bootstrap_task(task_root, output_root=project_root, use_llm=True)
    suggested_path = registry_root / "llm_task.suggested.json"
    notes = load_task_bootstrap_notes("llm_task", output_root=project_root)

    assert _load_json(manifest_path) == custom_manifest
    assert suggested_path.exists()
    assert _load_json(suggested_path)["mode"] == "heuristic_fallback"
    assert notes["llm_suggestions"]["task_id"] == "llm_task"
    assert any("use_llm requested" in warning for warning in result.warnings)


def test_manifest_bootstrap_cli_writes_files_and_reports_result(tmp_path, capsys):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "cli_task")

    exit_code = bootstrap_main(
        [
            "--task-root",
            str(task_root),
            "--output-root",
            str(project_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"task_id": "cli_task"' in captured.out
    assert (project_root / "registry" / "tasks" / "cli_task.json").exists()


def test_task_contract_draft_cli_and_api_generate_draft_files(tmp_path, capsys):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "register_task")
    registration_input_path = _write_registration_input(task_root)

    def _fake_run_registration_agent(task_root_arg, **kwargs):
        assert kwargs["registration_input_path"] == registration_input_path.resolve()
        return _mock_registration_agent_result(task_root_arg)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        registration_contract_mod,
        "run_registration_agent",
        _fake_run_registration_agent,
    )
    try:
        result = draft_task_contract(
            task_root,
            output_root=project_root,
            registration_input_path=registration_input_path,
        )
        notes = load_task_registration_notes("register_task", output_root=project_root)
        exit_code = contract_draft_main(
            [
                "--task-root",
                str(task_root),
                "--output-root",
                str(project_root),
                "--registration-input",
                str(registration_input_path),
            ]
        )
    finally:
        monkeypatch.undo()

    captured = capsys.readouterr()
    assert isinstance(result, ContractDraftResult)
    assert notes["task_id"] == "register_task"
    assert Path(notes["draft_path"]).exists()
    assert notes["registration_input_path"] == str(registration_input_path.resolve())
    assert notes["completion_source"] == "external_registration_schema"
    assert notes["warnings"] == []
    assert notes["attempt_count"] == 1
    assert result.attempt_count == 1
    assert exit_code == 0
    assert '"task_id": "register_task"' in captured.out


def test_task_contract_draft_retries_once_with_repair_feedback(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "repair_task")
    registration_input_path = _write_registration_input(task_root)
    calls: list[dict] = []

    def _fake_run_registration_agent(task_root_arg, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            result = _mock_registration_agent_result(task_root_arg)
            metric = result["summary"]["registration_contract_draft"]["judge_contract"]["metrics"][0]
            metric.pop("pass_condition", None)
            metric["threshold_key"] = "score_boundary"
            return result
        assert kwargs["repair_feedback"] is not None
        assert "validation_errors" in kwargs["repair_feedback"]["previous_attempt"]
        return _mock_registration_agent_result(task_root_arg)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        registration_contract_mod,
        "run_registration_agent",
        _fake_run_registration_agent,
    )
    try:
        result = draft_task_contract(
            task_root,
            output_root=project_root,
            registration_input_path=registration_input_path,
        )
    finally:
        monkeypatch.undo()

    notes = load_task_registration_notes("repair_task", output_root=project_root)
    assert result.attempt_count == 2
    assert len(result.attempt_summaries) == 2
    assert notes["attempt_count"] == 2
    assert notes["attempt_summaries"][0]["status"] == "invalid_contract"
    assert notes["attempt_summaries"][1]["status"] == "succeeded"


def test_task_register_cli_requires_confirmed_contract_and_generates_manifest(tmp_path, capsys):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "confirmed_task")
    _write_confirmed_minimal_contract(task_root)

    result = register_task(task_root, output_root=project_root)
    manifest = _load_json(result.manifest_path)
    notes = load_task_registration_notes("confirmed_task", output_root=project_root)
    exit_code = register_main(
        [
            "--task-root",
            str(task_root),
            "--output-root",
            str(project_root),
        ]
    )

    captured = capsys.readouterr()
    assert isinstance(result, TaskRegistrationResult)
    assert result.manifest_path.exists()
    assert result.judge_path.exists()
    assert manifest["runtime_env"]["ready"] is True
    assert manifest["runtime_env"]["python_executable"] == str(Path(sys.executable).resolve())
    assert notes["judge_generation"]["status"] == "ready"
    assert notes["runtime_env"]["status"] == "ready"
    assert exit_code == 0
    assert '"task_id": "confirmed_task"' in captured.out


def test_task_contract_draft_requires_registration_input(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "missing_input")

    with pytest.raises(FileNotFoundError, match="registration input not found"):
        draft_task_contract(task_root, output_root=project_root)


def test_register_task_requires_requirements_txt(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "missing_requirements")
    (task_root / "requirements.txt").unlink()
    _write_confirmed_minimal_contract(task_root)

    with pytest.raises(FileNotFoundError, match="requirements.txt not found"):
        register_task(task_root, output_root=project_root)


def test_registration_input_validation_and_prompt_are_user_driven(tmp_path):
    task_root = _write_minimal_task(tmp_path / "task")
    registration_input = _load_json(_write_registration_input(task_root))
    normalized = normalize_registration_input(registration_input)
    system_prompt = build_registration_agent_system_prompt(registration_input)
    prompt = build_registration_agent_prompt(
        task_root,
        registration_input,
        registration_input_path=task_root / "evaluation" / "registration_input.json",
    )

    assert validate_registration_input(registration_input) == []
    assert normalized["task_description_resources"][0]["path"] == "README.md"
    assert "User-declared registration input" in system_prompt["append"]
    assert "Do not restate README or source file contents verbatim." in system_prompt["append"]
    assert "network" not in system_prompt["append"].lower()
    assert "Recover a small array from synthetic data." not in system_prompt["append"]
    assert "Inspect the task directory and return ONLY a JSON object" in prompt
    assert "Do not use threshold_key" in prompt


def test_register_task_rejects_missing_script_metric_source_path_file(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "broken_paths")
    confirmed_path = _write_confirmed_minimal_contract(task_root)
    contract = _load_json(confirmed_path)
    contract["judge_contract"]["metrics"] = [
        {
            "name": "script_score",
            "kind": "script",
            "description": "Score returned by a helper function.",
            "source_path": "src/missing_metric_helper.py",
            "callable": "compute_score",
            "result_key": "score",
            "inputs": {
                "signal": {
                    "resource_path": "data/raw_data.npz",
                    "field": "signal",
                }
            },
            "pass_condition": {"operator": ">=", "threshold": 0.9},
        }
    ]
    confirmed_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="task-local path validation failed"):
        register_task(task_root, output_root=project_root)


def test_register_task_accepts_contract_metrics_with_pass_condition(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "pass_condition_task")
    confirmed_path = _write_confirmed_minimal_contract(task_root)

    result = register_task(task_root, output_root=project_root)
    manifest = _load_json(result.manifest_path)

    assert result.manifest_path.exists()
    assert manifest["judge_spec"]["ready"] is True


def test_register_task_rejects_threshold_key_in_confirmed_contract(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "threshold_key_task")
    confirmed_path = _write_confirmed_minimal_contract(task_root)
    contract = _load_json(confirmed_path)
    metric = contract["judge_contract"]["metrics"][0]
    metric.pop("pass_condition", None)
    metric["threshold_key"] = "score_boundary"
    confirmed_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="threshold_key is not supported"):
        register_task(task_root, output_root=project_root)


def test_generated_judge_reports_invalid_output_shape_before_metric_runtime_error(tmp_path):
    project_root = tmp_path / "project"
    task_root = _write_minimal_task(project_root / "tasks" / "shape_guard_task")
    confirmed_path = _write_confirmed_minimal_contract(task_root)
    contract = _load_json(confirmed_path)
    contract["output_contract"]["required_fields"] = ["signal", "axis"]
    contract["output_contract"]["numeric_fields"] = ["signal", "axis"]
    contract["output_contract"]["same_shape_fields"] = ["signal", "axis"]
    confirmed_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    result = register_task(task_root, output_root=project_root)
    manifest = _load_json(result.manifest_path)
    workspace_root = project_root / "runs" / "shape_guard_task"
    (workspace_root / "output").mkdir(parents=True, exist_ok=True)
    np.savez(
        workspace_root / "output" / "reconstruction.npz",
        signal=np.ones((1, 4), dtype=float),
        axis=np.ones((4,), dtype=float),
    )

    module = _load_module(result.judge_path, "shape_guard_judge")
    judge_result = module.evaluate_run(
        task_root,
        RunRecord(
            run_id="run-shape",
            task_id="shape_guard_task",
            provider="local",
            env_hash="env-1",
            skills_active=(),
            workspace_root=workspace_root,
        ),
        manifest,
    )

    assert not judge_result.all_metrics_passed
    assert "invalid_output_shape" in judge_result.failure_tags
