import json
from pathlib import Path

import numpy as np

from myevoskill.models import RunRecord
from myevoskill.task_adapters import (
    cars_spectroscopy_proxy_spec,
    cars_spectroscopy_public_policy,
    evaluate_cars_spectroscopy_run,
    manifest_output_contract_path,
    manifest_primary_output_path,
    manifest_proxy_spec,
)


def _write_cars_hidden_inputs(task_root: Path) -> None:
    (task_root / "data").mkdir(parents=True)
    (task_root / "evaluation").mkdir(parents=True)
    np.savez(
        task_root / "data" / "raw_data.npz",
        measurements=np.array([[0.1, 0.5, 0.9]], dtype=float),
        nu_axis=np.array([[2280.0, 2290.0, 2300.0]], dtype=float),
    )
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


def test_cars_spectroscopy_public_policy_preserves_eval_notes():
    policy = cars_spectroscopy_public_policy()
    assert policy.readme_policy.preserve_user_eval_notes is True
    assert "data/raw_data.npz" in policy.public_data_allowlist


def test_evaluate_cars_spectroscopy_run_reads_real_metric_thresholds(tmp_path):
    task_root = tmp_path / "cars"
    _write_cars_hidden_inputs(task_root)

    run_root = tmp_path / "run"
    (run_root / "output").mkdir(parents=True)
    np.savez(
        run_root / "output" / "reconstruction.npz",
        estimated_temperature_K=np.array([2405.0]),
        reconstructed_spectrum=np.array([[0.1, 0.5, 0.9]], dtype=float),
        nu_axis=np.array([[2280.0, 2290.0, 2300.0]], dtype=float),
    )
    run = RunRecord(
        run_id="run-1",
        task_id="cars_spectroscopy",
        provider="local_runner",
        env_hash="env-1",
        skills_active=[],
        workspace_root=run_root,
    )
    result = evaluate_cars_spectroscopy_run(task_root, run)
    assert result.all_metrics_passed is True
    assert result.failed_metrics == []
    assert result.metrics_actual["temperature_error_K"] == 5.0


def test_evaluate_cars_spectroscopy_run_fails_when_required_field_missing(tmp_path):
    task_root = tmp_path / "cars"
    _write_cars_hidden_inputs(task_root)

    run_root = tmp_path / "run_missing_field"
    (run_root / "output").mkdir(parents=True)
    np.savez(
        run_root / "output" / "reconstruction.npz",
        reconstructed_spectrum=np.array([[0.1, 0.5, 0.9]], dtype=float),
        nu_axis=np.array([[2280.0, 2290.0, 2300.0]], dtype=float),
    )
    run = RunRecord(
        run_id="run-2",
        task_id="cars_spectroscopy",
        provider="local_runner",
        env_hash="env-1",
        skills_active=[],
        workspace_root=run_root,
    )
    result = evaluate_cars_spectroscopy_run(task_root, run)
    assert result.all_metrics_passed is False
    assert "missing_required_field" in result.failure_tags


def test_evaluate_cars_spectroscopy_run_fails_when_nu_axis_shape_is_invalid(tmp_path):
    task_root = tmp_path / "cars"
    _write_cars_hidden_inputs(task_root)

    run_root = tmp_path / "run_bad_shape"
    (run_root / "output").mkdir(parents=True)
    np.savez(
        run_root / "output" / "reconstruction.npz",
        estimated_temperature_K=np.array([2400.0]),
        reconstructed_spectrum=np.array([[0.1, 0.5, 0.9]], dtype=float),
        nu_axis=np.array([[2280.0, 2290.0]], dtype=float),
    )
    run = RunRecord(
        run_id="run-3",
        task_id="cars_spectroscopy",
        provider="local_runner",
        env_hash="env-1",
        skills_active=[],
        workspace_root=run_root,
    )
    result = evaluate_cars_spectroscopy_run(task_root, run)
    assert result.all_metrics_passed is False
    assert "invalid_output_shape" in result.failure_tags


def test_cars_spectroscopy_proxy_spec_uses_task_output_name(tmp_path):
    run = RunRecord(
        run_id="run-1",
        task_id="cars_spectroscopy",
        provider="local_runner",
        env_hash="env-1",
        skills_active=[],
        workspace_root=tmp_path,
    )
    spec = cars_spectroscopy_proxy_spec(run)
    assert Path(spec["output_path"]).name == "reconstruction.npz"


def test_manifest_proxy_spec_uses_manifest_output_contract(tmp_path):
    run = RunRecord(
        run_id="run-1",
        task_id="task-a",
        provider="local_runner",
        env_hash="env-1",
        skills_active=[],
        workspace_root=tmp_path,
    )
    manifest = {
        "runtime_layout": {
            "data_dir": "data",
            "work_dir": "work",
            "output_dir": "output",
            "checkpoints_dir": "checkpoints",
            "public_bundle_dir": "public_bundle",
        },
        "output_contract": {
            "required_outputs": [{"path": "output/custom_result.npz", "format": "npz"}]
        },
        "proxy_spec": {"output_dtype": "npz"},
    }
    assert manifest_output_contract_path(manifest) == "output/custom_result.npz"
    assert manifest_primary_output_path(run, manifest) == tmp_path / "output" / "custom_result.npz"
    spec = manifest_proxy_spec(run, manifest)
    assert Path(spec["output_path"]).name == "custom_result.npz"
    assert spec["output_dtype"] == "npz"


def test_manifest_proxy_spec_reads_public_task_contract_from_task_root(tmp_path):
    task_root = tmp_path / "task_a"
    (task_root / "evaluation").mkdir(parents=True)
    (task_root / "evaluation" / "task_contract.public.json").write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "family": "optics",
                "files": [],
                "execution": {"entrypoint": "work/main.py"},
                "output": {
                    "path": "output/custom_result.npz",
                    "format": "npz",
                    "fields": [
                        {
                            "name": "reconstruction",
                            "dtype": "float32",
                            "shape": [1, 16, 16],
                            "semantics": "Prediction.",
                        }
                    ],
                },
                "metrics": [],
            }
        ),
        encoding="utf-8",
    )
    run = RunRecord(
        run_id="run-2",
        task_id="task-a",
        provider="local_runner",
        env_hash="env-1",
        skills_active=[],
        workspace_root=tmp_path / "workspace",
    )
    manifest = {
        "primary_output_path": "output/custom_result.npz",
        "task_contract_public_path": "evaluation/task_contract.public.json",
    }

    spec = manifest_proxy_spec(run, manifest, task_root=task_root)

    assert Path(spec["output_path"]).name == "custom_result.npz"
    assert spec["output_dtype"] == "npz"
    assert spec["field_specs"][0]["name"] == "reconstruction"
