import numpy as np

from myevoskill.models import RunRecord
from myevoskill.proxy import ProxyVerifier


def test_proxy_verifier_returns_low_leakage_feedback(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "output").mkdir(parents=True)
    output_path = run_root / "output" / "reconstruction.npy"
    output_path.write_text("placeholder", encoding="utf-8")

    record = RunRecord(
        run_id="run-1",
        task_id="task-1",
        provider="fallback",
        env_hash="env-1",
        skills_active=[],
        workspace_root=run_root,
        runtime_seconds=3.5,
    )
    feedback = ProxyVerifier().evaluate(
        record,
        {
            "output_path": output_path,
            "output_shape": [16, 16],
            "output_dtype": "float32",
            "physical_checks": {"finite_energy": True},
        },
    )
    assert feedback.output_exists is True
    assert feedback.output_shape == [16, 16]
    assert feedback.output_dtype == "float32"
    assert feedback.warnings == []


def test_proxy_verifier_checks_npz_required_fields_and_shapes(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "output").mkdir(parents=True)
    output_path = run_root / "output" / "reconstruction.npz"
    np.savez(
        output_path,
        estimated_temperature_K=np.array([2400.0]),
        reconstructed_spectrum=np.array([[1.0, 2.0]], dtype=float),
    )

    record = RunRecord(
        run_id="run-2",
        task_id="cars_spectroscopy",
        provider="fallback",
        env_hash="env-1",
        skills_active=[],
        workspace_root=run_root,
    )
    feedback = ProxyVerifier().evaluate(
        record,
        {
            "output_path": output_path,
            "output_dtype": "npz",
            "required_fields": [
                "estimated_temperature_K",
                "reconstructed_spectrum",
                "nu_axis",
            ],
            "numeric_fields": [
                "estimated_temperature_K",
                "reconstructed_spectrum",
            ],
            "same_shape_fields": [
                "reconstructed_spectrum",
                "nu_axis",
            ],
        },
    )
    assert feedback.output_exists is True
    assert any("missing required fields" in warning for warning in feedback.warnings)
