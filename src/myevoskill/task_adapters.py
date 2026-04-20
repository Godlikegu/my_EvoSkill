"""Task-specific policies, proxy specs, and hidden judge adapters."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

import numpy as np

from .judging import HiddenJudge, MetricRequirement
from .models import JudgeResult, PublicExposurePolicy, READMEPolicy, RunRecord
from .task_contract import load_task_contract
from .task_runtime import (
    coerce_runtime_layout,
    primary_output_relative_path,
    resolve_primary_output_path,
)
from .task_contract import output_field_map


def cars_spectroscopy_public_policy() -> PublicExposurePolicy:
    """Public exposure policy for the first real task."""

    return PublicExposurePolicy(
        readme_policy=READMEPolicy(
            preserve_sections=(
                "Method Hints",
                "References",
                "Data Description",
                "Problem Description",
            ),
            remove_sections=(),
            remove_path_patterns=(
                r"(?i)data/ground_truth\.npz",
                r"(?i)evaluation/tests/",
                r"(?i)\bsrc/",
                r"(?i)\bmain\.py\b",
                r"(?i)\bnotebooks/",
                r"(?i)\bplan/",
            ),
            preserve_user_eval_notes=True,
        ),
        public_data_allowlist=("data/raw_data.npz", "data/meta_data.json"),
        public_data_denylist=("ground_truth.npz", "baseline_reference.npz"),
    )


def cars_spectroscopy_proxy_spec(run_record: RunRecord) -> Dict[str, object]:
    output_path = resolve_primary_output_path(
        run_record.workspace_root,
        {
            "runtime_layout": coerce_runtime_layout(),
            "proxy_spec": {"primary_output": "output/reconstruction.npz"},
        },
    )
    return {
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
            "nu_axis",
        ],
        "same_shape_fields": ["reconstructed_spectrum", "nu_axis"],
        "runtime_seconds": run_record.runtime_seconds,
        "physical_checks": {"non_negative_runtime": run_record.runtime_seconds >= 0},
    }


def manifest_proxy_spec(
    run_record: RunRecord,
    manifest: Mapping[str, Any],
    task_root: Path | None = None,
) -> Dict[str, object]:
    """Resolve proxy spec from a manifest without task-name branching."""

    task_contract: dict[str, object] = {}
    if task_root is not None:
        try:
            task_contract = load_task_contract(task_root, public=True)
        except FileNotFoundError:
            task_contract = {}
    else:
        task_contract = dict(manifest.get("task_contract") or {})
    if task_contract:
        output_path = resolve_primary_output_path(run_record.workspace_root, manifest)
        output = dict(task_contract.get("output") or {})
        fields = output_field_map(task_contract)
        return {
            "output_path": output_path,
            "output_dtype": str(output.get("format", "") or ""),
            "field_specs": [dict(value) for value in fields.values()],
            "runtime_seconds": run_record.runtime_seconds,
            "physical_checks": {"non_negative_runtime": run_record.runtime_seconds >= 0},
            "public_baseline_delta": {},
        }

    spec = dict(manifest.get("proxy_spec") or {})
    output_path = resolve_primary_output_path(run_record.workspace_root, manifest)
    return {
        "output_path": output_path,
        "output_dtype": str(spec.get("output_dtype", "")),
        "required_fields": list(spec.get("required_fields", [])),
        "numeric_fields": list(spec.get("numeric_fields", [])),
        "same_shape_fields": list(spec.get("same_shape_fields", [])),
        "runtime_seconds": run_record.runtime_seconds,
        "physical_checks": {
            "non_negative_runtime": run_record.runtime_seconds >= 0,
            **dict(spec.get("physical_checks") or {}),
        },
        "public_baseline_delta": dict(spec.get("public_baseline_delta") or {}),
    }


def evaluate_cars_spectroscopy_run(task_root: Path, run_record: RunRecord) -> JudgeResult:
    """Evaluate cars_spectroscopy outputs from a run workspace."""

    task_root = Path(task_root)
    output_path = resolve_primary_output_path(
        run_record.workspace_root,
        {
            "runtime_layout": coerce_runtime_layout(),
            "proxy_spec": {"primary_output": "output/reconstruction.npz"},
        },
    )
    if not output_path.exists():
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=[
                MetricRequirement("ncc_vs_ref", 0.0, ">="),
                MetricRequirement("nrmse_vs_ref", 0.0, "<="),
                MetricRequirement("temperature_error_K", 0.0, "<="),
            ],
            failure_tags=["missing_output"],
        )

    metrics_cfg = json.loads(
        (task_root / "evaluation" / "metrics.json").read_text(encoding="utf-8")
    )
    requirements = [
        MetricRequirement("ncc_vs_ref", float(metrics_cfg["ncc_boundary"]), ">="),
        MetricRequirement("nrmse_vs_ref", float(metrics_cfg["nrmse_boundary"]), "<="),
        MetricRequirement(
            "temperature_error_K",
            float(metrics_cfg["temperature_error_K_boundary"]),
            "<=",
        ),
    ]

    try:
        payload = np.load(output_path)
    except Exception:
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=requirements,
            failure_tags=["invalid_output_file"],
        )

    required_fields = ("estimated_temperature_K", "reconstructed_spectrum", "nu_axis")
    missing_fields = [field for field in required_fields if field not in payload.files]
    if missing_fields:
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=requirements,
            failure_tags=["missing_required_field"],
        )

    try:
        estimated_temperature = _coerce_single_float(payload["estimated_temperature_K"])
        reconstructed_spectrum = _coerce_single_series(payload["reconstructed_spectrum"])
        nu_axis = _coerce_single_series(payload["nu_axis"])
    except TypeError:
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=requirements,
            failure_tags=["invalid_output_dtype"],
        )
    except ValueError:
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=requirements,
            failure_tags=["invalid_output_shape"],
        )

    if reconstructed_spectrum.shape != nu_axis.shape:
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=requirements,
            failure_tags=["invalid_output_shape"],
        )

    raw_payload = np.load(task_root / "data" / "raw_data.npz")
    ground_truth = np.load(task_root / "data" / "ground_truth.npz")
    measurements = _coerce_single_series(raw_payload["measurements"])
    expected_nu_axis = _coerce_single_series(raw_payload["nu_axis"])
    if expected_nu_axis.shape != nu_axis.shape:
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=requirements,
            failure_tags=["invalid_output_shape"],
        )
    if not np.allclose(nu_axis, expected_nu_axis, rtol=1e-6, atol=1e-6):
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=requirements,
            failure_tags=["reference_alignment_failed"],
        )

    compute_metrics = _load_cars_compute_metrics(task_root)
    metrics = compute_metrics(
        measurements,
        reconstructed_spectrum,
        {"temperature": estimated_temperature},
        params_true={"temperature": _coerce_single_float(ground_truth["temperature"])},
    )
    if "ncc" not in metrics or "nrmse" not in metrics or "temperature_error_K" not in metrics:
        return HiddenJudge().evaluate(
            task_id=run_record.task_id,
            metrics_actual={},
            requirements=requirements,
            failure_tags=["reference_alignment_failed"],
        )

    metrics_actual = {
        "ncc_vs_ref": float(metrics["ncc"]),
        "nrmse_vs_ref": float(metrics["nrmse"]),
        "temperature_error_K": float(metrics["temperature_error_K"]),
    }
    failure_tags: List[str] = []
    if metrics_actual["temperature_error_K"] > float(metrics_cfg["temperature_error_K_boundary"]):
        failure_tags.append("temperature_mismatch")
    return HiddenJudge().evaluate(
        task_id=run_record.task_id,
        metrics_actual=metrics_actual,
        requirements=requirements,
        failure_tags=failure_tags,
    )


def manifest_primary_output_path(
    run_record: RunRecord,
    manifest: Mapping[str, Any],
) -> Path:
    """Expose manifest-driven output resolution for tests and plugins."""

    return resolve_primary_output_path(run_record.workspace_root, manifest)


def manifest_output_contract_path(manifest: Mapping[str, Any]) -> str:
    """Expose the manifest primary output path relative to runtime root."""

    return primary_output_relative_path(manifest)


def _load_cars_compute_metrics(task_root: Path) -> Callable[..., Dict[str, float]]:
    helper_path = task_root / "src" / "visualization.py"
    if helper_path.exists():
        spec = importlib.util.spec_from_file_location(
            "myevoskill_cars_visualization",
            helper_path,
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            compute_metrics = getattr(module, "compute_metrics", None)
            if callable(compute_metrics):
                return compute_metrics
    return _fallback_compute_metrics


def _fallback_compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params_pred: Mapping[str, float],
    params_true: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics: Dict[str, float] = {}
    mse = float(np.mean((y_true - y_pred) ** 2))
    metrics["mse"] = mse
    if mse > 0:
        metrics["psnr_dB"] = float(10 * np.log10(1.0 / mse))
    norm_true = np.linalg.norm(y_true)
    norm_pred = np.linalg.norm(y_pred)
    if norm_true > 0 and norm_pred > 0:
        metrics["ncc"] = float(np.dot(y_true.ravel(), y_pred.ravel()) / (norm_true * norm_pred))
    drange = float(y_true.max() - y_true.min())
    if drange > 0:
        metrics["nrmse"] = float(np.sqrt(mse) / drange)
    if params_true is not None and "temperature" in params_true:
        metrics["temperature_error_K"] = float(
            abs(float(params_true["temperature"]) - float(params_pred["temperature"]))
        )
    return metrics


def _coerce_single_float(value: Any) -> float:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("expected numeric scalar")
    if array.ndim == 0:
        return float(array)
    flattened = array.reshape(-1)
    if flattened.size != 1:
        raise ValueError("expected scalar or length-1 array")
    return float(flattened[0])


def _coerce_single_series(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("expected numeric array")
    if array.ndim == 1:
        return array.astype(float, copy=False)
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1).astype(float, copy=False)
    raise ValueError("expected 1D array or single-row/single-column 2D array")
