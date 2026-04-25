"""Utilities for migrating bootstrap tasks into formal task contracts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .manifest_bootstrap import _infer_family, _read_readme_info, _resolve_output_root
from .task_registration import register_task


FORMALIZED_TASK_IDS = {
    "cars_spectroscopy",
    "conventional_ptychography",
    "mri_grappa",
    "mri_t2_mapping",
    "mri_varnet",
    "SSNP_ODT",
}

HIDDEN_DATA_FILENAMES = {
    "ground_truth.npz",
    "baseline_reference.npz",
}
PRIVATE_FILE_PATTERNS = (
    "evaluation/reference_outputs/",
    "evaluation/task_contract.json",
    "evaluation/task_contract.public.json",
    "evaluation/judge_adapter.py",
    "evaluation/task_metric_adapter.py",
)
PUBLIC_DATA_ROLE_BY_SUFFIX = {
    ".json": "metadata",
    ".npz": "input_data",
    ".npy": "input_data",
    ".mat": "input_data",
    ".pt": "input_data",
    ".pth": "input_data",
    ".fits": "input_data",
    ".uvfits": "input_data",
}
OUTPUT_AUX_FIELD_TOKENS = (
    "loss",
    "history",
    "metric",
    "time",
    "x_",
    "y_",
    "z_",
    "theta",
    "angle",
    "coord",
    "axis",
    "profile",
    "signal",
    "dcf",
    "likelihood",
    "cost",
)
CENTERED_NCC_TASKS = {
    "confocal-nlos-fk",
    "eht_black_hole_dynamic",
    "eht_black_hole_original",
    "lucky_imaging",
    "microscope_denoising",
    "plane_wave_ultrasound",
    "reflection_ODT",
    "weather_radar_data_assimilation",
}
PHASE_COMPLEX_TASKS = {
    "fourier_ptychography",
}
MAGNITUDE_COMPLEX_TASKS = {
    "xray_laminography_tike",
}
PRIMARY_REFERENCE_FILE_PRIORITY = (
    "data/ground_truth.npz",
    "data/baseline_reference.npz",
    "evaluation/reference_outputs/reconstruction.npz",
    "evaluation/reference_outputs/reference_reconstruction.npz",
    "evaluation/reference_outputs/reconstructions.npz",
    "evaluation/reference_outputs/reconstruction.npy",
    "evaluation/reference_outputs/ground_truth.npy",
    "evaluation/reference_outputs/posterior_mean.npy",
    "evaluation/reference_outputs/emission_3d.npy",
    "evaluation/reference_outputs/baseline_reference.npz",
    "evaluation/reference_outputs/baseline_reference.npy",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _relative(task_root: Path, path: Path) -> str:
    return path.relative_to(task_root).as_posix()


def _load_structured_file(path: Path) -> dict[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            result: dict[str, dict[str, Any]] = {}
            for field in payload.files:
                try:
                    value = np.asarray(payload[field])
                except Exception:
                    continue
                result[str(field)] = {
                    "dtype": str(value.dtype),
                    "shape": [int(dim) for dim in value.shape],
                }
            return result
    if suffix == ".npy":
        value = np.asarray(np.load(path, allow_pickle=False))
        return {
            "__array__": {
                "dtype": str(value.dtype),
                "shape": [int(dim) for dim in value.shape],
            }
        }
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _describe_json_payload(payload)
    return {}


def _describe_json_payload(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, Mapping):
        result: dict[str, dict[str, Any]] = {}
        for key, value in payload.items():
            array = np.asarray(value)
            result[str(key)] = {
                "dtype": str(array.dtype) if hasattr(array, "dtype") else type(value).__name__,
                "shape": [int(dim) for dim in array.shape] if hasattr(array, "shape") else [],
            }
        return result
    array = np.asarray(payload)
    return {
        "__value__": {
            "dtype": str(array.dtype) if hasattr(array, "dtype") else type(payload).__name__,
            "shape": [int(dim) for dim in array.shape] if hasattr(array, "shape") else [],
        }
    }


def _strip_leading_batch(shape: Sequence[int]) -> tuple[list[int], dict[str, Any]]:
    normalized = [int(dim) for dim in shape]
    selectors: dict[str, Any] = {}
    if normalized == [1]:
        return [], {"index": 0}
    if normalized and normalized[0] == 1:
        return normalized[1:], {"index": 0}
    return normalized, selectors


def _apply_selectors_to_shape(
    shape: Sequence[int],
    selectors: Optional[Mapping[str, Any]] = None,
) -> list[int]:
    result = [int(dim) for dim in shape]
    selectors = dict(selectors or {})
    if "slice" in selectors and selectors.get("slice") is not None:
        start, stop, step = list(selectors.get("slice") or [])[:3]
        start = None if start is None else int(start)
        stop = None if stop is None else int(stop)
        step = None if step is None else int(step)
        count = len(list(range(result[0]))[slice(start, stop, step)])
        result = [count, *result[1:]]
    if "index" in selectors and selectors.get("index") is not None:
        if not result:
            raise ValueError("cannot index into a scalar")
        result = result[1:]
    if bool(selectors.get("squeeze", False)):
        result = [dim for dim in result if dim != 1]
    return result


def _binding(
    *,
    source: str,
    field: str,
    base_shape: Sequence[int],
    file_id: str = "",
    selectors: Optional[Mapping[str, Any]] = None,
    preprocess: str = "identity",
    value: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": source,
        "field": field,
        "preprocess": preprocess,
        "expected_shape": _apply_selectors_to_shape(base_shape, selectors),
    }
    if selectors:
        payload["selectors"] = _jsonable(dict(selectors))
    if source == "file":
        payload["file_id"] = file_id
    if source == "value":
        payload["value"] = _jsonable(value)
    return payload


def _value_binding(*, value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "source": "value",
        "field": "",
        "value": _jsonable(value),
        "preprocess": "identity",
        "expected_shape": [int(dim) for dim in array.shape],
    }


def _public_file_entry(
    *,
    file_id: str,
    path: str,
    role: str,
    semantics: str,
    fields: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "id": file_id,
        "path": path,
        "visibility": "public",
        "role": role,
        "semantics": semantics,
    }
    if fields:
        payload["fields"] = _jsonable(fields)
    return payload


def _private_file_entry(
    *,
    file_id: str,
    path: str,
    role: str,
    semantics: str,
    fields: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "id": file_id,
        "path": path,
        "visibility": "private",
        "role": role,
        "semantics": semantics,
    }
    if fields:
        payload["fields"] = _jsonable(fields)
    return payload


def _load_metrics_config(task_root: Path) -> dict[str, Any]:
    metrics_path = task_root / "evaluation" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _metric_goal(metric_name: str) -> str:
    normalized = str(metric_name).lower()
    if any(token in normalized for token in ("ncc", "snr", "cnr", "sharpness", "n_locs")):
        return "maximize"
    return "minimize"


def _metric_key_stems(metrics_config: Mapping[str, Any]) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    for key, value in dict(metrics_config).items():
        normalized_key = str(key)
        if "_boundary" not in normalized_key:
            continue
        result.append((normalized_key.replace("_boundary", "", 1), float(value)))
    return result


def _ast_function_args(python_path: Path, function_name: str) -> list[str]:
    try:
        tree = ast.parse(python_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return [arg.arg for arg in node.args.args]
    return []


def _find_visualization_function(task_root: Path, candidates: Sequence[str]) -> Optional[str]:
    visualization_path = task_root / "src" / "visualization.py"
    if not visualization_path.exists():
        return None
    try:
        source = visualization_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    for name in candidates:
        if f"def {name}(" in source:
            return name
    return None


def _preferred_public_data_files(task_root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    readme_path = task_root / "README.md"
    if readme_path.exists():
        files.append(
            _public_file_entry(
                file_id="readme",
                path="README.md",
                role="task_description",
                semantics="Authoritative public task description and method hints.",
            )
        )
    requirements_path = task_root / "requirements.txt"
    if requirements_path.exists():
        files.append(
            _public_file_entry(
                file_id="requirements",
                path="requirements.txt",
                role="runtime_dependencies",
                semantics="Available Python dependencies for the execution workspace.",
            )
        )
    data_dir = task_root / "data"
    if not data_dir.exists():
        return files
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in HIDDEN_DATA_FILENAMES:
            continue
        relative_path = _relative(task_root, path)
        role = PUBLIC_DATA_ROLE_BY_SUFFIX.get(path.suffix.lower(), "input_data")
        file_id = relative_path.replace("/", "_").replace(".", "_").replace("-", "_")
        semantics = "Public task-local resource available to the execution agent."
        fields = _load_structured_file(path)
        files.append(
            _public_file_entry(
                file_id=file_id,
                path=relative_path,
                role=role,
                semantics=semantics,
                fields=fields or None,
            )
        )
    return files


def _candidate_reference_fields(path: Path) -> list[str]:
    fields = _load_structured_file(path)
    if not fields:
        return []
    names = list(fields)
    if names == ["__array__"]:
        return names
    filtered: list[str] = []
    for name in names:
        lowered = name.lower()
        if any(token in lowered for token in OUTPUT_AUX_FIELD_TOKENS):
            continue
        filtered.append(name)
    return filtered or names


def _pick_reference_resource(task_root: Path, task_id: str) -> tuple[Path, str]:
    overrides: dict[str, tuple[str, str]] = {
        "confocal-nlos-fk": ("evaluation/reference_outputs/reconstruction.npz", "fk"),
        "eht_black_hole_dynamic": ("evaluation/reference_outputs/ground_truth.npy", "__array__"),
        "eht_black_hole_tomography": ("data/ground_truth.npz", "emission_3d"),
        "eht_black_hole_UQ": ("data/ground_truth.npz", "image"),
        "fpm_inr_reconstruction": ("data/ground_truth.npz", "I_stack"),
        "hessian_sim": ("evaluation/reference_outputs/hessian_sim.npz", "data"),
        "light_field_microscope": ("data/ground_truth.npz", "target_images"),
        "mcr_hyperspectral": ("data/ground_truth.npz", "concentrations_ravel"),
        "raman_cell_phenotyping": ("data/baseline_reference.npz", "abundance_lipids"),
        "weather_radar_data_assimilation": ("data/ground_truth.npz", "target_frames"),
        "xray_ptychography_tike": ("data/baseline_reference.npz", "object_phase"),
    }
    if task_id in overrides:
        relative_path, field_name = overrides[task_id]
        return task_root / relative_path, field_name
    for relative_path in PRIMARY_REFERENCE_FILE_PRIORITY:
        path = task_root / relative_path
        if not path.exists():
            continue
        fields = _candidate_reference_fields(path)
        if not fields:
            continue
        preferred_names = (
            "reconstruction",
            "image",
            "phantom",
            "activity_map",
            "delta_n",
            "ground_truth_image",
            "object_phase",
            "volume",
            "velocity",
            "state",
            "fa_map",
            "posterior_mean",
            "emission_3d",
        )
        for preferred_name in preferred_names:
            if preferred_name in fields:
                return path, preferred_name
        return path, fields[0]
    raise FileNotFoundError(f"could not infer a reference resource for task {task_id}")


def _infer_simple_output_field(task_root: Path, task_id: str, reference_field: str) -> str:
    overrides = {
        "confocal-nlos-fk": "fk",
        "eht_black_hole_dynamic": "reconstruction",
        "eht_black_hole_UQ": "posterior_mean",
        "fpm_inr_reconstruction": "amplitude_stack",
        "light_field_microscope": "rl_reconstructions",
        "mcr_hyperspectral": "concentrations_ravel",
        "weather_radar_data_assimilation": "reconstructed_frames",
    }
    if task_id in overrides:
        return overrides[task_id]
    if reference_field == "__array__":
        save_field_candidates = _saved_npz_fields(task_root / "main.py")
        if save_field_candidates:
            return save_field_candidates[0]
        return "reconstruction"
    return reference_field


def _saved_npz_fields(main_py: Path) -> list[str]:
    if not main_py.exists():
        return []
    try:
        tree = ast.parse(main_py.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    fields: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name not in {"np.savez", "np.savez_compressed"}:
            continue
        for keyword in node.keywords:
            if keyword.arg and not any(token in keyword.arg.lower() for token in OUTPUT_AUX_FIELD_TOKENS):
                fields.append(str(keyword.arg))
    return list(dict.fromkeys(fields))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _adapter_metric(metric_name: str, threshold: float, inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": metric_name,
        "goal": _metric_goal(metric_name),
        "threshold": float(threshold),
        "helper": {
            "interface": "metric_adapter",
            "file_id": "metric_adapter",
            "callable": "compute_metric",
        },
        "inputs": _jsonable(inputs),
    }


def _np_field_descriptor(path: Path, field_name: str) -> dict[str, Any]:
    fields = _load_structured_file(path)
    if field_name not in fields:
        raise KeyError(f"field {field_name!r} not found in {path}")
    return dict(fields[field_name])


def _simple_single_contract(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path, reference_field = _pick_reference_resource(task_root, task_id)
    reference_desc = _np_field_descriptor(reference_path, reference_field)
    output_shape, reference_selectors = _strip_leading_batch(reference_desc["shape"])
    output_field = _infer_simple_output_field(task_root, task_id, reference_field)
    metrics_config = _load_metrics_config(task_root)
    ncc_op = "centered_ncc" if task_id in CENTERED_NCC_TASKS else "cosine_ncc"
    nrmse_op = "nrmse"
    if task_id in PHASE_COMPLEX_TASKS:
        ncc_op = "phase_centered_ncc"
        nrmse_op = "phase_centered_nrmse"
    elif task_id in MAGNITUDE_COMPLEX_TASKS:
        ncc_op = "magnitude_cosine_ncc"
        nrmse_op = "magnitude_nrmse"
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="reference",
            path=_relative(task_root, reference_path),
            role="reference_data",
            semantics="Hidden reference resource used by judge metrics.",
            fields=_load_structured_file(reference_path),
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    output_fields = [
        {
            "name": output_field,
            "dtype": reference_desc["dtype"],
            "shape": output_shape,
            "semantics": "Primary task output used by the evaluation metrics.",
        }
    ]
    metrics: list[dict[str, Any]] = []
    adapter_recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        if "ncc" in metric_name:
            metric_inputs = {
                "estimate": _binding(
                    source="output",
                    field=output_field,
                    base_shape=output_shape,
                ),
                "reference": _binding(
                    source="file",
                    file_id="reference",
                    field=reference_field,
                    base_shape=reference_desc["shape"],
                    selectors=reference_selectors or None,
                ),
            }
            metrics.append(_adapter_metric(metric_name, threshold, metric_inputs))
            adapter_recipes[metric_name] = {"op": ncc_op}
            continue
        if "nrmse" in metric_name:
            metric_inputs = {
                "estimate": _binding(
                    source="output",
                    field=output_field,
                    base_shape=output_shape,
                ),
                "reference": _binding(
                    source="file",
                    file_id="reference",
                    field=reference_field,
                    base_shape=reference_desc["shape"],
                    selectors=reference_selectors or None,
                ),
            }
            metrics.append(_adapter_metric(metric_name, threshold, metric_inputs))
            adapter_recipes[metric_name] = {"op": nrmse_op}
            continue
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=output_fields,
        metrics=metrics,
    )
    return contract, adapter_recipes


def _base_contract(
    *,
    task_root: Path,
    task_id: str,
    files: list[dict[str, Any]],
    output_fields: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    output_path: str = "output/reconstruction.npz",
) -> dict[str, Any]:
    readme_info = _read_readme_info(task_root / "README.md")
    public_file_ids = [item["id"] for item in files if item["visibility"] == "public"]
    execution_files = [
        file_id
        for file_id in ("readme", "requirements", *public_file_ids)
        if file_id in public_file_ids or file_id in {"readme", "requirements"}
    ]
    seen: list[str] = []
    for file_id in execution_files:
        if file_id not in seen:
            seen.append(file_id)
    return {
        "version": 2,
        "task_id": task_id,
        "family": _infer_family(readme_info),
        "files": files,
        "execution": {
            "read_first": seen,
            "readable_files": seen,
            "entrypoint": "work/main.py",
            "writable_paths": ["work/", "output/", "checkpoints/"],
        },
        "output": {
            "path": output_path,
            "format": "npz",
            "fields": output_fields,
        },
        "metrics": metrics,
    }


def _contract_for_ct_dual_energy(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_path = task_root / "data" / "ground_truth.npz"
    gt_fields = _load_structured_file(gt_path)
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="ground_truth",
            path="data/ground_truth.npz",
            role="reference_data",
            semantics="Hidden material-decomposition reference maps.",
            fields=gt_fields,
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    metrics_config = _load_metrics_config(task_root)
    output_fields = []
    for field_name in ("tissue_map", "bone_map"):
        desc = gt_fields[field_name]
        shape, selectors = _strip_leading_batch(desc["shape"])
        output_fields.append(
            {
                "name": field_name,
                "dtype": desc["dtype"],
                "shape": shape,
                "semantics": f"Estimated {field_name.replace('_', ' ')}.",
            }
        )
        gt_fields[field_name]["_selectors"] = selectors
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "tissue_est": _binding(
                        source="output",
                        field="tissue_map",
                        base_shape=output_fields[0]["shape"],
                    ),
                    "tissue_ref": _binding(
                        source="file",
                        file_id="ground_truth",
                        field="tissue_map",
                        base_shape=gt_fields["tissue_map"]["shape"],
                        selectors=gt_fields["tissue_map"].get("_selectors") or None,
                    ),
                    "bone_est": _binding(
                        source="output",
                        field="bone_map",
                        base_shape=output_fields[1]["shape"],
                    ),
                    "bone_ref": _binding(
                        source="file",
                        file_id="ground_truth",
                        field="bone_map",
                        base_shape=gt_fields["bone_map"]["shape"],
                        selectors=gt_fields["bone_map"].get("_selectors") or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {
            "op": "mean_of_pairs",
            "pairs": [
                {"estimate": "tissue_est", "reference": "tissue_ref"},
                {"estimate": "bone_est", "reference": "bone_ref"},
            ],
            "base_op": "cosine_ncc" if metric_name == "ncc" else "nrmse",
        }
    return _base_contract(task_root=task_root, task_id=task_id, files=files, output_fields=output_fields, metrics=metrics), recipes


def _contract_for_diff_deflectometry(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_path = task_root / "data" / "ground_truth.npz"
    gt_fields = _load_structured_file(gt_path)
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="ground_truth",
            path="data/ground_truth.npz",
            role="reference_data",
            semantics="Hidden reference optical parameters used for evaluation.",
            fields=gt_fields,
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    output_fields = []
    for field_name in ("surface_0_roc_mm", "surface_1_roc_mm", "thickness_mm"):
        desc = gt_fields[field_name]
        shape, selectors = _strip_leading_batch(desc["shape"])
        output_fields.append(
            {
                "name": field_name,
                "dtype": desc["dtype"],
                "shape": shape,
                "semantics": f"Estimated scalar parameter {field_name}.",
            }
        )
        gt_fields[field_name]["_selectors"] = selectors
    metrics_config = _load_metrics_config(task_root)
    input_bindings = {
        "surface_0_est": _binding(source="output", field="surface_0_roc_mm", base_shape=[]),
        "surface_0_ref": _binding(
            source="file",
            file_id="ground_truth",
            field="surface_0_roc_mm",
            base_shape=gt_fields["surface_0_roc_mm"]["shape"],
            selectors=gt_fields["surface_0_roc_mm"].get("_selectors") or None,
        ),
        "surface_1_est": _binding(source="output", field="surface_1_roc_mm", base_shape=[]),
        "surface_1_ref": _binding(
            source="file",
            file_id="ground_truth",
            field="surface_1_roc_mm",
            base_shape=gt_fields["surface_1_roc_mm"]["shape"],
            selectors=gt_fields["surface_1_roc_mm"].get("_selectors") or None,
        ),
        "thickness_est": _binding(source="output", field="thickness_mm", base_shape=[]),
        "thickness_ref": _binding(
            source="file",
            file_id="ground_truth",
            field="thickness_mm",
            base_shape=gt_fields["thickness_mm"]["shape"],
            selectors=gt_fields["thickness_mm"].get("_selectors") or None,
        ),
    }
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        if metric_name not in {"ncc", "nrmse"}:
            continue
        metrics.append(_adapter_metric(metric_name, threshold, input_bindings))
        recipes[metric_name] = {
            "op": "vector_metric",
            "estimate_keys": ["surface_0_est", "surface_1_est", "thickness_est"],
            "reference_keys": ["surface_0_ref", "surface_1_ref", "thickness_ref"],
            "base_op": "cosine_ncc" if metric_name == "ncc" else "nrmse",
        }
    return _base_contract(task_root=task_root, task_id=task_id, files=files, output_fields=output_fields, metrics=metrics), recipes


def _contract_for_diffusion_mri_dti(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_path = task_root / "data" / "ground_truth.npz"
    gt_fields = _load_structured_file(gt_path)
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="ground_truth",
            path="data/ground_truth.npz",
            role="reference_data",
            semantics="Hidden DTI reference maps and tissue mask.",
            fields=gt_fields,
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    output_fields = []
    output_names = ("fa_map", "md_map", "tensor_elements")
    output_shapes: dict[str, list[int]] = {}
    selectors_by_field: dict[str, dict[str, Any]] = {}
    for field_name in output_names:
        desc = gt_fields[field_name]
        shape, selectors = _strip_leading_batch(desc["shape"])
        output_shapes[field_name] = shape
        selectors_by_field[field_name] = selectors
        output_fields.append(
            {
                "name": field_name,
                "dtype": desc["dtype"],
                "shape": shape,
                "semantics": f"Estimated {field_name.replace('_', ' ')}.",
            }
        )
    metrics_config = _load_metrics_config(task_root)
    mask_shape, mask_selectors = _strip_leading_batch(gt_fields["tissue_mask"]["shape"])
    metric_specs = {
        "ncc": ("fa_map", "centered_ncc", "tissue_mask"),
        "nrmse": ("fa_map", "nrmse", "tissue_mask"),
        "ncc_md": ("md_map", "centered_ncc", "tissue_mask"),
        "nrmse_md": ("md_map", "nrmse", "tissue_mask"),
    }
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        if metric_name not in metric_specs:
            continue
        field_name, op_name, mask_field = metric_specs[metric_name]
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field=field_name, base_shape=output_shapes[field_name]),
                    "reference": _binding(
                        source="file",
                        file_id="ground_truth",
                        field=field_name,
                        base_shape=gt_fields[field_name]["shape"],
                        selectors=selectors_by_field[field_name] or None,
                    ),
                    "mask": _binding(
                        source="file",
                        file_id="ground_truth",
                        field=mask_field,
                        base_shape=gt_fields[mask_field]["shape"],
                        selectors=mask_selectors or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": op_name, "mask_key": "mask"}
    return _base_contract(task_root=task_root, task_id=task_id, files=files, output_fields=output_fields, metrics=metrics), recipes


def _contract_for_eht_dynamic(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path = task_root / "evaluation" / "reference_outputs" / "ground_truth.npy"
    reference_desc = _np_field_descriptor(reference_path, "__array__")
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="reference_npz",
            path="evaluation/reference_outputs/task_contract_refs_ground_truth.npz",
            role="reference_data",
            semantics="Generated hidden dynamic image sequence reference used for metric binding.",
            fields={
                "reference": {
                    "dtype": reference_desc["dtype"],
                    "shape": reference_desc["shape"],
                }
            },
        )
    )
    files.append(
        _private_file_entry(
            file_id="reference",
            path="evaluation/reference_outputs/ground_truth.npy",
            role="reference_data",
            semantics="Hidden dynamic image sequence used for evaluation.",
            fields=_load_structured_file(reference_path),
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    output_shape = reference_desc["shape"]
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field="reconstruction", base_shape=output_shape),
                    "reference": _binding(
                        source="file",
                        file_id="reference_npz",
                        field="reference",
                        base_shape=reference_desc["shape"],
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": "framewise_mean", "base_op": "centered_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "reconstruction",
                "dtype": reference_desc["dtype"],
                "shape": output_shape,
                "semantics": "Estimated dynamic image sequence.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_eht_feature_extraction_dynamic(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_path = task_root / "data" / "ground_truth.npz"
    gt_fields = _load_structured_file(gt_path)
    pa_shape, pa_selectors = _strip_leading_batch(gt_fields["position_angle_deg"]["shape"])
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="ground_truth",
            path="data/ground_truth.npz",
            role="reference_data",
            semantics="Hidden geometric feature references for the dynamic sequence.",
            fields=gt_fields,
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = [
        _adapter_metric(
            "position_angle_mae_deg",
            float(metrics_config["position_angle_mae_boundary_deg"]),
            {
                "estimate": _binding(source="output", field="position_angle_deg", base_shape=pa_shape),
                "reference": _binding(
                    source="file",
                    file_id="ground_truth",
                    field="position_angle_deg",
                    base_shape=gt_fields["position_angle_deg"]["shape"],
                    selectors=pa_selectors or None,
                ),
            },
        )
    ]
    recipes = {"position_angle_mae_deg": {"op": "mean_absolute_error"}}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "position_angle_deg",
                "dtype": gt_fields["position_angle_deg"]["dtype"],
                "shape": pa_shape,
                "semantics": "Estimated position angle per frame in degrees.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_era5(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path = task_root / "evaluation" / "reference_outputs" / "trajectory.npy"
    reference_desc = _np_field_descriptor(reference_path, "__array__")
    lat_weight_fields = _load_structured_file(task_root / "data" / "raw_data.npz")
    lat_shape, lat_selectors = _strip_leading_batch(lat_weight_fields["lat_weight_matrix"]["shape"])
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="reference_npz",
            path="evaluation/reference_outputs/task_contract_refs_trajectory.npz",
            role="reference_data",
            semantics="Generated hidden trajectory tensor used for metric binding.",
            fields={
                "reference": {
                    "dtype": reference_desc["dtype"],
                    "shape": reference_desc["shape"],
                }
            },
        )
    )
    files.append(
        _private_file_entry(
            file_id="reference",
            path="evaluation/reference_outputs/trajectory.npy",
            role="reference_data",
            semantics="Hidden trajectory reference used for evaluation.",
            fields=_load_structured_file(reference_path),
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    metrics_config = _load_metrics_config(task_root)
    common_inputs = {
        "estimate": _binding(source="output", field="state", base_shape=reference_desc["shape"]),
        "reference": _binding(source="file", file_id="reference_npz", field="reference", base_shape=reference_desc["shape"]),
        "lat_weight_matrix": _binding(
            source="file",
            file_id="data_raw_data_npz",
            field="lat_weight_matrix",
            base_shape=lat_weight_fields["lat_weight_matrix"]["shape"],
            selectors=lat_selectors or None,
        ),
    }
    metrics = []
    recipes: dict[str, Any] = {}
    channel_names = {
        "weighted_nrmse_geopotential": 0,
        "weighted_nrmse_temperature": 1,
        "weighted_nrmse_humidity": 2,
        "weighted_nrmse_wind_u": 3,
        "weighted_nrmse_wind_v": 4,
    }
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(_adapter_metric(metric_name, threshold, common_inputs))
        if metric_name == "ncc":
            recipes[metric_name] = {"op": "channel_mean_cosine_ncc"}
        elif metric_name == "nrmse":
            recipes[metric_name] = {"op": "channel_mean_nrmse"}
        elif metric_name == "weighted_nrmse":
            recipes[metric_name] = {"op": "weighted_nrmse_mean"}
        else:
            recipes[metric_name] = {
                "op": "weighted_nrmse_channel",
                "channel": int(channel_names[metric_name]),
            }
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "state",
                "dtype": reference_desc["dtype"],
                "shape": reference_desc["shape"],
                "semantics": "Estimated trajectory tensor with shape (time, channel, lat, lon).",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_exoplanet(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_path = task_root / "evaluation" / "reference_outputs" / "klip_result.npz"
    reference_fields = _load_structured_file(reference_path)
    ref_desc = reference_fields["K10"]
    output_shape, ref_selectors = _strip_leading_batch(ref_desc["shape"])
    meta = json.loads((task_root / "data" / "meta_data.json").read_text(encoding="utf-8"))
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="reference",
            path="evaluation/reference_outputs/klip_result.npz",
            role="reference_data",
            semantics="Hidden KLIP reference image used for evaluation.",
            fields=reference_fields,
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    metrics_config = _load_metrics_config(task_root)
    common_inputs = {
        "image": _binding(source="output", field="klip_image", base_shape=output_shape),
        "reference": _binding(
            source="file",
            file_id="reference",
            field="K10",
            base_shape=ref_desc["shape"],
            selectors=ref_selectors or None,
        ),
    }
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        if metric_name == "snr":
            metric_inputs = {
                "image": _binding(source="output", field="klip_image", base_shape=output_shape),
                "planet_x": _value_binding(
                    value=np.array(float(meta["known_companion"]["planet_x_klip"]), dtype=np.float64)
                ),
                "planet_y": _value_binding(
                    value=np.array(float(meta["known_companion"]["planet_y_klip"]), dtype=np.float64)
                ),
                "fwhm": _value_binding(value=np.array(float(meta["fwhm_px"]), dtype=np.float64)),
                "exclude_nearest": _value_binding(value=np.array(1, dtype=np.int64)),
            }
            metrics.append(_adapter_metric(metric_name, threshold, metric_inputs))
            recipes[metric_name] = {"op": "exoplanet_snr"}
            continue
        metrics.append(_adapter_metric(metric_name, threshold, common_inputs))
        recipes[metric_name] = {"op": "cosine_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "klip_image",
                "dtype": ref_desc["dtype"],
                "shape": output_shape,
                "semantics": "Estimated KLIP-ADI detection map aligned to the reference image.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_light_field(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_fields = _load_structured_file(task_root / "data" / "ground_truth.npz")
    ref_desc = gt_fields["target_images"]
    output_shape, ref_selectors = _strip_leading_batch(ref_desc["shape"])
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="ground_truth",
            path="data/ground_truth.npz",
            role="reference_data",
            semantics="Hidden target images used for evaluation.",
            fields=gt_fields,
        )
    )
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field="rl_reconstructions", base_shape=output_shape),
                    "reference": _binding(
                        source="file",
                        file_id="ground_truth",
                        field="target_images",
                        base_shape=ref_desc["shape"],
                        selectors=ref_selectors or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": "cosine_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "rl_reconstructions",
                "dtype": ref_desc["dtype"],
                "shape": output_shape,
                "semantics": "Estimated reconstructed target images across the provided cases.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_lucky(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ref_fields = _load_structured_file(task_root / "data" / "baseline_reference.npz")
    ref_desc = ref_fields["stacked"]
    output_shape, ref_selectors = _strip_leading_batch(ref_desc["shape"])
    best_desc = _np_field_descriptor(task_root / "evaluation" / "reference_outputs" / "best_frame.npy", "__array__")
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="baseline_reference",
                path="data/baseline_reference.npz",
                role="reference_data",
                semantics="Hidden stacked reference image used for evaluation.",
                fields=ref_fields,
            ),
            _private_file_entry(
                file_id="best_frame_reference",
                path="evaluation/reference_outputs/best_frame.npy",
                role="reference_data",
                semantics="Hidden best single frame used for sharpness comparison.",
                fields=_load_structured_file(task_root / "evaluation" / "reference_outputs" / "best_frame.npy"),
            ),
            _private_file_entry(
                file_id="best_frame_reference_npz",
                path="evaluation/reference_outputs/task_contract_refs_best_frame.npz",
                role="reference_data",
                semantics="Generated hidden best-frame reference used for metric binding.",
                fields={
                    "best_frame": best_desc,
                },
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        if metric_name == "sharpness_ratio_vs_best":
            metrics.append(
                _adapter_metric(
                    metric_name,
                    threshold,
                    {
                        "stacked": _binding(source="output", field="stacked", base_shape=output_shape),
                        "best_frame": _binding(
                            source="file",
                            file_id="best_frame_reference_npz",
                            field="best_frame",
                            base_shape=best_desc["shape"],
                        ),
                    },
                )
            )
            recipes[metric_name] = {"op": "lucky_sharpness_ratio"}
            continue
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field="stacked", base_shape=output_shape),
                    "reference": _binding(
                        source="file",
                        file_id="baseline_reference",
                        field="stacked",
                        base_shape=ref_desc["shape"],
                        selectors=ref_selectors or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": "centered_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "stacked",
                "dtype": ref_desc["dtype"],
                "shape": output_shape,
                "semantics": "Final lucky-imaging stacked output.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_mcr(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_fields = _load_structured_file(task_root / "data" / "ground_truth.npz")
    ref_desc = gt_fields["concentrations_ravel"]
    output_shape, ref_selectors = _strip_leading_batch(ref_desc["shape"])
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="ground_truth",
                path="data/ground_truth.npz",
                role="reference_data",
                semantics="Hidden concentration references used for evaluation.",
                fields=gt_fields,
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field="concentrations_ravel", base_shape=output_shape),
                    "reference": _binding(
                        source="file",
                        file_id="ground_truth",
                        field="concentrations_ravel",
                        base_shape=ref_desc["shape"],
                        selectors=ref_selectors or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": "cosine_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "concentrations_ravel",
                "dtype": ref_desc["dtype"],
                "shape": output_shape,
                "semantics": "Estimated hyperspectral concentration matrix.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_microscope(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ref_fields = _load_structured_file(task_root / "data" / "baseline_reference.npz")
    den_desc = ref_fields["denoised"]
    dec_desc = ref_fields["deconvolved"]
    den_shape, den_selectors = _strip_leading_batch(den_desc["shape"])
    dec_shape, dec_selectors = _strip_leading_batch(dec_desc["shape"])
    raw_fields = _load_structured_file(task_root / "data" / "raw_data.npz")
    meas_shape, meas_selectors = _strip_leading_batch(raw_fields["measurements"]["shape"])
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="baseline_reference",
                path="data/baseline_reference.npz",
                role="reference_data",
                semantics="Hidden denoised and deconvolved reference outputs.",
                fields=ref_fields,
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        if metric_name == "noise_reduction_factor":
            metrics.append(
                _adapter_metric(
                    metric_name,
                    threshold,
                    {
                        "noisy": _binding(
                            source="file",
                            file_id="data_raw_data_npz",
                            field="measurements",
                            base_shape=raw_fields["measurements"]["shape"],
                            selectors={"index": 0, "slice": [0, 1, None], "squeeze": True},
                        ),
                        "denoised": _binding(source="output", field="denoised", base_shape=den_shape),
                    },
                )
            )
            recipes[metric_name] = {"op": "microscope_noise_reduction_factor"}
            continue
        if metric_name == "sharpness_improvement_factor":
            metrics.append(
                _adapter_metric(
                    metric_name,
                    threshold,
                    {
                        "denoised": _binding(source="output", field="denoised", base_shape=den_shape),
                        "deconvolved": _binding(source="output", field="deconvolved", base_shape=dec_shape),
                    },
                )
            )
            recipes[metric_name] = {"op": "microscope_sharpness_improvement_factor"}
            continue
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field="denoised", base_shape=den_shape),
                    "reference": _binding(
                        source="file",
                        file_id="baseline_reference",
                        field="denoised",
                        base_shape=den_desc["shape"],
                        selectors=den_selectors or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": "centered_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "denoised",
                "dtype": den_desc["dtype"],
                "shape": den_shape,
                "semantics": "Stage-1 denoised output.",
            },
            {
                "name": "deconvolved",
                "dtype": dec_desc["dtype"],
                "shape": dec_shape,
                "semantics": "Stage-2 deconvolved output.",
            },
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_plane_wave(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ref_fields = _load_structured_file(task_root / "data" / "baseline_reference.npz")
    fib_desc = ref_fields["bmode_fibers"]
    cyst_desc = ref_fields["bmode_cysts"]
    fib_shape, fib_selectors = _strip_leading_batch(fib_desc["shape"])
    cyst_shape, cyst_selectors = _strip_leading_batch(cyst_desc["shape"])
    x_fib_desc = _np_field_descriptor(task_root / "evaluation" / "reference_outputs" / "x_fibers.npy", "__array__")
    z_fib_desc = _np_field_descriptor(task_root / "evaluation" / "reference_outputs" / "z_fibers.npy", "__array__")
    x_cyst_desc = _np_field_descriptor(task_root / "evaluation" / "reference_outputs" / "x_cysts.npy", "__array__")
    z_cyst_desc = _np_field_descriptor(task_root / "evaluation" / "reference_outputs" / "z_cysts.npy", "__array__")
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="baseline_reference",
                path="data/baseline_reference.npz",
                role="reference_data",
                semantics="Hidden B-mode references used for evaluation.",
                fields=ref_fields,
            ),
            _private_file_entry(
                file_id="fibers_axes",
                path="evaluation/reference_outputs/task_contract_refs_fibers_axes.npz",
                role="reference_data",
                semantics="Generated hidden coordinate vectors for the fibers phantom.",
                fields={
                    "x": x_fib_desc,
                    "z": z_fib_desc,
                },
            ),
            _private_file_entry(
                file_id="cysts_axes",
                path="evaluation/reference_outputs/task_contract_refs_cysts_axes.npz",
                role="reference_data",
                semantics="Generated hidden coordinate vectors for the cysts phantom.",
                fields={
                    "x": x_cyst_desc,
                    "z": z_cyst_desc,
                },
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        if metric_name == "ncc_fibers":
            metrics.append(
                _adapter_metric(
                    metric_name,
                    threshold,
                    {
                        "estimate": _binding(source="output", field="bmode_fibers", base_shape=fib_shape),
                        "reference": _binding(
                            source="file",
                            file_id="baseline_reference",
                            field="bmode_fibers",
                            base_shape=fib_desc["shape"],
                            selectors=fib_selectors or None,
                        ),
                    },
                )
            )
            recipes[metric_name] = {"op": "centered_ncc"}
            continue
        if metric_name == "nrmse_fibers":
            metrics.append(
                _adapter_metric(
                    metric_name,
                    threshold,
                    {
                        "estimate": _binding(source="output", field="bmode_fibers", base_shape=fib_shape),
                        "reference": _binding(
                            source="file",
                            file_id="baseline_reference",
                            field="bmode_fibers",
                            base_shape=fib_desc["shape"],
                            selectors=fib_selectors or None,
                        ),
                    },
                )
            )
            recipes[metric_name] = {"op": "nrmse"}
            continue
        if metric_name == "ncc_cysts":
            metrics.append(
                _adapter_metric(
                    metric_name,
                    threshold,
                    {
                        "estimate": _binding(source="output", field="bmode_cysts", base_shape=cyst_shape),
                        "reference": _binding(
                            source="file",
                            file_id="baseline_reference",
                            field="bmode_cysts",
                            base_shape=cyst_desc["shape"],
                            selectors=cyst_selectors or None,
                        ),
                    },
                )
            )
            recipes[metric_name] = {"op": "centered_ncc"}
            continue
        if metric_name == "nrmse_cysts":
            metrics.append(
                _adapter_metric(
                    metric_name,
                    threshold,
                    {
                        "estimate": _binding(source="output", field="bmode_cysts", base_shape=cyst_shape),
                        "reference": _binding(
                            source="file",
                            file_id="baseline_reference",
                            field="bmode_cysts",
                            base_shape=cyst_desc["shape"],
                            selectors=cyst_selectors or None,
                        ),
                    },
                )
            )
            recipes[metric_name] = {"op": "nrmse"}
            continue
        if metric_name == "psf_fwhm_mm_mean":
            metrics.append(
                _adapter_metric(
                    metric_name,
                    threshold,
                    {
                        "bmode": _binding(source="output", field="bmode_fibers", base_shape=fib_shape),
                        "x": _binding(source="file", file_id="fibers_axes", field="x", base_shape=x_fib_desc["shape"]),
                        "z": _binding(source="file", file_id="fibers_axes", field="z", base_shape=z_fib_desc["shape"]),
                        "z_targets": _value_binding(value=np.array([0.01 * k for k in range(1, 9)], dtype=np.float64)),
                    },
                )
            )
            recipes[metric_name] = {"op": "plane_wave_psf_fwhm_mean"}
            continue
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "bmode": _binding(source="output", field="bmode_cysts", base_shape=cyst_shape),
                    "x": _binding(source="file", file_id="cysts_axes", field="x", base_shape=x_cyst_desc["shape"]),
                    "z": _binding(source="file", file_id="cysts_axes", field="z", base_shape=z_cyst_desc["shape"]),
                    "cyst_centers": _value_binding(value=np.array([[-0.010, 0.020], [0.007, 0.023]], dtype=np.float64)),
                },
            )
        )
        recipes[metric_name] = {"op": "plane_wave_cnr_mean"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "bmode_fibers",
                "dtype": fib_desc["dtype"],
                "shape": fib_shape,
                "semantics": "Estimated B-mode image for the fibers phantom.",
            },
            {
                "name": "bmode_cysts",
                "dtype": cyst_desc["dtype"],
                "shape": cyst_shape,
                "semantics": "Estimated B-mode image for the cysts phantom.",
            },
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_raman(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ref_fields = _load_structured_file(task_root / "data" / "baseline_reference.npz")
    component_names = (
        "abundance_lipids",
        "abundance_nucleus",
        "abundance_cytoplasm",
        "abundance_background",
    )
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="baseline_reference",
                path="data/baseline_reference.npz",
                role="reference_data",
                semantics="Hidden abundance-map references used for evaluation.",
                fields=ref_fields,
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    output_fields = []
    pair_specs: list[dict[str, str]] = []
    for field_name in component_names:
        desc = ref_fields[field_name]
        shape, selectors = _strip_leading_batch(desc["shape"])
        output_fields.append(
            {
                "name": field_name,
                "dtype": desc["dtype"],
                "shape": shape,
                "semantics": f"Estimated abundance map for {field_name.replace('abundance_', '')}.",
            }
        )
        pair_specs.append(
            {
                "estimate": f"{field_name}_est",
                "reference": f"{field_name}_ref",
                "field_name": field_name,
                "selectors": selectors,
            }
        )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        inputs: dict[str, Any] = {}
        recipe_pairs = []
        for spec in pair_specs:
            field_name = spec["field_name"]
            shape = next(item["shape"] for item in output_fields if item["name"] == field_name)
            inputs[spec["estimate"]] = _binding(source="output", field=field_name, base_shape=shape)
            inputs[spec["reference"]] = _binding(
                source="file",
                file_id="baseline_reference",
                field=field_name,
                base_shape=ref_fields[field_name]["shape"],
                selectors=spec["selectors"] or None,
            )
            recipe_pairs.append({"estimate": spec["estimate"], "reference": spec["reference"]})
        metrics.append(_adapter_metric(metric_name, threshold, inputs))
        recipes[metric_name] = {
            "op": "mean_of_pairs",
            "pairs": recipe_pairs,
            "base_op": "cosine_ncc" if metric_name == "ncc" else "nrmse",
        }
    contract = _base_contract(task_root=task_root, task_id=task_id, files=files, output_fields=output_fields, metrics=metrics)
    return contract, recipes


def _contract_for_seismic_traveltime(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ref_fields = _load_structured_file(task_root / "evaluation" / "reference_outputs" / "baseline_reference.npz")
    ref_desc = ref_fields["velocity_perturbation"]
    out_shape, ref_selectors = _strip_leading_batch(ref_desc["shape"])
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="reference",
                path="evaluation/reference_outputs/baseline_reference.npz",
                role="reference_data",
                semantics="Hidden velocity perturbation reference used for evaluation.",
                fields=ref_fields,
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field="velocity_perturbation", base_shape=out_shape),
                    "reference": _binding(
                        source="file",
                        file_id="reference",
                        field="velocity_perturbation",
                        base_shape=ref_desc["shape"],
                        selectors=ref_selectors or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": "cosine_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "velocity_perturbation",
                "dtype": ref_desc["dtype"],
                "shape": out_shape,
                "semantics": "Estimated velocity perturbation relative to the background model.",
            }
        ],
        metrics=metrics,
        output_path="output/velocity_reconstructed.npz",
    )
    return contract, recipes


def _contract_for_shack_hartmann(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_fields = _load_structured_file(task_root / "data" / "ground_truth.npz")
    desc = gt_fields["wavefront_phases"]
    out_shape, ref_selectors = _strip_leading_batch(desc["shape"])
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="ground_truth",
                path="data/ground_truth.npz",
                role="reference_data",
                semantics="Hidden wavefront-phase references used for evaluation.",
                fields=gt_fields,
            ),
            _private_file_entry(
                file_id="ground_truth_levels",
                path="evaluation/reference_outputs/task_contract_refs_wavefront_levels.npz",
                role="reference_data",
                semantics="Generated hidden per-level wavefront slices used for metric binding.",
                fields={
                    f"level_{index}": {
                        "dtype": desc["dtype"],
                        "shape": [int(out_shape[1])],
                    }
                    for index in range(int(out_shape[0]))
                },
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    pair_count = int(out_shape[0])
    for metric_name, threshold in _metric_key_stems(metrics_config):
        inputs: dict[str, Any] = {}
        pairs = []
        for index in range(pair_count):
            est_key = f"estimate_{index}"
            ref_key = f"reference_{index}"
            inputs[est_key] = _binding(
                source="output",
                field="reconstructed_phases",
                base_shape=out_shape,
                selectors={"slice": [index, index + 1, None], "squeeze": True},
            )
            inputs[ref_key] = _binding(
                source="file",
                file_id="ground_truth_levels",
                field=f"level_{index}",
                base_shape=[int(out_shape[1])],
            )
            pairs.append({"estimate": est_key, "reference": ref_key})
        metrics.append(_adapter_metric(metric_name, threshold, inputs))
        recipes[metric_name] = {
            "op": "mean_of_pairs",
            "pairs": pairs,
            "base_op": "cosine_ncc" if metric_name == "ncc" else "nrmse",
        }
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "reconstructed_phases",
                "dtype": desc["dtype"],
                "shape": out_shape,
                "semantics": "Estimated wavefront phase for each WFE level.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_single_molecule(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_config = _load_metrics_config(task_root)
    files = _preferred_public_data_files(task_root)
    files.append(
        _private_file_entry(
            file_id="metric_adapter",
            path="evaluation/task_metric_adapter.py",
            role="metric_helper",
            semantics="Generated private metric adapter for the formal contract.",
        )
    )
    output_fields = [
        {
            "name": "n_locs_3d_filtered",
            "dtype": "int64",
            "shape": [],
            "semantics": "Number of retained 3D localisations after quality filtering.",
        },
        {
            "name": "median_lateral_err_nm",
            "dtype": "float64",
            "shape": [],
            "semantics": "Median lateral localisation error in nanometers.",
        },
        {
            "name": "median_axial_err_nm",
            "dtype": "float64",
            "shape": [],
            "semantics": "Median axial localisation error in nanometers.",
        },
    ]
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {"value": _binding(source="output", field=metric_name, base_shape=[])},
            )
        )
        recipes[metric_name] = {"op": "identity_scalar", "key": "value"}
    contract = _base_contract(task_root=task_root, task_id=task_id, files=files, output_fields=output_fields, metrics=metrics)
    return contract, recipes


def _contract_for_weather(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_fields = _load_structured_file(task_root / "data" / "ground_truth.npz")
    ref_desc = gt_fields["target_frames"]
    out_shape, ref_selectors = _strip_leading_batch(ref_desc["shape"])
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="ground_truth",
                path="data/ground_truth.npz",
                role="reference_data",
                semantics="Hidden target frames used for evaluation.",
                fields=gt_fields,
            ),
            _private_file_entry(
                file_id="ground_truth_frames",
                path="evaluation/reference_outputs/task_contract_refs_target_frames.npz",
                role="reference_data",
                semantics="Generated hidden per-frame targets used for metric binding.",
                fields={
                    f"frame_{index}": {
                        "dtype": ref_desc["dtype"],
                        "shape": [int(out_shape[1]), int(out_shape[2])],
                    }
                    for index in range(int(out_shape[0]))
                },
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        frame_index = int(metric_name[-1]) - 1
        base_metric = "centered_ncc" if metric_name.startswith("ncc_") else "nrmse"
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(
                        source="output",
                        field="reconstructed_frames",
                        base_shape=out_shape,
                        selectors={"slice": [frame_index, frame_index + 1, None], "squeeze": True},
                    ),
                    "reference": _binding(
                        source="file",
                        file_id="ground_truth_frames",
                        field=f"frame_{frame_index}",
                        base_shape=[int(out_shape[1]), int(out_shape[2])],
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": base_metric}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "reconstructed_frames",
                "dtype": ref_desc["dtype"],
                "shape": out_shape,
                "semantics": "Estimated future frame sequence.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_xray_ptycho(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ref_fields = _load_structured_file(task_root / "data" / "baseline_reference.npz")
    ref_desc = ref_fields["object_phase"]
    out_shape, ref_selectors = _strip_leading_batch(ref_desc["shape"])
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="baseline_reference",
                path="data/baseline_reference.npz",
                role="reference_data",
                semantics="Hidden object-phase reference used for evaluation.",
                fields=ref_fields,
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field="object_phase", base_shape=out_shape),
                    "reference": _binding(
                        source="file",
                        file_id="baseline_reference",
                        field="object_phase",
                        base_shape=ref_desc["shape"],
                        selectors=ref_selectors or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": "cosine_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "object_phase",
                "dtype": ref_desc["dtype"],
                "shape": out_shape,
                "semantics": "Estimated object phase.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


def _contract_for_fpm_inr(task_root: Path, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gt_fields = _load_structured_file(task_root / "data" / "ground_truth.npz")
    ref_desc = gt_fields["I_stack"]
    out_shape, ref_selectors = _strip_leading_batch(ref_desc["shape"])
    files = _preferred_public_data_files(task_root)
    files.extend(
        [
            _private_file_entry(
                file_id="ground_truth",
                path="data/ground_truth.npz",
                role="reference_data",
                semantics="Hidden amplitude stack used for evaluation.",
                fields=gt_fields,
            ),
            _private_file_entry(
                file_id="metric_adapter",
                path="evaluation/task_metric_adapter.py",
                role="metric_helper",
                semantics="Generated private metric adapter for the formal contract.",
            ),
        ]
    )
    metrics_config = _load_metrics_config(task_root)
    metrics = []
    recipes: dict[str, Any] = {}
    for metric_name, threshold in _metric_key_stems(metrics_config):
        metrics.append(
            _adapter_metric(
                metric_name,
                threshold,
                {
                    "estimate": _binding(source="output", field="amplitude_stack", base_shape=out_shape),
                    "reference": _binding(
                        source="file",
                        file_id="ground_truth",
                        field="I_stack",
                        base_shape=ref_desc["shape"],
                        selectors=ref_selectors or None,
                    ),
                },
            )
        )
        recipes[metric_name] = {"op": "cosine_ncc" if metric_name == "ncc" else "nrmse"}
    contract = _base_contract(
        task_root=task_root,
        task_id=task_id,
        files=files,
        output_fields=[
            {
                "name": "amplitude_stack",
                "dtype": ref_desc["dtype"],
                "shape": out_shape,
                "semantics": "Estimated FPM amplitude stack.",
            }
        ],
        metrics=metrics,
    )
    return contract, recipes


SPECIAL_CONTRACT_BUILDERS: dict[str, Callable[[Path, str], tuple[dict[str, Any], dict[str, Any]]]] = {
    "ct_dual_energy": _contract_for_ct_dual_energy,
    "differentiable_deflectometry": _contract_for_diff_deflectometry,
    "diffusion_mri_dti": _contract_for_diffusion_mri_dti,
    "eht_black_hole_dynamic": _contract_for_eht_dynamic,
    "eht_black_hole_feature_extraction_dynamic": _contract_for_eht_feature_extraction_dynamic,
    "era5_tensorvar": _contract_for_era5,
    "exoplanet_imaging": _contract_for_exoplanet,
    "fpm_inr_reconstruction": _contract_for_fpm_inr,
    "light_field_microscope": _contract_for_light_field,
    "lucky_imaging": _contract_for_lucky,
    "mcr_hyperspectral": _contract_for_mcr,
    "microscope_denoising": _contract_for_microscope,
    "plane_wave_ultrasound": _contract_for_plane_wave,
    "raman_cell_phenotyping": _contract_for_raman,
    "seismic_traveltime_tomography": _contract_for_seismic_traveltime,
    "shack-hartmann": _contract_for_shack_hartmann,
    "single_molecule_light_field": _contract_for_single_molecule,
    "weather_radar_data_assimilation": _contract_for_weather,
    "xray_ptychography_tike": _contract_for_xray_ptycho,
}


METRIC_ADAPTER_TEMPLATE = """from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


METRIC_RECIPES = {recipes_literal}


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value)


def _cosine_ncc(estimate: Any, reference: Any, *, centered: bool = False, mask: Any = None) -> float:
    est = np.asarray(estimate)
    ref = np.asarray(reference)
    if mask is not None:
        mask_array = np.asarray(mask).astype(bool)
        est = est[mask_array]
        ref = ref[mask_array]
    est = est.astype(np.float64).ravel()
    ref = ref.astype(np.float64).ravel()
    if centered:
        est = est - float(np.mean(est))
        ref = ref - float(np.mean(ref))
    denom = float(np.linalg.norm(est) * np.linalg.norm(ref) + 1e-12)
    return float(np.dot(est, ref) / denom)


def _nrmse(estimate: Any, reference: Any, *, mask: Any = None) -> float:
    est = np.asarray(estimate)
    ref = np.asarray(reference)
    if mask is not None:
        mask_array = np.asarray(mask).astype(bool)
        est = est[mask_array]
        ref = ref[mask_array]
    est = est.astype(np.float64)
    ref = ref.astype(np.float64)
    dynamic_range = float(np.max(ref) - np.min(ref) + 1e-12)
    return float(np.sqrt(np.mean((est - ref) ** 2)) / dynamic_range)


def _phase_centered(value: Any) -> np.ndarray:
    phase = np.angle(np.asarray(value))
    return phase - float(np.mean(phase))


def _magnitude(value: Any) -> np.ndarray:
    return np.abs(np.asarray(value))


def _resolve_key(inputs: dict[str, Any], key: str) -> Any:
    if key not in inputs:
        raise KeyError(f"metric input not found: {key}")
    return inputs[key]


def _mean_of_pairs(recipe: dict[str, Any], inputs: dict[str, Any]) -> float:
    base_op = str(recipe.get("base_op", "") or "")
    values = []
    for pair in list(recipe.get("pairs", []) or []):
        estimate = _resolve_key(inputs, str(pair["estimate"]))
        reference = _resolve_key(inputs, str(pair["reference"]))
        if base_op == "cosine_ncc":
            values.append(_cosine_ncc(estimate, reference, centered=False))
        elif base_op == "centered_ncc":
            values.append(_cosine_ncc(estimate, reference, centered=True))
        elif base_op == "nrmse":
            values.append(_nrmse(estimate, reference))
        else:
            raise KeyError(f"unsupported pair base_op: {base_op}")
    return float(np.mean(values))


def _vector_metric(recipe: dict[str, Any], inputs: dict[str, Any]) -> float:
    estimate = np.asarray([float(np.asarray(_resolve_key(inputs, key)).reshape(-1)[0]) for key in recipe["estimate_keys"]], dtype=np.float64)
    reference = np.asarray([float(np.asarray(_resolve_key(inputs, key)).reshape(-1)[0]) for key in recipe["reference_keys"]], dtype=np.float64)
    base_op = str(recipe.get("base_op", "") or "")
    if base_op == "cosine_ncc":
        return _cosine_ncc(estimate, reference, centered=False)
    if base_op == "nrmse":
        return _nrmse(estimate, reference)
    raise KeyError(f"unsupported vector base_op: {base_op}")


def _framewise_mean(recipe: dict[str, Any], inputs: dict[str, Any]) -> float:
    estimate = np.asarray(_resolve_key(inputs, "estimate"))
    reference = np.asarray(_resolve_key(inputs, "reference"))
    values = []
    for index in range(int(estimate.shape[0])):
        if str(recipe.get("base_op", "") or "") == "centered_ncc":
            values.append(_cosine_ncc(estimate[index], reference[index], centered=True))
        else:
            values.append(_nrmse(estimate[index], reference[index]))
    return float(np.mean(values))


def _weighted_nrmse_per_channel(estimate: np.ndarray, reference: np.ndarray, lat_weight_matrix: np.ndarray) -> np.ndarray:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    lat_weight_matrix = np.asarray(lat_weight_matrix, dtype=np.float64)
    if lat_weight_matrix.ndim == 4 and lat_weight_matrix.shape[0] == 1:
        lat_weight_matrix = lat_weight_matrix[0]
    errors = np.empty(estimate.shape[1], dtype=np.float64)
    for channel in range(estimate.shape[1]):
        est = estimate[:, channel]
        ref = reference[:, channel]
        weights = lat_weight_matrix[channel]
        numerator = np.linalg.norm(((est - ref) * weights).reshape(-1))
        denominator = np.linalg.norm((ref * weights).reshape(-1)) + 1e-12
        errors[channel] = numerator / denominator
    return errors


def _channel_mean_cosine_ncc(inputs: dict[str, Any]) -> float:
    estimate = np.asarray(_resolve_key(inputs, "estimate"))
    reference = np.asarray(_resolve_key(inputs, "reference"))
    values = []
    for channel in range(int(estimate.shape[1])):
        values.append(_cosine_ncc(estimate[:, channel], reference[:, channel], centered=False))
    return float(np.mean(values))


def _channel_mean_nrmse(inputs: dict[str, Any]) -> float:
    estimate = np.asarray(_resolve_key(inputs, "estimate"))
    reference = np.asarray(_resolve_key(inputs, "reference"))
    values = []
    for channel in range(int(estimate.shape[1])):
        values.append(_nrmse(estimate[:, channel], reference[:, channel]))
    return float(np.mean(values))


def _exoplanet_snr(inputs: dict[str, Any]) -> float:
    image = np.asarray(_resolve_key(inputs, "image"), dtype=np.float64)
    planet_x = float(np.asarray(_resolve_key(inputs, "planet_x")).reshape(-1)[0])
    planet_y = float(np.asarray(_resolve_key(inputs, "planet_y")).reshape(-1)[0])
    fwhm = float(np.asarray(_resolve_key(inputs, "fwhm")).reshape(-1)[0])
    exclude_nearest = int(np.asarray(_resolve_key(inputs, "exclude_nearest")).reshape(-1)[0])
    center_x = (image.shape[1] - 1) / 2.0
    center_y = (image.shape[0] - 1) / 2.0
    dx = planet_x - center_x
    dy = planet_y - center_y
    r_px = float(np.sqrt(dx * dx + dy * dy))
    pa_deg = float(np.degrees(np.arctan2(dy, dx)))
    n_ap = max(1, int(2 * np.pi * r_px / max(fwhm, 1e-6)))
    start = np.deg2rad(pa_deg + 90.0)
    delta = 2 * np.pi / n_ap
    values = []
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    radius = fwhm / 2.0
    for aperture_index in range(n_ap):
        if 0 < aperture_index < 1 + exclude_nearest:
            continue
        if aperture_index >= n_ap - exclude_nearest:
            continue
        theta = start + aperture_index * delta
        cx = center_x + r_px * np.cos(theta)
        cy = center_y + r_px * np.sin(theta)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        values.append(float(np.nanmedian(image[mask])))
    if len(values) < 2:
        return float(values[0] if values else 0.0)
    signal = values[0]
    noise = np.asarray(values[1:], dtype=np.float64)
    if noise.size < 2:
        return float(signal / (np.std(noise) + 1e-12))
    noise_mean = float(np.mean(noise))
    noise_std = float(np.std(noise, ddof=1) + 1e-12)
    return float((signal - noise_mean) / (noise_std * np.sqrt(1.0 + 1.0 / noise.size)))


def _laplacian_variance(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float64)
    lap = (
        -4.0 * image
        + np.roll(image, 1, axis=0)
        + np.roll(image, -1, axis=0)
        + np.roll(image, 1, axis=1)
        + np.roll(image, -1, axis=1)
    )
    return float(np.var(lap))


def _microscope_noise_reduction_factor(inputs: dict[str, Any]) -> float:
    noisy = np.asarray(_resolve_key(inputs, "noisy"), dtype=np.float64)
    denoised = np.asarray(_resolve_key(inputs, "denoised"), dtype=np.float64)

    def _bg_std(image: np.ndarray) -> float:
        block_scores = []
        for row in range(0, image.shape[0] - 16, 16):
            for col in range(0, image.shape[1] - 16, 16):
                block_scores.append((float(np.mean(image[row : row + 16, col : col + 16])), row, col))
        _, row, col = min(block_scores, key=lambda item: item[0])
        return float(np.std(image[row : row + 32, col : col + 32]))

    return float(_bg_std(noisy) / (_bg_std(denoised) + 1e-6))


def _microscope_sharpness_improvement_factor(inputs: dict[str, Any]) -> float:
    denoised = np.asarray(_resolve_key(inputs, "denoised"), dtype=np.float64)
    deconvolved = np.asarray(_resolve_key(inputs, "deconvolved"), dtype=np.float64)
    return float(_laplacian_variance(deconvolved) / (_laplacian_variance(denoised) + 1e-6))


def _lucky_sharpness_ratio(inputs: dict[str, Any]) -> float:
    stacked = np.asarray(_resolve_key(inputs, "stacked"), dtype=np.float64)
    best_frame = np.asarray(_resolve_key(inputs, "best_frame"), dtype=np.float64)
    if stacked.ndim == 3:
        stacked = np.mean(stacked, axis=-1)
    if best_frame.ndim == 3:
        best_frame = np.mean(best_frame, axis=-1)
    return float(_laplacian_variance(stacked) / (_laplacian_variance(best_frame) + 1e-6))


def _plane_wave_psf_fwhm_mean(inputs: dict[str, Any]) -> float:
    bmode = np.asarray(_resolve_key(inputs, "bmode"), dtype=np.float64)
    x = np.asarray(_resolve_key(inputs, "x"), dtype=np.float64)
    z = np.asarray(_resolve_key(inputs, "z"), dtype=np.float64)
    z_targets = np.asarray(_resolve_key(inputs, "z_targets"), dtype=np.float64).reshape(-1)
    values = []
    for target in z_targets:
        if not (float(np.min(z)) <= float(target) <= float(np.max(z))):
            continue
        row_index = int(np.argmin(np.abs(z - target)))
        row = bmode[row_index]
        peak_index = int(np.argmax(row))
        half_max = float(row[peak_index] / 2.0)
        left = peak_index
        while left > 0 and float(row[left]) >= half_max:
            left -= 1
        right = peak_index
        while right < len(row) - 1 and float(row[right]) >= half_max:
            right += 1
        values.append(float((x[right] - x[left]) * 1e3))
    return float(np.mean(values))


def _plane_wave_cnr_mean(inputs: dict[str, Any]) -> float:
    bmode = np.asarray(_resolve_key(inputs, "bmode"), dtype=np.float64)
    x = np.asarray(_resolve_key(inputs, "x"), dtype=np.float64)
    z = np.asarray(_resolve_key(inputs, "z"), dtype=np.float64)
    cyst_centers = np.asarray(_resolve_key(inputs, "cyst_centers"), dtype=np.float64)
    xx, zz = np.meshgrid(x, z)
    values = []
    for center_x, center_z in cyst_centers:
        dist = np.sqrt((xx - center_x) ** 2 + (zz - center_z) ** 2)
        inside = dist <= 2e-3
        outside = (dist > 3e-3) & (dist <= 5e-3)
        mu_in = float(np.mean(bmode[inside]))
        mu_out = float(np.mean(bmode[outside]))
        sigma_out = float(np.std(bmode[outside]) + np.spacing(1))
        values.append(abs(mu_in - mu_out) / sigma_out)
    return float(np.mean(values))


def compute_metric(metric_name: str, inputs: dict[str, Any]) -> float:
    recipe = dict(METRIC_RECIPES[metric_name])
    op = str(recipe.get("op", "") or "")
    if op == "cosine_ncc":
        return _cosine_ncc(_resolve_key(inputs, "estimate"), _resolve_key(inputs, "reference"), centered=False, mask=inputs.get(recipe.get("mask_key", "")))
    if op == "centered_ncc":
        return _cosine_ncc(_resolve_key(inputs, "estimate"), _resolve_key(inputs, "reference"), centered=True, mask=inputs.get(recipe.get("mask_key", "")))
    if op == "nrmse":
        return _nrmse(_resolve_key(inputs, "estimate"), _resolve_key(inputs, "reference"), mask=inputs.get(recipe.get("mask_key", "")))
    if op == "magnitude_cosine_ncc":
        return _cosine_ncc(_magnitude(_resolve_key(inputs, "estimate")), _magnitude(_resolve_key(inputs, "reference")), centered=False)
    if op == "magnitude_nrmse":
        return _nrmse(_magnitude(_resolve_key(inputs, "estimate")), _magnitude(_resolve_key(inputs, "reference")))
    if op == "phase_centered_ncc":
        return _cosine_ncc(_phase_centered(_resolve_key(inputs, "estimate")), _phase_centered(_resolve_key(inputs, "reference")), centered=False)
    if op == "phase_centered_nrmse":
        return _nrmse(_phase_centered(_resolve_key(inputs, "estimate")), _phase_centered(_resolve_key(inputs, "reference")))
    if op == "mean_of_pairs":
        return _mean_of_pairs(recipe, inputs)
    if op == "vector_metric":
        return _vector_metric(recipe, inputs)
    if op == "mean_absolute_error":
        estimate = np.asarray(_resolve_key(inputs, "estimate"), dtype=np.float64)
        reference = np.asarray(_resolve_key(inputs, "reference"), dtype=np.float64)
        return float(np.mean(np.abs(estimate - reference)))
    if op == "framewise_mean":
        return _framewise_mean(recipe, inputs)
    if op == "channel_mean_cosine_ncc":
        return _channel_mean_cosine_ncc(inputs)
    if op == "channel_mean_nrmse":
        return _channel_mean_nrmse(inputs)
    if op == "weighted_nrmse_mean":
        values = _weighted_nrmse_per_channel(_resolve_key(inputs, "estimate"), _resolve_key(inputs, "reference"), _resolve_key(inputs, "lat_weight_matrix"))
        return float(np.mean(values))
    if op == "weighted_nrmse_channel":
        values = _weighted_nrmse_per_channel(_resolve_key(inputs, "estimate"), _resolve_key(inputs, "reference"), _resolve_key(inputs, "lat_weight_matrix"))
        return float(values[int(recipe["channel"])])
    if op == "exoplanet_snr":
        return _exoplanet_snr(inputs)
    if op == "lucky_sharpness_ratio":
        return _lucky_sharpness_ratio(inputs)
    if op == "microscope_noise_reduction_factor":
        return _microscope_noise_reduction_factor(inputs)
    if op == "microscope_sharpness_improvement_factor":
        return _microscope_sharpness_improvement_factor(inputs)
    if op == "plane_wave_psf_fwhm_mean":
        return _plane_wave_psf_fwhm_mean(inputs)
    if op == "plane_wave_cnr_mean":
        return _plane_wave_cnr_mean(inputs)
    if op == "identity_scalar":
        return float(np.asarray(_resolve_key(inputs, str(recipe.get("key", "value")))).reshape(-1)[0])
    raise KeyError(f"unsupported metric op: {op}")
"""


def _write_metric_adapter(task_root: Path, recipes: Mapping[str, Any]) -> Path:
    adapter_path = task_root / "evaluation" / "task_metric_adapter.py"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = METRIC_ADAPTER_TEMPLATE.replace(
        "{recipes_literal}",
        json.dumps(_jsonable(recipes), indent=2, ensure_ascii=False),
    )
    adapter_path.write_text(
        rendered,
        encoding="utf-8",
    )
    return adapter_path


def build_task_contract(task_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task_root = Path(task_root).resolve()
    task_id = task_root.name
    if task_id in SPECIAL_CONTRACT_BUILDERS:
        return SPECIAL_CONTRACT_BUILDERS[task_id](task_root, task_id)
    return _simple_single_contract(task_root, task_id)


def _write_auxiliary_reference_files(task_root: Path) -> None:
    task_root = Path(task_root).resolve()
    if task_root.name == "eht_black_hole_dynamic":
        source_path = task_root / "evaluation" / "reference_outputs" / "ground_truth.npy"
        reference = np.asarray(np.load(source_path, allow_pickle=False))
        output_path = task_root / "evaluation" / "reference_outputs" / "task_contract_refs_ground_truth.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, reference=reference.astype(reference.dtype))
    if task_root.name == "era5_tensorvar":
        source_path = task_root / "evaluation" / "reference_outputs" / "trajectory.npy"
        reference = np.asarray(np.load(source_path, allow_pickle=False))
        output_path = task_root / "evaluation" / "reference_outputs" / "task_contract_refs_trajectory.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, reference=reference.astype(reference.dtype))
    if task_root.name == "lucky_imaging":
        source_path = task_root / "evaluation" / "reference_outputs" / "best_frame.npy"
        best_frame = np.asarray(np.load(source_path, allow_pickle=False))
        output_path = task_root / "evaluation" / "reference_outputs" / "task_contract_refs_best_frame.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, best_frame=best_frame.astype(best_frame.dtype))
    if task_root.name == "plane_wave_ultrasound":
        fibers_x = np.asarray(np.load(task_root / "evaluation" / "reference_outputs" / "x_fibers.npy", allow_pickle=False))
        fibers_z = np.asarray(np.load(task_root / "evaluation" / "reference_outputs" / "z_fibers.npy", allow_pickle=False))
        cysts_x = np.asarray(np.load(task_root / "evaluation" / "reference_outputs" / "x_cysts.npy", allow_pickle=False))
        cysts_z = np.asarray(np.load(task_root / "evaluation" / "reference_outputs" / "z_cysts.npy", allow_pickle=False))
        fibers_path = task_root / "evaluation" / "reference_outputs" / "task_contract_refs_fibers_axes.npz"
        cysts_path = task_root / "evaluation" / "reference_outputs" / "task_contract_refs_cysts_axes.npz"
        fibers_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fibers_path, x=fibers_x.astype(fibers_x.dtype), z=fibers_z.astype(fibers_z.dtype))
        np.savez_compressed(cysts_path, x=cysts_x.astype(cysts_x.dtype), z=cysts_z.astype(cysts_z.dtype))
    if task_root.name == "weather_radar_data_assimilation":
        source_path = task_root / "data" / "ground_truth.npz"
        with np.load(source_path, allow_pickle=False) as payload:
            target_frames = np.asarray(payload["target_frames"])
        output_path = task_root / "evaluation" / "reference_outputs" / "task_contract_refs_target_frames.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            frame_0=target_frames[0, 0].astype(target_frames.dtype),
            frame_1=target_frames[0, 1].astype(target_frames.dtype),
            frame_2=target_frames[0, 2].astype(target_frames.dtype),
        )
    if task_root.name == "shack-hartmann":
        source_path = task_root / "data" / "ground_truth.npz"
        with np.load(source_path, allow_pickle=False) as payload:
            wavefront_phases = np.asarray(payload["wavefront_phases"])
        output_path = task_root / "evaluation" / "reference_outputs" / "task_contract_refs_wavefront_levels.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            level_0=wavefront_phases[0, 0].astype(wavefront_phases.dtype),
            level_1=wavefront_phases[0, 1].astype(wavefront_phases.dtype),
            level_2=wavefront_phases[0, 2].astype(wavefront_phases.dtype),
            level_3=wavefront_phases[0, 3].astype(wavefront_phases.dtype),
        )


def write_task_contract(task_root: Path) -> tuple[Path, Path]:
    task_root = Path(task_root).resolve()
    _write_auxiliary_reference_files(task_root)
    contract, recipes = build_task_contract(task_root)
    evaluation_dir = task_root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    contract_path = evaluation_dir / "task_contract.json"
    contract_path.write_text(
        json.dumps(_jsonable(contract), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    adapter_path = _write_metric_adapter(task_root, recipes)
    return contract_path, adapter_path


def migrate_task(task_root: Path, *, output_root: Path) -> dict[str, Any]:
    task_root = Path(task_root).resolve()
    contract_path, adapter_path = write_task_contract(task_root)
    registration_result = register_task(task_root, output_root=output_root)
    return {
        "task_id": task_root.name,
        "task_contract_path": str(contract_path),
        "metric_adapter_path": str(adapter_path),
        "manifest_path": str(registration_result.manifest_path),
        "judge_path": str(registration_result.judge_path),
        "warnings": list(registration_result.warnings),
        "missing_items": list(registration_result.missing_items),
    }


def migrate_bootstrap_tasks(
    *,
    tasks_root: Path,
    output_root: Path,
    task_ids: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    tasks_root = Path(tasks_root).resolve()
    output_root = _resolve_output_root(output_root)
    requested = set(task_ids or [])
    results: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        if task_dir.name in FORMALIZED_TASK_IDS:
            continue
        if requested and task_dir.name not in requested:
            continue
        results.append(migrate_task(task_dir, output_root=output_root))
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate formal task_contract.json files for bootstrap tasks and register them."
    )
    parser.add_argument("--tasks-root", default="tasks", help="Root directory containing task folders.")
    parser.add_argument("--output-root", default="", help="Project root for registry output (defaults to cwd).")
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Optional task id to migrate. Can be passed multiple times.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_root = Path(args.output_root).resolve() if args.output_root else Path.cwd().resolve()
    results = migrate_bootstrap_tasks(
        tasks_root=Path(args.tasks_root),
        output_root=output_root,
        task_ids=args.task_id or None,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
