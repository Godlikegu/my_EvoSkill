"""Canonical task contract helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .judging import MetricRequirement
from .resource_probe import (
    ResourceProbeError,
    apply_shape_selectors,
    apply_value_selectors,
    load_resource_payload,
    normalize_shape_spec,
    shape_list_for_value,
)

TASK_CONTRACT_FILENAME = "task_contract.json"
PUBLIC_TASK_CONTRACT_FILENAME = "task_contract.public.json"

ALLOWED_VISIBILITIES = {"public", "private"}
ALLOWED_METRIC_GOALS = {"maximize", "minimize"}
ALLOWED_INPUT_SOURCES = {"output", "file", "value"}
ALLOWED_PREPROCESS = {"identity", "abs", "angle", "real", "imag"}
ALLOWED_HELPER_INTERFACES = {"python_callable", "metric_adapter", "mapping_adapter", "builtin"}
ALLOWED_HELPER_INVOCATION_MODES = {"metric_name_and_inputs", "kwargs", "mapping"}
ALLOWED_HELPER_RESULT_MODES = {"scalar", "mapping_key"}


def task_contract_paths(task_root: Path) -> dict[str, Path]:
    evaluation_dir = Path(task_root).resolve() / "evaluation"
    return {
        "evaluation_dir": evaluation_dir,
        "contract_path": evaluation_dir / TASK_CONTRACT_FILENAME,
        "public_contract_path": evaluation_dir / PUBLIC_TASK_CONTRACT_FILENAME,
    }


def load_task_contract(
    task_root: Path,
    *,
    public: bool = False,
) -> dict[str, Any]:
    paths = task_contract_paths(task_root)
    path = paths["public_contract_path"] if public else paths["contract_path"]
    if not path.exists():
        raise FileNotFoundError(f"task contract not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def task_contract_path_from_manifest(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("task_contract_path") or f"evaluation/{TASK_CONTRACT_FILENAME}")


def load_task_contract_from_manifest(
    task_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    relative_path = task_contract_path_from_manifest(manifest)
    path = Path(task_root).resolve() / relative_path
    if not path.exists():
        raise FileNotFoundError(f"task contract not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def public_task_contract_path_from_manifest(task_spec: Mapping[str, Any]) -> str:
    return str(task_spec.get("task_contract_public_path") or PUBLIC_TASK_CONTRACT_FILENAME)


def load_public_task_contract_from_root(
    root: Path,
    task_spec: Mapping[str, Any],
) -> dict[str, Any]:
    relative_path = public_task_contract_path_from_manifest(task_spec)
    path = Path(root).resolve() / relative_path
    if not path.exists():
        raise FileNotFoundError(f"public task contract not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_slice_selector(raw_value: Any) -> list[int | None] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes, bytearray)):
        raise ValueError("selectors.slice must be an array")
    values = list(raw_value)
    if len(values) > 3:
        raise ValueError("selectors.slice must have at most 3 items")
    normalized: list[int | None] = []
    for index, item in enumerate(values):
        if item is None:
            normalized.append(None)
            continue
        if isinstance(item, bool):
            raise ValueError(f"selectors.slice[{index}] must be an integer or null")
        normalized.append(int(item))
    while len(normalized) < 3:
        normalized.append(None)
    return normalized


def _normalize_selectors(raw_selectors: Mapping[str, Any] | None) -> dict[str, Any]:
    selectors = dict(raw_selectors or {})
    normalized: dict[str, Any] = {}
    if "index" in selectors and selectors.get("index") is not None:
        raw_index = selectors.get("index")
        if isinstance(raw_index, bool):
            raise ValueError("selectors.index must be an integer")
        normalized["index"] = int(raw_index)
    slice_selector = normalize_slice_selector(selectors.get("slice"))
    if slice_selector is not None:
        normalized["slice"] = slice_selector
    if "squeeze" in selectors:
        normalized["squeeze"] = bool(selectors.get("squeeze"))
    return normalized


def _looks_like_binding_spec(node: Any) -> bool:
    if not isinstance(node, Mapping):
        return False
    return any(
        key in node
        for key in ("source", "field", "file_id", "value", "selectors", "preprocess", "expected_shape")
    )


def _looks_like_binding_map(node: Any) -> bool:
    if not isinstance(node, Mapping) or not node:
        return False
    return all(_looks_like_binding_spec(value) or _looks_like_binding_map(value) for value in node.values())


def _apply_slice_value(value: Any, slice_selector: Sequence[int | None] | None) -> Any:
    if slice_selector is None:
        return value
    start, stop, step = list(slice_selector)[:3]
    return np.asarray(value)[slice(start, stop, step)]


def _apply_slice_shape(shape: Sequence[int], slice_selector: Sequence[int | None] | None) -> list[int]:
    normalized = normalize_shape_spec(shape)
    if slice_selector is None:
        return normalized
    if not normalized:
        raise ResourceProbeError("selectors.slice cannot be applied to a scalar shape")
    start, stop, step = list(slice_selector)[:3]
    indices = list(range(normalized[0]))[slice(start, stop, step)]
    return [len(indices), *normalized[1:]]


def _apply_preprocess(value: Any, preprocess: str) -> Any:
    mode = str(preprocess or "identity").strip().lower() or "identity"
    array = np.asarray(value)
    if mode == "identity":
        return array
    if mode == "abs":
        return np.abs(array)
    if mode == "angle":
        return np.angle(array)
    if mode == "real":
        return np.real(array)
    if mode == "imag":
        return np.imag(array)
    raise ValueError(f"unsupported preprocess: {preprocess!r}")


def _apply_preprocess_shape(shape: Sequence[int], preprocess: str) -> list[int]:
    mode = str(preprocess or "identity").strip().lower() or "identity"
    if mode not in ALLOWED_PREPROCESS:
        raise ValueError(f"unsupported preprocess: {preprocess!r}")
    return normalize_shape_spec(shape)


def file_map(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw_file in contract.get("files", []) or []:
        if not isinstance(raw_file, Mapping):
            continue
        file_id = str(raw_file.get("id", "") or "").strip()
        if file_id:
            result[file_id] = raw_file
    return result


def output_field_map(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    output = dict(contract.get("output") or {})
    for raw_field in output.get("fields", []) or []:
        if not isinstance(raw_field, Mapping):
            continue
        name = str(raw_field.get("name", "") or "").strip()
        if name:
            result[name] = raw_field
    return result


def metric_map(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw_metric in contract.get("metrics", []) or []:
        if not isinstance(raw_metric, Mapping):
            continue
        name = str(raw_metric.get("name", "") or "").strip()
        if name:
            result[name] = raw_metric
    return result


def task_contract_primary_output_path(contract: Mapping[str, Any]) -> str:
    output = dict(contract.get("output") or {})
    return str(output.get("path") or "output/reconstruction.npz")


def task_contract_output_format(contract: Mapping[str, Any]) -> str:
    output = dict(contract.get("output") or {})
    return str(output.get("format") or "npz").strip().lower() or "npz"


def task_contract_execution(contract: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(contract.get("execution") or {})
    return {
        "read_first": [str(item) for item in execution.get("read_first", []) or [] if str(item)],
        "readable_files": [
            str(item) for item in execution.get("readable_files", []) or [] if str(item)
        ],
        "entrypoint": str(execution.get("entrypoint") or "work/main.py"),
        "writable_paths": [
            str(item) for item in execution.get("writable_paths", []) or [] if str(item)
        ] or ["work/", "output/", "checkpoints/"],
    }


def normalize_metric_helper(helper: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(helper or {})
    interface = str(raw.get("interface", "python_callable") or "python_callable").strip()
    if interface == "builtin":
        return {
            "interface": "builtin",
            "builtin": str(raw.get("builtin", "") or "").strip(),
        }
    if interface == "metric_adapter":
        return {
            "interface": "python_callable",
            "file_id": str(raw.get("file_id", "") or "").strip(),
            "callable": str(raw.get("callable", "") or "").strip(),
            "invocation": {"mode": "metric_name_and_inputs"},
            "result": {"mode": "scalar"},
        }
    if interface == "mapping_adapter":
        return {
            "interface": "python_callable",
            "file_id": str(raw.get("file_id", "") or "").strip(),
            "callable": str(raw.get("callable", "") or "").strip(),
            "invocation": {"mode": "kwargs"},
            "result": {
                "mode": "mapping_key",
                "key": str(raw.get("result_key", "") or "").strip(),
            },
        }
    invocation = dict(raw.get("invocation") or {})
    result = dict(raw.get("result") or {})
    return {
        "interface": "python_callable",
        "file_id": str(raw.get("file_id", "") or "").strip(),
        "callable": str(raw.get("callable", "") or "").strip(),
        "invocation": {
            "mode": str(
                invocation.get("mode", "metric_name_and_inputs") or "metric_name_and_inputs"
            ).strip()
        },
        "result": {
            "mode": str(result.get("mode", "scalar") or "scalar").strip(),
            "key": str(
                result.get("key", "") or result.get("result_key", "") or raw.get("result_key", "")
            ).strip(),
        },
    }


def validate_task_contract(
    contract: Mapping[str, Any],
    *,
    require_private_paths: bool = True,
) -> list[str]:
    errors: list[str] = []
    payload = dict(contract or {})
    for field_name in ("version", "task_id", "family", "files", "execution", "output", "metrics"):
        if field_name not in payload:
            errors.append(f"missing required field: {field_name}")

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        errors.append("files must be a non-empty list")
    else:
        seen_file_ids: set[str] = set()
        for index, raw_file in enumerate(files):
            file_payload = raw_file if isinstance(raw_file, Mapping) else {}
            prefix = f"files[{index}]"
            file_id = str(file_payload.get("id", "") or "").strip()
            path = str(file_payload.get("path", "") or "").strip()
            visibility = str(file_payload.get("visibility", "") or "").strip()
            role = str(file_payload.get("role", "") or "").strip()
            semantics = str(file_payload.get("semantics", "") or "").strip()
            if not file_id:
                errors.append(f"{prefix}.id is required")
            elif file_id in seen_file_ids:
                errors.append(f"{prefix}.id must be unique")
            else:
                seen_file_ids.add(file_id)
            if not path:
                errors.append(f"{prefix}.path is required")
            elif Path(path).is_absolute():
                errors.append(f"{prefix}.path must be relative")
            if visibility not in ALLOWED_VISIBILITIES:
                errors.append(f"{prefix}.visibility must be one of: {sorted(ALLOWED_VISIBILITIES)}")
            if not role:
                errors.append(f"{prefix}.role is required")
            if not semantics:
                errors.append(f"{prefix}.semantics is required")
            fields_payload = file_payload.get("fields")
            if fields_payload is not None and not isinstance(fields_payload, Mapping):
                errors.append(f"{prefix}.fields must be an object when provided")
            for field_name, raw_field in dict(fields_payload or {}).items():
                field_prefix = f"{prefix}.fields[{field_name!r}]"
                field = raw_field if isinstance(raw_field, Mapping) else {}
                dtype = str(field.get("dtype", "") or "").strip()
                if not dtype:
                    errors.append(f"{field_prefix}.dtype is required")
                try:
                    normalize_shape_spec(field.get("shape"))
                except Exception as exc:
                    errors.append(f"{field_prefix}.shape is invalid ({exc})")

    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("execution must be an object")
    else:
        if not str(execution.get("entrypoint", "") or "").strip():
            errors.append("execution.entrypoint is required")
        for list_field in ("read_first", "readable_files", "writable_paths"):
            value = execution.get(list_field)
            if value is None:
                continue
            if not isinstance(value, list):
                errors.append(f"execution.{list_field} must be a list")

    output = payload.get("output")
    if not isinstance(output, Mapping):
        errors.append("output must be an object")
    else:
        output_path = str(output.get("path", "") or "").strip()
        output_format = str(output.get("format", "") or "").strip()
        output_fields = output.get("fields")
        if not output_path:
            errors.append("output.path is required")
        elif Path(output_path).is_absolute():
            errors.append("output.path must be relative")
        if not output_format:
            errors.append("output.format is required")
        if not isinstance(output_fields, list) or not output_fields:
            errors.append("output.fields must be a non-empty list")
        else:
            seen_output_fields: set[str] = set()
            for index, raw_field in enumerate(output_fields):
                field = raw_field if isinstance(raw_field, Mapping) else {}
                prefix = f"output.fields[{index}]"
                name = str(field.get("name", "") or "").strip()
                dtype = str(field.get("dtype", "") or "").strip()
                semantics = str(field.get("semantics", "") or "").strip()
                if not name:
                    errors.append(f"{prefix}.name is required")
                elif name in seen_output_fields:
                    errors.append(f"{prefix}.name must be unique")
                else:
                    seen_output_fields.add(name)
                if not dtype:
                    errors.append(f"{prefix}.dtype is required")
                if not semantics:
                    errors.append(f"{prefix}.semantics is required")
                try:
                    normalize_shape_spec(field.get("shape"))
                except Exception as exc:
                    errors.append(f"{prefix}.shape is invalid ({exc})")

    files_by_id = file_map(payload)
    output_fields_by_name = output_field_map(payload)
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append("metrics must be a non-empty list")
    else:
        seen_metric_names: set[str] = set()
        for index, raw_metric in enumerate(metrics):
            metric = raw_metric if isinstance(raw_metric, Mapping) else {}
            prefix = f"metrics[{index}]"
            name = str(metric.get("name", "") or "").strip()
            goal = str(metric.get("goal", "") or "").strip()
            helper = metric.get("helper")
            inputs = metric.get("inputs")
            threshold = metric.get("threshold")
            if not name:
                errors.append(f"{prefix}.name is required")
            elif name in seen_metric_names:
                errors.append(f"{prefix}.name must be unique")
            else:
                seen_metric_names.add(name)
            if goal not in ALLOWED_METRIC_GOALS:
                errors.append(f"{prefix}.goal must be one of: {sorted(ALLOWED_METRIC_GOALS)}")
            if threshold is None or isinstance(threshold, bool):
                errors.append(f"{prefix}.threshold is required")
            else:
                try:
                    float(threshold)
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.threshold must be numeric")
            if not isinstance(helper, Mapping):
                errors.append(f"{prefix}.helper must be an object")
            else:
                raw_interface = str(helper.get("interface", "python_callable") or "python_callable").strip()
                normalized_helper = normalize_metric_helper(helper)
                interface = str(normalized_helper.get("interface", "") or "").strip()
                file_id = str(normalized_helper.get("file_id", "") or "").strip()
                callable_name = str(normalized_helper.get("callable", "") or "").strip()
                if raw_interface not in ALLOWED_HELPER_INTERFACES:
                    errors.append(
                        f"{prefix}.helper.interface must be one of: {sorted(ALLOWED_HELPER_INTERFACES)}"
                    )
                if interface == "builtin":
                    builtin_name = str(normalized_helper.get("builtin", "") or "").strip()
                    if not builtin_name:
                        errors.append(f"{prefix}.helper.builtin is required for builtin metrics")
                else:
                    if not file_id:
                        errors.append(f"{prefix}.helper.file_id is required")
                    elif file_id not in files_by_id and require_private_paths:
                        errors.append(f"{prefix}.helper.file_id references an unknown file id: {file_id}")
                    if not callable_name:
                        errors.append(f"{prefix}.helper.callable is required")
                    invocation_mode = str(
                        dict(normalized_helper.get("invocation") or {}).get("mode", "") or ""
                    ).strip()
                    if invocation_mode not in ALLOWED_HELPER_INVOCATION_MODES:
                        errors.append(
                            f"{prefix}.helper.invocation.mode must be one of: "
                            f"{sorted(ALLOWED_HELPER_INVOCATION_MODES)}"
                        )
                    result_mode = str(
                        dict(normalized_helper.get("result") or {}).get("mode", "") or ""
                    ).strip()
                    if result_mode not in ALLOWED_HELPER_RESULT_MODES:
                        errors.append(
                            f"{prefix}.helper.result.mode must be one of: "
                            f"{sorted(ALLOWED_HELPER_RESULT_MODES)}"
                        )
                    if result_mode == "mapping_key":
                        result_key = str(
                            dict(normalized_helper.get("result") or {}).get("key", "") or ""
                        ).strip()
                        if not result_key:
                            errors.append(
                                f"{prefix}.helper.result.key is required when "
                                "helper.result.mode == 'mapping_key'"
                            )
            if not isinstance(inputs, Mapping) or not dict(inputs):
                errors.append(f"{prefix}.inputs must be a non-empty object")
                continue
            for input_name, raw_input in dict(inputs).items():
                input_prefix = f"{prefix}.inputs[{input_name!r}]"
                input_payload = raw_input if isinstance(raw_input, Mapping) else {}
                source = str(input_payload.get("source", "") or "").strip()
                field_name = str(input_payload.get("field", "") or "").strip()
                preprocess = str(input_payload.get("preprocess", "identity") or "identity").strip()
                if source not in ALLOWED_INPUT_SOURCES:
                    errors.append(
                        f"{input_prefix}.source must be one of: {sorted(ALLOWED_INPUT_SOURCES)}"
                    )
                if source != "value" and not field_name:
                    errors.append(f"{input_prefix}.field is required")
                if preprocess not in ALLOWED_PREPROCESS:
                    errors.append(
                        f"{input_prefix}.preprocess must be one of: {sorted(ALLOWED_PREPROCESS)}"
                    )
                if not (source == "value" and _looks_like_binding_map(input_payload.get("value"))):
                    try:
                        normalize_shape_spec(input_payload.get("expected_shape"))
                    except Exception as exc:
                        errors.append(f"{input_prefix}.expected_shape is invalid ({exc})")
                try:
                    _normalize_selectors(input_payload.get("selectors"))
                except Exception as exc:
                    errors.append(f"{input_prefix}.selectors is invalid ({exc})")
                if source == "file":
                    file_id = str(input_payload.get("file_id", "") or "").strip()
                    if not file_id:
                        errors.append(f"{input_prefix}.file_id is required for file inputs")
                    elif file_id not in files_by_id and require_private_paths:
                        errors.append(f"{input_prefix}.file_id references an unknown file id: {file_id}")
                if source == "output" and field_name and field_name not in output_fields_by_name:
                    errors.append(
                        f"{input_prefix}.field references an unknown output field: {field_name}"
                    )
                if source == "value" and "value" not in input_payload:
                    errors.append(f"{input_prefix}.value is required for value inputs")
    return sorted(dict.fromkeys(errors))


def validate_task_contract_task_paths(task_root: Path, contract: Mapping[str, Any]) -> list[str]:
    task_root = Path(task_root).resolve()
    errors: list[str] = []
    for raw_file in contract.get("files", []) or []:
        if not isinstance(raw_file, Mapping):
            continue
        file_id = str(raw_file.get("id", "") or "").strip()
        path_value = str(raw_file.get("path", "") or "").strip()
        if not path_value:
            continue
        candidate = task_root / path_value
        if not candidate.exists():
            errors.append(f"files[{file_id!r}].path does not exist: {path_value}")
        elif not candidate.is_file():
            errors.append(f"files[{file_id!r}].path must be a file: {path_value}")
    output_path = str((contract.get("output") or {}).get("path", "") or "").strip()
    if output_path and Path(output_path).is_absolute():
        errors.append("output.path must be relative")
    return sorted(dict.fromkeys(errors))


def _resolve_file_field_shape(file_entry: Mapping[str, Any], field_name: str) -> list[int]:
    fields_payload = dict(file_entry.get("fields") or {})
    if field_name not in fields_payload:
        raise KeyError(f"field {field_name!r} not declared in file {file_entry.get('id', '')!r}")
    return normalize_shape_spec(dict(fields_payload[field_name]).get("shape"))


def _resolve_output_field_shape(contract: Mapping[str, Any], field_name: str) -> list[int]:
    output_fields = output_field_map(contract)
    if field_name not in output_fields:
        raise KeyError(f"unknown output field: {field_name}")
    return normalize_shape_spec(dict(output_fields[field_name]).get("shape"))


def _binding_base_shape(contract: Mapping[str, Any], binding: Mapping[str, Any]) -> list[int]:
    source = str(binding.get("source", "") or "").strip()
    field_name = str(binding.get("field", "") or "").strip()
    if source == "output":
        return _resolve_output_field_shape(contract, field_name)
    if source == "file":
        file_id = str(binding.get("file_id", "") or "").strip()
        files_by_id = file_map(contract)
        if file_id not in files_by_id:
            raise KeyError(f"unknown file id: {file_id}")
        return _resolve_file_field_shape(files_by_id[file_id], field_name)
    if source == "value":
        if _looks_like_binding_map(binding.get("value")):
            raise ValueError("composite value bindings do not have a scalar shape")
        return shape_list_for_value(binding.get("value"))
    raise KeyError(f"unsupported binding source: {source}")


def derive_binding_shape(contract: Mapping[str, Any], binding: Mapping[str, Any]) -> list[int]:
    base_shape = _binding_base_shape(contract, binding)
    selectors = _normalize_selectors(binding.get("selectors"))
    shape_after_slice = _apply_slice_shape(base_shape, selectors.get("slice"))
    shape_after_selectors = apply_shape_selectors(
        shape_after_slice,
        index=selectors.get("index"),
        squeeze=bool(selectors.get("squeeze", False)),
    )
    return _apply_preprocess_shape(
        shape_after_selectors,
        str(binding.get("preprocess", "identity") or "identity"),
    )


def validate_task_contract_shapes(task_root: Path, contract: Mapping[str, Any]) -> list[str]:
    task_root = Path(task_root).resolve()
    errors: list[str] = []
    for index, raw_metric in enumerate(contract.get("metrics", []) or []):
        if not isinstance(raw_metric, Mapping):
            continue
        metric_name = str(raw_metric.get("name", f"metric_{index}") or f"metric_{index}")
        for input_name, raw_input in dict(raw_metric.get("inputs") or {}).items():
            input_payload = raw_input if isinstance(raw_input, Mapping) else {}
            input_prefix = f"metrics[{metric_name!r}].inputs[{input_name!r}]"
            if str(input_payload.get("source", "") or "").strip() == "value" and _looks_like_binding_map(
                input_payload.get("value")
            ):
                continue
            try:
                expected_shape = normalize_shape_spec(input_payload.get("expected_shape"))
                derived_shape = derive_binding_shape(contract, input_payload)
            except Exception as exc:
                errors.append(f"{input_prefix} shape derivation failed ({exc})")
                continue
            if derived_shape != expected_shape:
                errors.append(
                    f"{input_prefix}.expected_shape does not match derived shape: "
                    f"declared {expected_shape}, derived {derived_shape}"
                )
            source = str(input_payload.get("source", "") or "").strip()
            if source != "file":
                continue
            file_id = str(input_payload.get("file_id", "") or "").strip()
            field_name = str(input_payload.get("field", "") or "").strip()
            file_entry = file_map(contract).get(file_id)
            if not isinstance(file_entry, Mapping):
                continue
            try:
                value = load_resource_payload(task_root, str(file_entry.get("path", "")), field=field_name)
                selectors = _normalize_selectors(input_payload.get("selectors"))
                value = _apply_slice_value(value, selectors.get("slice"))
                value = apply_value_selectors(
                    value,
                    index=selectors.get("index"),
                    squeeze=bool(selectors.get("squeeze", False)),
                )
                value = _apply_preprocess(value, str(input_payload.get("preprocess", "identity")))
            except Exception as exc:
                errors.append(f"{input_prefix} could not be resolved from file payload ({exc})")
                continue
            observed_shape = shape_list_for_value(value)
            if observed_shape != expected_shape:
                errors.append(
                    f"{input_prefix}.expected_shape does not match observed file-derived shape: "
                    f"declared {expected_shape}, observed {observed_shape}"
                )
    return sorted(dict.fromkeys(errors))


def derive_public_task_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    def _public_binding_view(
        binding: Mapping[str, Any],
        *,
        public_file_ids: set[str],
    ) -> dict[str, Any]:
        item = {
            "source": str(binding.get("source", "") or ""),
            "field": str(binding.get("field", "") or ""),
            "expected_shape": normalize_shape_spec(binding.get("expected_shape")),
            "preprocess": str(binding.get("preprocess", "identity") or "identity"),
            "selectors": dict(binding.get("selectors") or {}),
        }
        if item["source"] == "file":
            file_id = str(binding.get("file_id", "") or "")
            item["file_id"] = file_id
            item["visibility"] = "public" if file_id in public_file_ids else "private"
            if file_id in public_file_ids:
                file_entry = file_map(contract).get(file_id) or {}
                item["path"] = str(file_entry.get("path", "") or "")
        elif item["source"] == "value":
            nested_value = binding.get("value")
            if _looks_like_binding_map(nested_value):
                item["value"] = {
                    str(name): _public_binding_view(child_binding, public_file_ids=public_file_ids)
                    for name, child_binding in dict(nested_value).items()
                    if isinstance(child_binding, Mapping)
                }
            elif "value" in binding:
                item["value"] = binding.get("value")
        return item

    files_payload: list[dict[str, Any]] = []
    public_file_ids: set[str] = set()
    for raw_file in contract.get("files", []) or []:
        if not isinstance(raw_file, Mapping):
            continue
        if str(raw_file.get("visibility", "") or "") != "public":
            continue
        normalized = dict(raw_file)
        files_payload.append(normalized)
        public_file_ids.add(str(normalized.get("id", "") or ""))

    metrics_payload: list[dict[str, Any]] = []
    for raw_metric in contract.get("metrics", []) or []:
        if not isinstance(raw_metric, Mapping):
            continue
        public_metric = {
            "name": str(raw_metric.get("name", "") or ""),
            "goal": str(raw_metric.get("goal", "") or ""),
            "threshold": raw_metric.get("threshold"),
            "inputs": {},
        }
        for input_name, raw_input in dict(raw_metric.get("inputs") or {}).items():
            if not isinstance(raw_input, Mapping):
                continue
            public_metric["inputs"][str(input_name)] = _public_binding_view(
                raw_input,
                public_file_ids=public_file_ids,
            )
        metrics_payload.append(public_metric)

    execution = task_contract_execution(contract)
    execution["read_first"] = [
        item for item in execution["read_first"] if item in public_file_ids
    ]
    execution["readable_files"] = [
        item for item in execution["readable_files"] if item in public_file_ids
    ]
    return {
        "version": int(contract.get("version", 2) or 2),
        "task_id": str(contract.get("task_id", "") or ""),
        "family": str(contract.get("family", "") or ""),
        "files": files_payload,
        "execution": execution,
        "output": dict(contract.get("output") or {}),
        "metrics": metrics_payload,
    }


def output_requirements_from_contract(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = dict(contract.get("output") or {})
    fields = [dict(item or {}) for item in output.get("fields", []) or [] if isinstance(item, Mapping)]
    return [
        {
            "path": task_contract_primary_output_path(contract),
            "format": task_contract_output_format(contract),
            "fields": fields,
        }
    ]


def validate_output_payload_against_contract(
    output_payload: Any,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    missing_fields: list[str] = []
    field_checks: list[dict[str, Any]] = []
    output_fields = output_field_map(contract)
    for field_name, spec in output_fields.items():
        if field_name not in output_payload.files:
            missing_fields.append(field_name)
            continue
        value = np.asarray(output_payload[field_name])
        expected_shape = normalize_shape_spec(spec.get("shape"))
        observed_shape = list(value.shape)
        if observed_shape != expected_shape:
            warnings.append(
                f"invalid shape for field {field_name}: expected {expected_shape}, observed {observed_shape}"
            )
        expected_dtype = str(spec.get("dtype", "") or "").strip()
        observed_dtype = str(value.dtype)
        if expected_dtype and observed_dtype != expected_dtype:
            warnings.append(
                f"invalid dtype for field {field_name}: expected {expected_dtype}, observed {observed_dtype}"
            )
        if not np.issubdtype(value.dtype, np.number):
            warnings.append(f"non-numeric field: {field_name}")
        elif np.any(~np.isfinite(np.asarray(value, dtype=np.complex128).view(np.float64))):
            warnings.append(f"nan_or_inf field: {field_name}")
        field_checks.append(
            {
                "field": field_name,
                "expected_shape": expected_shape,
                "observed_shape": observed_shape,
                "expected_dtype": expected_dtype,
                "observed_dtype": observed_dtype,
            }
        )
    return {
        "missing_fields": sorted(dict.fromkeys(missing_fields)),
        "warnings": list(dict.fromkeys(warnings)),
        "field_checks": field_checks,
    }


def output_metric_input_checks(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    def _collect_checks(
        binding_name: str,
        binding: Mapping[str, Any],
        *,
        metric_name: str,
        output_path: str,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        if str(binding.get("source", "") or "").strip() == "output":
            collected.append(
                {
                    "output_path": output_path,
                    "metric_name": metric_name,
                    "input_name": binding_name,
                    "field": str(binding.get("field", "") or ""),
                    "expected_shape": normalize_shape_spec(binding.get("expected_shape")),
                    "selectors": dict(binding.get("selectors") or {}),
                    "preprocess": str(binding.get("preprocess", "identity") or "identity"),
                }
            )
        if str(binding.get("source", "") or "").strip() == "value" and _looks_like_binding_map(
            binding.get("value")
        ):
            for child_name, child_binding in dict(binding.get("value") or {}).items():
                if not isinstance(child_binding, Mapping):
                    continue
                collected.extend(
                    _collect_checks(
                        f"{binding_name}.{child_name}",
                        child_binding,
                        metric_name=metric_name,
                        output_path=output_path,
                    )
                )
        return collected

    checks: list[dict[str, Any]] = []
    output_path = task_contract_primary_output_path(contract)
    for raw_metric in contract.get("metrics", []) or []:
        if not isinstance(raw_metric, Mapping):
            continue
        metric_name = str(raw_metric.get("name", "") or "")
        for input_name, raw_input in dict(raw_metric.get("inputs") or {}).items():
            if not isinstance(raw_input, Mapping):
                continue
            checks.extend(
                _collect_checks(
                    str(input_name),
                    raw_input,
                    metric_name=metric_name,
                    output_path=output_path,
                )
            )
    return checks


def goal_to_operator(goal: str) -> str:
    normalized = str(goal or "").strip().lower()
    if normalized == "maximize":
        return ">="
    if normalized == "minimize":
        return "<="
    raise ValueError(f"unsupported metric goal: {goal!r}")


def build_metric_requirements(contract: Mapping[str, Any]) -> list[MetricRequirement]:
    requirements: list[MetricRequirement] = []
    for raw_metric in contract.get("metrics", []) or []:
        if not isinstance(raw_metric, Mapping):
            continue
        name = str(raw_metric.get("name", "") or "").strip()
        if not name:
            continue
        requirements.append(
            MetricRequirement(
                name,
                float(raw_metric.get("threshold")),
                goal_to_operator(str(raw_metric.get("goal", "") or "")),
            )
        )
    return requirements


def load_metric_callable(
    task_root: Path,
    contract: Mapping[str, Any],
    metric: Mapping[str, Any],
):
    helper = dict(metric.get("helper") or {})
    normalized_helper = normalize_metric_helper(helper)
    interface = str(normalized_helper.get("interface", "python_callable") or "python_callable").strip()
    if interface == "builtin":
        return None
    file_id = str(normalized_helper.get("file_id", "") or "").strip()
    callable_name = str(normalized_helper.get("callable", "") or "").strip()
    file_entry = file_map(contract).get(file_id)
    if not isinstance(file_entry, Mapping):
        raise KeyError(f"metric helper file not found: {file_id}")
    module_path = Path(task_root).resolve() / str(file_entry.get("path", "") or "")
    spec = importlib.util.spec_from_file_location(
        f"myevoskill_task_metric_{module_path.stem}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import metric helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    helper_callable = getattr(module, callable_name, None)
    if not callable(helper_callable):
        raise AttributeError(f"metric helper callable not found: {module_path}:{callable_name}")
    return helper_callable


def _builtin_metric(metric_name: str, helper: Mapping[str, Any], inputs: Mapping[str, Any]) -> float:
    builtin_name = str(helper.get("builtin", "") or metric_name).strip()
    estimate = np.asarray(inputs.get("estimate"))
    reference = np.asarray(inputs.get("reference"))
    if builtin_name == "ncc":
        estimate_arr = np.asarray(estimate, dtype=np.float64).ravel()
        reference_arr = np.asarray(reference, dtype=np.float64).ravel()
        denom = float(np.linalg.norm(estimate_arr) * np.linalg.norm(reference_arr) + 1e-12)
        return float(np.dot(estimate_arr, reference_arr) / denom)
    if builtin_name == "nrmse":
        estimate_arr = np.asarray(estimate, dtype=np.float64)
        reference_arr = np.asarray(reference, dtype=np.float64)
        dynamic_range = float(reference_arr.max() - reference_arr.min() + 1e-12)
        return float(np.sqrt(np.mean((estimate_arr - reference_arr) ** 2)) / dynamic_range)
    if builtin_name == "phase_ncc":
        estimate_phase = np.angle(np.asarray(estimate))
        reference_phase = np.angle(np.asarray(reference))
        estimate_phase = estimate_phase - float(np.mean(estimate_phase))
        reference_phase = reference_phase - float(np.mean(reference_phase))
        denom = float(np.linalg.norm(estimate_phase) * np.linalg.norm(reference_phase) + 1e-12)
        return float(np.sum(estimate_phase * reference_phase) / denom)
    if builtin_name == "phase_nrmse":
        estimate_phase = np.angle(np.asarray(estimate))
        reference_phase = np.angle(np.asarray(reference))
        estimate_phase = estimate_phase - float(np.mean(estimate_phase))
        reference_phase = reference_phase - float(np.mean(reference_phase))
        dynamic_range = float(reference_phase.max() - reference_phase.min() + 1e-12)
        return float(np.sqrt(np.mean((estimate_phase - reference_phase) ** 2)) / dynamic_range)
    raise KeyError(f"unsupported builtin metric: {builtin_name}")


def resolve_metric_input_value(
    task_root: Path,
    contract: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    output_payload: Any,
) -> Any:
    source = str(binding.get("source", "") or "").strip()
    field_name = str(binding.get("field", "") or "").strip()
    if source == "output":
        if output_payload is None or not hasattr(output_payload, "files"):
            raise ResourceProbeError("output payload is required for output-bound metric inputs")
        if field_name not in output_payload.files:
            raise ResourceProbeError(f"missing output field: {field_name}")
        value = np.asarray(output_payload[field_name])
    elif source == "file":
        file_id = str(binding.get("file_id", "") or "").strip()
        file_entry = file_map(contract).get(file_id)
        if not isinstance(file_entry, Mapping):
            raise ResourceProbeError(f"unknown file id: {file_id}")
        value = load_resource_payload(
            Path(task_root).resolve(),
            str(file_entry.get("path", "") or ""),
            field=field_name,
        )
    elif source == "value":
        nested_value = binding.get("value")
        if _looks_like_binding_map(nested_value):
            value = {
                str(name): resolve_metric_input_value(
                    task_root,
                    contract,
                    nested_binding,
                    output_payload=output_payload,
                )
                for name, nested_binding in dict(nested_value).items()
            }
            return value
        value = nested_value
    else:
        raise ResourceProbeError(f"unsupported metric input source: {source!r}")
    selectors = _normalize_selectors(binding.get("selectors"))
    value = _apply_slice_value(value, selectors.get("slice"))
    value = apply_value_selectors(
        value,
        index=selectors.get("index"),
        squeeze=bool(selectors.get("squeeze", False)),
    )
    value = _apply_preprocess(value, str(binding.get("preprocess", "identity") or "identity"))
    expected_shape = normalize_shape_spec(binding.get("expected_shape"))
    observed_shape = shape_list_for_value(value)
    if observed_shape != expected_shape:
        raise ResourceProbeError(
            f"metric input shape mismatch: declared {expected_shape}, observed {observed_shape}"
        )
    return value


def resolve_metric_inputs(
    task_root: Path,
    contract: Mapping[str, Any],
    metric: Mapping[str, Any],
    *,
    output_payload: Any,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for input_name, raw_binding in dict(metric.get("inputs") or {}).items():
        if not isinstance(raw_binding, Mapping):
            continue
        resolved[str(input_name)] = resolve_metric_input_value(
            task_root,
            contract,
            raw_binding,
            output_payload=output_payload,
        )
    return resolved


def evaluate_metric(
    task_root: Path,
    contract: Mapping[str, Any],
    metric: Mapping[str, Any],
    *,
    output_payload: Any,
) -> float:
    metric_name = str(metric.get("name", "") or "").strip()
    resolved_inputs = resolve_metric_inputs(
        task_root,
        contract,
        metric,
        output_payload=output_payload,
    )
    helper = dict(metric.get("helper") or {})
    normalized_helper = normalize_metric_helper(helper)
    interface = str(
        normalized_helper.get("interface", "python_callable") or "python_callable"
    ).strip()
    if interface == "builtin":
        result = _builtin_metric(metric_name, normalized_helper, resolved_inputs)
    else:
        helper_callable = load_metric_callable(task_root, contract, metric)
        invocation_mode = str(
            dict(normalized_helper.get("invocation") or {}).get("mode", "") or ""
        ).strip()
        if invocation_mode == "kwargs":
            raw_result = helper_callable(**resolved_inputs)
        elif invocation_mode == "mapping":
            raw_result = helper_callable(resolved_inputs)
        else:
            raw_result = helper_callable(metric_name, resolved_inputs)
        result_mode = str(dict(normalized_helper.get("result") or {}).get("mode", "") or "").strip()
        if result_mode == "mapping_key":
            if not isinstance(raw_result, Mapping):
                raise TypeError("metric helper must return a mapping for result.mode == 'mapping_key'")
            result_key = str(dict(normalized_helper.get("result") or {}).get("key", "") or "").strip()
            if result_key not in raw_result:
                raise KeyError(f"result key not found in metric helper result: {result_key}")
            result = raw_result[result_key]
        else:
            result = raw_result
    if isinstance(result, bool):
        raise TypeError("metric helper must return a numeric scalar, not bool")
    if isinstance(result, np.ndarray):
        if result.shape != ():
            raise TypeError("metric helper must return a scalar, not an array")
        return float(result.item())
    try:
        return float(result)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"metric helper must return a numeric scalar for {metric_name!r}"
        ) from exc
