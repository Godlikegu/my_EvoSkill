"""Helpers for probing task-local resource files and declared interface shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - optional import availability is environment-specific
    import h5py
except Exception:  # pragma: no cover - optional import availability is environment-specific
    h5py = None


_ARRAY_FILE_SUFFIXES = {".npy", ".npz"}
_HDF5_SUFFIXES = {".h5", ".hdf5"}


class ResourceProbeError(RuntimeError):
    """Raised when a task-local resource cannot be safely or consistently probed."""


def normalize_shape_spec(shape: Any) -> list[int]:
    """Normalize a declared shape specification into a JSON-friendly integer list."""

    if shape is None:
        raise ResourceProbeError("shape is required")
    if isinstance(shape, np.ndarray):
        shape = shape.tolist()
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
        raise ResourceProbeError("shape must be an array of non-negative integers")
    normalized: list[int] = []
    for index, raw_dim in enumerate(shape):
        if isinstance(raw_dim, bool):
            raise ResourceProbeError(f"shape[{index}] must be an integer, not boolean")
        try:
            dim = int(raw_dim)
        except (TypeError, ValueError) as exc:
            raise ResourceProbeError(
                f"shape[{index}] must be an integer"
            ) from exc
        if dim < 0:
            raise ResourceProbeError(f"shape[{index}] must be >= 0")
        normalized.append(dim)
    return normalized


def shape_list_for_value(value: Any) -> list[int]:
    """Return the runtime shape for one value using [] for scalars."""

    if isinstance(value, np.ndarray):
        return [int(dim) for dim in value.shape]
    if np.isscalar(value):
        return []
    array = np.asarray(value)
    return [int(dim) for dim in array.shape]


def dtype_name_for_value(value: Any) -> str:
    """Return a stable dtype/type label for one probed value."""

    if isinstance(value, np.ndarray):
        return str(value.dtype)
    if np.isscalar(value):
        return type(value).__name__
    array = np.asarray(value)
    return str(array.dtype)


def descriptor_for_value(*, path: str, field: str, value: Any) -> dict[str, Any]:
    """Build one normalized shape/dtype descriptor payload."""

    return {
        "path": str(path).replace("\\", "/"),
        "field": str(field or ""),
        "observed_dtype": dtype_name_for_value(value),
        "observed_shape": shape_list_for_value(value),
    }


def resolve_task_local_path(task_root: Path, resource_path: str) -> Path:
    """Resolve one task-local resource path and reject escape attempts."""

    normalized = str(resource_path or "").strip().replace("\\", "/")
    if not normalized:
        raise ResourceProbeError("resource_path is required")
    candidate = Path(normalized)
    if candidate.is_absolute():
        raise ResourceProbeError(f"resource_path must be relative: {normalized}")
    resolved_root = Path(task_root).resolve()
    resolved_path = (resolved_root / candidate).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ResourceProbeError(
            f"resource_path escapes task root: {normalized}"
        ) from exc
    if not resolved_path.exists():
        raise ResourceProbeError(f"resource_path does not exist: {normalized}")
    if not resolved_path.is_file():
        raise ResourceProbeError(f"resource_path must point to a file: {normalized}")
    return resolved_path


def _apply_field_selector(payload: Any, field: str, *, resource_path: str) -> Any:
    if not field:
        return payload
    current = payload
    for raw_part in str(field).replace("\\", "/").split("/"):
        part = raw_part.strip()
        if not part:
            continue
        if isinstance(current, Mapping):
            if part not in current:
                raise ResourceProbeError(
                    f"field {field!r} not found in {resource_path}"
                )
            current = current[part]
            continue
        raise ResourceProbeError(
            f"field {field!r} cannot be resolved inside {resource_path}"
        )
    return current


def _describe_json_entries(payload: Any, *, resource_path: str) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return [descriptor_for_value(path=resource_path, field="", value=payload)]
    entries: list[dict[str, Any]] = []
    for key, value in payload.items():
        entries.append(descriptor_for_value(path=resource_path, field=str(key), value=value))
    return entries


def _describe_hdf5_entries(handle, *, resource_path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def _visit(name: str, node: Any) -> None:
        if not hasattr(node, "shape") or not hasattr(node, "dtype"):
            return
        entries.append(
            {
                "path": str(resource_path).replace("\\", "/"),
                "field": str(name),
                "observed_dtype": str(node.dtype),
                "observed_shape": [int(dim) for dim in tuple(node.shape)],
            }
        )

    handle.visititems(_visit)
    return entries


def load_resource_payload(
    task_root: Path,
    resource_path: str,
    *,
    field: str = "",
) -> Any:
    """Load one resource payload or field from a supported task-local file."""

    path = resolve_task_local_path(task_root, resource_path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if field:
                if field not in payload.files:
                    raise ResourceProbeError(
                        f"field {field!r} not found in {resource_path}"
                    )
                return np.asarray(payload[field])
            if len(payload.files) == 1:
                return np.asarray(payload[payload.files[0]])
            raise ResourceProbeError(
                f"field is required for multi-field npz resource: {resource_path}"
            )
    if suffix == ".npy":
        if field:
            raise ResourceProbeError(f"field is not supported for npy resource: {resource_path}")
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _apply_field_selector(payload, field, resource_path=resource_path)
    if suffix in _HDF5_SUFFIXES:
        if h5py is None:
            raise ResourceProbeError(
                f"h5py is required to read {resource_path}, but it is not installed"
            )
        with h5py.File(path, "r") as handle:
            if field:
                if field not in handle:
                    raise ResourceProbeError(
                        f"field {field!r} not found in {resource_path}"
                    )
                return np.asarray(handle[field])
            dataset_entries = _describe_hdf5_entries(handle, resource_path=resource_path)
            if len(dataset_entries) == 1:
                return np.asarray(handle[dataset_entries[0]["field"]])
            raise ResourceProbeError(
                f"field is required for multi-dataset hdf5 resource: {resource_path}"
            )
    raise ResourceProbeError(f"unsupported resource type: {resource_path}")


def apply_value_selectors(
    value: Any,
    *,
    index: Any = None,
    scalar_index: Any = None,
    squeeze: bool = False,
) -> Any:
    """Apply contract selectors to one loaded value."""

    resolved = value
    if index is not None:
        if isinstance(index, bool):
            raise ResourceProbeError("index must be an integer, not boolean")
        try:
            resolved = np.asarray(resolved)[int(index)]
        except Exception as exc:
            raise ResourceProbeError(f"index {index!r} could not be applied") from exc
    if scalar_index is not None:
        if isinstance(scalar_index, bool):
            raise ResourceProbeError("scalar_index must be an integer, not boolean")
        try:
            flat = np.asarray(resolved).reshape(-1)
            resolved = flat[int(scalar_index)]
        except Exception as exc:
            raise ResourceProbeError(
                f"scalar_index {scalar_index!r} could not be applied"
            ) from exc
    if bool(squeeze):
        resolved = np.squeeze(resolved)
    return resolved


def load_task_resource_value(
    task_root: Path,
    spec: Mapping[str, Any],
    *,
    output_payload: Any = None,
) -> Any:
    """Resolve one contract metric input spec into a concrete runtime value."""

    if "value" in spec:
        resolved = spec["value"]
    elif output_payload is not None and spec.get("output_field"):
        field_name = str(spec.get("output_field", "") or "").strip()
        if not field_name:
            raise ResourceProbeError("output_field is required when reading from output payload")
        if hasattr(output_payload, "files"):
            if field_name not in output_payload.files:
                raise ResourceProbeError(f"missing output field: {field_name}")
            resolved = np.asarray(output_payload[field_name])
        else:
            try:
                resolved = output_payload[field_name]
            except Exception as exc:
                raise ResourceProbeError(f"missing output field: {field_name}") from exc
    else:
        resource_path = str(
            spec.get("resource_path") or spec.get("reference_resource_path") or ""
        ).strip()
        field_name = str(spec.get("field") or spec.get("reference_field") or "").strip()
        resolved = load_resource_payload(task_root, resource_path, field=field_name)
    return apply_value_selectors(
        resolved,
        index=spec.get("index"),
        scalar_index=spec.get("scalar_index"),
        squeeze=bool(spec.get("squeeze", False)),
    )


def apply_shape_selectors(
    shape: Sequence[int],
    *,
    index: Any = None,
    scalar_index: Any = None,
    squeeze: bool = False,
) -> list[int]:
    """Apply contract selectors to one declared base shape."""

    normalized = normalize_shape_spec(shape)
    if index is not None:
        if not normalized:
            raise ResourceProbeError("index cannot be applied to a scalar shape")
        normalized = normalized[1:]
    if scalar_index is not None:
        normalized = []
    if bool(squeeze):
        normalized = [dim for dim in normalized if dim != 1]
    return normalized


def describe_task_resource(
    task_root: Path,
    resource_path: str,
    *,
    field: str = "",
    index: Any = None,
    scalar_index: Any = None,
    squeeze: bool = False,
) -> dict[str, Any]:
    """Describe one task-local resource or a selected field as JSON-safe metadata."""

    path = resolve_task_local_path(task_root, resource_path)
    suffix = path.suffix.lower()
    normalized_path = str(resource_path).replace("\\", "/")
    if field or index is not None or scalar_index is not None or bool(squeeze):
        value = load_task_resource_value(
            task_root,
            {
                "resource_path": normalized_path,
                "field": field,
                "index": index,
                "scalar_index": scalar_index,
                "squeeze": bool(squeeze),
            },
        )
        return {
            "path": normalized_path,
            "format": suffix.lstrip("."),
            "field": str(field or ""),
            "observed_dtype": dtype_name_for_value(value),
            "observed_shape": shape_list_for_value(value),
            "selectors": {
                "index": index,
                "scalar_index": scalar_index,
                "squeeze": bool(squeeze),
            },
        }
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {
                "path": normalized_path,
                "format": "npz",
                "entries": [
                    descriptor_for_value(path=normalized_path, field=name, value=np.asarray(payload[name]))
                    for name in payload.files
                ],
            }
    if suffix == ".npy":
        value = np.asarray(np.load(path, allow_pickle=False))
        return {
            "path": normalized_path,
            "format": "npy",
            **descriptor_for_value(path=normalized_path, field="", value=value),
        }
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "path": normalized_path,
            "format": "json",
            "entries": _describe_json_entries(payload, resource_path=normalized_path),
        }
    if suffix in _HDF5_SUFFIXES:
        if h5py is None:
            raise ResourceProbeError(
                f"h5py is required to read {resource_path}, but it is not installed"
            )
        with h5py.File(path, "r") as handle:
            return {
                "path": normalized_path,
                "format": suffix.lstrip("."),
                "entries": _describe_hdf5_entries(handle, resource_path=normalized_path),
            }
    raise ResourceProbeError(f"unsupported resource type: {resource_path}")


def _probe_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m myevoskill.resource_probe",
        description="Read-only task-local resource probe for registration-agent shape checks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe_parser = subparsers.add_parser("probe", help="Describe one task-local resource")
    probe_parser.add_argument("task_root")
    probe_parser.add_argument("resource_path")
    probe_parser.add_argument("--field", default="")
    probe_parser.add_argument("--index", type=int, default=None)
    probe_parser.add_argument("--scalar-index", type=int, default=None)
    probe_parser.add_argument("--squeeze", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command != "probe":
        raise ResourceProbeError(f"unsupported command: {args.command}")

    payload = describe_task_resource(
        Path(args.task_root),
        str(args.resource_path),
        field=str(args.field or ""),
        index=args.index,
        scalar_index=args.scalar_index,
        squeeze=bool(args.squeeze),
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for read-only shape probing."""

    try:
        return _probe_cli(argv)
    except ResourceProbeError as exc:
        print(
            json.dumps(
                {"error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI exercised through subprocess usage
    raise SystemExit(main())
