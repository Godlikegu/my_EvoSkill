"""Low-leakage proxy verification helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .models import ProxyFeedback, RunRecord


class ProxyVerifier:
    """Generate low-leakage proxy feedback from run artifacts."""

    def evaluate(
        self,
        run_record: RunRecord,
        proxy_spec: Mapping[str, object] | None = None,
    ) -> ProxyFeedback:
        proxy_spec = dict(proxy_spec or {})
        output_path = Path(
            proxy_spec.get("output_path", run_record.workspace_root / "output" / "reconstruction.npy")
        )
        output_exists = output_path.exists()
        warnings = []
        shape = list(proxy_spec.get("output_shape", []))
        dtype = str(proxy_spec.get("output_dtype", ""))
        has_nan_or_inf = bool(proxy_spec.get("has_nan_or_inf", False))
        runtime_seconds = float(proxy_spec.get("runtime_seconds", run_record.runtime_seconds))
        public_baseline_delta = dict(proxy_spec.get("public_baseline_delta", {}))
        physical_checks = dict(proxy_spec.get("physical_checks", {}))
        required_fields = list(proxy_spec.get("required_fields", []))
        numeric_fields = list(proxy_spec.get("numeric_fields", []))
        same_shape_fields = list(proxy_spec.get("same_shape_fields", []))
        field_specs = [
            dict(item)
            for item in proxy_spec.get("field_specs", []) or []
            if isinstance(item, Mapping)
        ]

        if not output_exists:
            warnings.append("missing output artifact")
        elif dtype == "npz":
            try:
                payload = np.load(output_path, allow_pickle=False)
            except Exception:
                warnings.append("unreadable npz artifact")
            else:
                if field_specs:
                    required_fields = [
                        str(item.get("name", "") or "")
                        for item in field_specs
                        if str(item.get("name", "") or "")
                    ]
                    numeric_fields = list(required_fields)
                missing_fields = [field for field in required_fields if field not in payload.files]
                if missing_fields:
                    warnings.append(
                        "missing required fields: " + ", ".join(sorted(missing_fields))
                    )
                detected_shapes = {}
                for field in numeric_fields:
                    if field not in payload.files:
                        continue
                    value = np.asarray(payload[field])
                    if not np.issubdtype(value.dtype, np.number):
                        warnings.append(f"non-numeric field: {field}")
                        continue
                    if np.any(~np.isfinite(value)):
                        has_nan_or_inf = True
                    if field in same_shape_fields:
                        detected_shapes[field] = tuple(value.shape)
                if field_specs:
                    for item in field_specs:
                        field_name = str(item.get("name", "") or "")
                        if not field_name or field_name not in payload.files:
                            continue
                        value = np.asarray(payload[field_name])
                        expected_shape = list(item.get("shape", []) or [])
                        observed_shape = list(value.shape)
                        if expected_shape and observed_shape != expected_shape:
                            warnings.append(
                                f"invalid shape for field {field_name}: expected {expected_shape}, observed {observed_shape}"
                            )
                        expected_dtype = str(item.get("dtype", "") or "")
                        if expected_dtype and str(value.dtype) != expected_dtype:
                            warnings.append(
                                f"invalid dtype for field {field_name}: expected {expected_dtype}, observed {value.dtype}"
                            )
                if detected_shapes and len(set(detected_shapes.values())) > 1:
                    warnings.append("required fields have inconsistent shapes")
        if has_nan_or_inf:
            warnings.append("output contains NaN or Inf")

        return ProxyFeedback(
            task_id=run_record.task_id,
            output_exists=output_exists,
            output_shape=shape,
            output_dtype=dtype,
            has_nan_or_inf=has_nan_or_inf,
            runtime_seconds=runtime_seconds,
            warnings=warnings,
            public_baseline_delta=public_baseline_delta,
            physical_checks=physical_checks,
        )
