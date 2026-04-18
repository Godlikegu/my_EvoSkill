"""Manifest bootstrap helpers for new scientific tasks."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .models import BootstrapResult
from .task_runtime import DEFAULT_RUNTIME_LAYOUT, DEFAULT_RUNTIME_POLICY

DEFAULT_OUTPUT_PATH = "output/reconstruction.npz"
DEFAULT_JUDGE_ADAPTER_PATH = "evaluation/judge_adapter.py"
DEFAULT_METRICS_CONFIG_PATH = "evaluation/metrics.json"
DEFAULT_REMOVE_PATH_PATTERNS = [
    r"(?i)data/ground_truth\.",
    r"(?i)data/baseline_reference\.",
    r"(?i)data/simu\.",
    r"(?i)evaluation/reference_outputs/",
    r"(?i)evaluation/tests/",
    r"(?i)\bmain\.py\b",
    r"(?i)\bsrc/",
    r"(?i)\bnotebooks/",
    r"(?i)\bplan/",
]
README_PRESERVE_CANDIDATES = (
    "Method Hints",
    "References",
    "Data Description",
    "Problem Description",
    "Background",
)
SCALAR_FIELD_TOKENS = ("temperature", "loss", "error", "score", "runtime")
HIDDEN_DATA_TOKENS = ("ground_truth", "baseline_reference", "sim", "reference", "hidden")


class _PythonSignalVisitor(ast.NodeVisitor):
    """Collect save/export signals from task Python sources."""

    def __init__(self, relative_path: str):
        self.relative_path = relative_path
        self.save_calls: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: D401
        func_name = self._resolve_call_name(node.func)
        path_hint = self._resolve_string_arg(node.args[0]) if node.args else ""
        if func_name.endswith("savez"):
            self.save_calls.append(
                {
                    "kind": "np.savez",
                    "file": self.relative_path,
                    "path_hint": path_hint,
                    "fields": [str(keyword.arg) for keyword in node.keywords if keyword.arg],
                }
            )
        elif func_name.endswith("save"):
            self.save_calls.append(
                {
                    "kind": "np.save",
                    "file": self.relative_path,
                    "path_hint": path_hint,
                    "fields": [],
                }
            )
        elif func_name.endswith("File"):
            self.save_calls.append(
                {
                    "kind": "h5py.File",
                    "file": self.relative_path,
                    "path_hint": path_hint,
                    "fields": [],
                }
            )
        elif func_name == "save_results":
            self.save_calls.append(
                {
                    "kind": "save_results",
                    "file": self.relative_path,
                    "path_hint": path_hint,
                    "fields": [],
                }
            )
        self.generic_visit(node)

    def _resolve_call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._resolve_call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def _resolve_string_arg(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
            return "".join(parts)
        return ""


def bootstrap_task(
    task_root: Path,
    *,
    output_root: Path,
    use_llm: bool = False,
) -> BootstrapResult:
    """Bootstrap a manifest, notes file, and judge stub for one task."""

    return _bootstrap_task(
        task_root=Path(task_root),
        output_root=Path(output_root),
        use_llm=use_llm,
        task_id_override=None,
    )


def load_task_bootstrap_notes(
    task_id: str,
    *,
    output_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Load a previously generated bootstrap notes file."""

    project_root = _resolve_output_root(output_root)
    notes_path = _registry_tasks_root(project_root) / f"{task_id}.notes.json"
    return json.loads(notes_path.read_text(encoding="utf-8"))


def _bootstrap_task(
    *,
    task_root: Path,
    output_root: Path,
    use_llm: bool,
    task_id_override: Optional[str],
) -> BootstrapResult:
    task_root = task_root.resolve()
    output_root = _resolve_output_root(output_root)
    task_id = str(task_id_override or task_root.name).strip()
    registry_root = _registry_tasks_root(output_root)
    manifest_path = registry_root / f"{task_id}.json"
    notes_path = registry_root / f"{task_id}.notes.json"
    suggested_path = registry_root / f"{task_id}.suggested.json"
    judge_stub_path = task_root / DEFAULT_JUDGE_ADAPTER_PATH
    warnings: list[str] = []

    readme_info = _read_readme_info(task_root / "README.md")
    data_summary = _build_data_summary(task_root / "data")
    evaluation_summary = _build_evaluation_summary(task_root / "evaluation")
    output_detection = _build_output_detection(task_root, data_summary, readme_info, task_id)
    public_policy = _build_public_policy(task_root, readme_info, data_summary, evaluation_summary)
    public_eval_spec = _build_public_eval_spec(data_summary, output_detection)
    judge_metric_names = _infer_metric_names(
        metrics_summary=evaluation_summary.get("metrics_summary", {}),
        readme_info=readme_info,
        output_detection=output_detection,
        task_id=task_id,
    )
    family = _infer_family(readme_info)
    source_task_dir = _relative_source_task_dir(task_root, output_root)
    output_fields = list(output_detection["candidate_output_fields"])
    same_shape_fields = _infer_same_shape_fields(output_fields)
    manifest = _build_manifest_template(
        task_id=task_id,
        family=family,
        source_task_dir=source_task_dir,
        public_policy=public_policy,
        output_fields=output_fields,
        same_shape_fields=same_shape_fields,
        public_eval_spec=public_eval_spec,
        judge_metric_names=judge_metric_names,
        metrics_config_exists=bool(evaluation_summary.get("metrics_config_exists", False)),
    )

    existing_manifest_payload: dict[str, Any] = {}
    if manifest_path.exists():
        existing_manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        warnings.append(
            f"manifest already exists at {manifest_path}; keeping existing file unchanged"
        )
    else:
        registry_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    effective_manifest_payload = existing_manifest_payload or manifest

    if judge_stub_path.exists():
        warnings.append(
            f"judge adapter already exists at {judge_stub_path}; keeping existing file unchanged"
        )
    else:
        judge_stub_path.parent.mkdir(parents=True, exist_ok=True)
        judge_stub_path.write_text(
            _render_judge_stub(
                output_fields=output_fields,
                judge_metric_names=judge_metric_names,
            ),
            encoding="utf-8",
        )

    llm_suggestions: dict[str, Any] = {}
    if use_llm:
        llm_suggestions = _build_heuristic_llm_suggestions(
            task_id=task_id,
            public_policy=public_policy,
            output_fields=output_fields,
            same_shape_fields=same_shape_fields,
            public_eval_spec=public_eval_spec,
            judge_metric_names=judge_metric_names,
        )
        suggested_path.write_text(
            json.dumps(llm_suggestions, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        warnings.append(
            "use_llm requested; wrote heuristic suggestion file without modifying the manifest"
        )

    recommended_hidden_paths = _recommended_hidden_paths(task_root, data_summary, evaluation_summary)
    auto_fields, manual_fields = _build_field_classification(
        family=family,
        output_fields=output_fields,
        metrics_config_exists=bool(evaluation_summary.get("metrics_config_exists", False)),
    )
    missing_items = _missing_items_from_manual_fields(
        manual_fields,
        effective_manifest_payload,
    )
    notes_payload = {
        "task_id": task_id,
        "task_root": str(task_root),
        "manifest_path": str(manifest_path),
        "judge_stub_path": str(judge_stub_path),
        "auto_fields": auto_fields,
        "manual_fields": manual_fields,
        "missing_items": missing_items,
        "warnings": sorted(dict.fromkeys(warnings + output_detection["warnings"])),
        "readme_summary": readme_info,
        "data_summary": data_summary,
        "evaluation_summary": evaluation_summary,
        "output_detection": output_detection,
        "recommended_hidden_paths": recommended_hidden_paths,
        "existing_manifest": existing_manifest_payload,
        "llm_suggestions": llm_suggestions,
    }
    registry_root.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(
        json.dumps(notes_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return BootstrapResult(
        task_id=task_id,
        manifest_path=manifest_path,
        judge_stub_path=judge_stub_path,
        notes_path=notes_path,
        missing_items=missing_items,
        warnings=sorted(dict.fromkeys(notes_payload["warnings"])),
    )


def _resolve_output_root(output_root: Optional[Path]) -> Path:
    if output_root is not None:
        return Path(output_root).resolve()
    return Path.cwd().resolve()


def _registry_tasks_root(output_root: Path) -> Path:
    return Path(output_root) / "registry" / "tasks"


def _relative_source_task_dir(task_root: Path, output_root: Path) -> str:
    relative = os.path.relpath(task_root, output_root)
    return Path(relative).as_posix()


def _read_readme_info(readme_path: Path) -> dict[str, Any]:
    if not readme_path.exists():
        return {
            "exists": False,
            "title": "",
            "headings": [],
            "domain": "",
            "output_excerpt": "",
        }
    text = readme_path.read_text(encoding="utf-8")
    headings = [
        match.group("heading").strip()
        for match in re.finditer(r"^\s*#+\s*(?P<heading>.+?)\s*$", text, re.MULTILINE)
    ]
    title = headings[0] if headings else ""
    domain_match = re.search(r"Domain:\s*([^|\n]+)", text, flags=re.IGNORECASE)
    output_excerpt = ""
    output_match = re.search(
        r"^\s*\*\*?Output\*?\*?\s*:\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if output_match:
        output_excerpt = output_match.group(1).strip()
    return {
        "exists": True,
        "title": title,
        "headings": headings,
        "domain": domain_match.group(1).strip() if domain_match else "",
        "output_excerpt": output_excerpt,
    }


def _build_data_summary(data_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": data_dir.exists(),
        "files": [],
        "raw_data": {"exists": False, "arrays": {}},
        "ground_truth": {"exists": False, "arrays": {}},
        "meta_data": {"exists": False, "keys": []},
    }
    if not data_dir.exists():
        return summary
    summary["files"] = [
        path.relative_to(data_dir).as_posix()
        for path in sorted(item for item in data_dir.rglob("*") if item.is_file())
    ]
    raw_data_path = data_dir / "raw_data.npz"
    ground_truth_path = data_dir / "ground_truth.npz"
    meta_data_path = data_dir / "meta_data.json"
    summary["raw_data"] = {
        "exists": raw_data_path.exists(),
        "arrays": _inspect_npz_file(raw_data_path) if raw_data_path.exists() else {},
    }
    summary["ground_truth"] = {
        "exists": ground_truth_path.exists(),
        "arrays": _inspect_npz_file(ground_truth_path) if ground_truth_path.exists() else {},
    }
    summary["meta_data"] = {
        "exists": meta_data_path.exists(),
        "keys": _inspect_json_keys(meta_data_path) if meta_data_path.exists() else [],
    }
    return summary


def _build_evaluation_summary(evaluation_dir: Path) -> dict[str, Any]:
    metrics_path = evaluation_dir / "metrics.json"
    reference_dir = evaluation_dir / "reference_outputs"
    metrics_summary: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            metrics_summary = {"parse_error": str(exc)}
    return {
        "exists": evaluation_dir.exists(),
        "metrics_config_exists": metrics_path.exists(),
        "metrics_config_path": DEFAULT_METRICS_CONFIG_PATH if metrics_path.exists() else "",
        "metrics_summary": metrics_summary,
        "metric_threshold_keys": sorted(
            key for key in metrics_summary.keys() if key.endswith("_boundary")
        ),
        "reference_outputs_exists": reference_dir.exists(),
        "reference_outputs_files": [
            path.relative_to(evaluation_dir).as_posix()
            for path in sorted(item for item in reference_dir.rglob("*") if item.is_file())
        ]
        if reference_dir.exists()
        else [],
    }


def _inspect_npz_file(path: Path) -> dict[str, dict[str, Any]]:
    arrays: dict[str, dict[str, Any]] = {}
    try:
        with np.load(path, allow_pickle=False) as payload:
            for name in payload.files:
                value = np.asarray(payload[name])
                arrays[name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
    except Exception as exc:  # pragma: no cover - defensive fallback
        arrays["__error__"] = {"message": str(exc)}
    return arrays


def _inspect_json_keys(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        return sorted(str(key) for key in payload.keys())
    return []


def _build_output_detection(
    task_root: Path,
    data_summary: Mapping[str, Any],
    readme_info: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    save_calls = _scan_python_sources(task_root)
    warnings: list[str] = []
    candidate_output_fields: list[str] = []
    candidate_sources: list[str] = []

    raw_arrays = dict((data_summary.get("raw_data") or {}).get("arrays") or {})
    ground_truth_arrays = dict((data_summary.get("ground_truth") or {}).get("arrays") or {})
    domain_text = " ".join(
        str(item or "")
        for item in (
            readme_info.get("title", ""),
            readme_info.get("domain", ""),
            readme_info.get("output_excerpt", ""),
            task_id,
        )
    ).lower()
    recommended_fields, recommended_sources = _recommend_output_fields(
        raw_arrays=raw_arrays,
        ground_truth_arrays=ground_truth_arrays,
        domain_text=domain_text,
    )
    if recommended_fields:
        candidate_output_fields = recommended_fields
        candidate_sources.extend(recommended_sources)
    else:
        for call in save_calls:
            if call["kind"] == "np.savez" and call["fields"]:
                candidate_output_fields = [str(field) for field in call["fields"]]
                candidate_sources.append(f"{call['file']}:{call['kind']}")
                break
    has_hdf5_signal = any(
        call["kind"] in {"h5py.File", "save_results"}
        or str(call.get("path_hint", "")).lower().endswith((".h5", ".hdf5"))
        for call in save_calls
    )
    if not candidate_output_fields and "object" in ground_truth_arrays and len(ground_truth_arrays) == 1:
        candidate_output_fields = ["object"]
        candidate_sources.append("ground_truth.npz:single-key")
    if not candidate_output_fields:
        generic_fields: list[str] = []
        generic_sources: list[str] = []
        if "temperature" in ground_truth_arrays:
            generic_fields.append("estimated_temperature_K")
            generic_sources.append("ground_truth.npz:temperature")
        if "measurements" in raw_arrays:
            generic_fields.append("reconstructed_spectrum")
            generic_sources.append("raw_data.npz:measurements")
        if "nu_axis" in raw_arrays:
            generic_fields.append("nu_axis")
            generic_sources.append("raw_data.npz:nu_axis")
        if generic_fields:
            candidate_output_fields = generic_fields
            candidate_sources.extend(generic_sources)
    if not candidate_output_fields:
        candidate_output_fields = ["TODO_field_name"]
        candidate_sources.append("template:TODO_field_name")

    if has_hdf5_signal:
        warnings.append(
            "task appears to export HDF5 or custom binary outputs; add a public npz export at output/reconstruction.npz"
        )

    return {
        "output_path": DEFAULT_OUTPUT_PATH,
        "output_format": "npz",
        "candidate_output_fields": candidate_output_fields,
        "candidate_sources": candidate_sources,
        "save_calls": save_calls,
        "needs_public_npz_export": has_hdf5_signal,
        "warnings": warnings,
    }


def _recommend_output_fields(
    *,
    raw_arrays: Mapping[str, Any],
    ground_truth_arrays: Mapping[str, Any],
    domain_text: str,
) -> tuple[list[str], list[str]]:
    if (
        "ptychograph" in domain_text
        and "object" in ground_truth_arrays
    ):
        return ["object"], ["ground_truth.npz:object"]

    if (
        {"measurements", "nu_axis"}.issubset(raw_arrays.keys())
        and {"temperature", "spectrum"}.issubset(ground_truth_arrays.keys())
    ):
        return [
            "estimated_temperature_K",
            "reconstructed_spectrum",
            "nu_axis",
        ], [
            "ground_truth.npz:temperature",
            "ground_truth.npz:spectrum",
            "raw_data.npz:nu_axis",
        ]

    return [], []


def _scan_python_sources(task_root: Path) -> list[dict[str, Any]]:
    python_files: list[Path] = []
    main_path = task_root / "main.py"
    if main_path.exists():
        python_files.append(main_path)
    src_dir = task_root / "src"
    if src_dir.exists():
        python_files.extend(sorted(item for item in src_dir.rglob("*.py") if item.is_file()))
    collected: list[dict[str, Any]] = []
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        visitor = _PythonSignalVisitor(path.relative_to(task_root).as_posix())
        visitor.visit(tree)
        collected.extend(visitor.save_calls)
    return collected


def _build_public_policy(
    task_root: Path,
    readme_info: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
) -> dict[str, Any]:
    headings = [str(item) for item in readme_info.get("headings", []) or []]
    preserve_sections = [
        section for section in README_PRESERVE_CANDIDATES if section in headings
    ]
    data_files = [str(item) for item in data_summary.get("files", []) or []]
    allowlist = [
        path
        for path in ("data/raw_data.npz", "data/meta_data.json")
        if Path(task_root / path).exists()
    ]
    denylist = sorted(
        {
            Path(path).name
            for path in data_files
            if path not in allowlist
            and any(token in Path(path).name.lower() for token in HIDDEN_DATA_TOKENS)
        }
    )
    if evaluation_summary.get("reference_outputs_exists", False):
        denylist.append("reference_outputs")
    return {
        "readme_policy": {
            "preserve_sections": preserve_sections,
            "remove_sections": [],
            "remove_path_patterns": list(DEFAULT_REMOVE_PATH_PATTERNS),
            "preserve_user_eval_notes": True,
        },
        "public_data_allowlist": allowlist,
        "public_data_denylist": sorted(set(denylist)),
    }


def _build_public_eval_spec(
    data_summary: Mapping[str, Any],
    output_detection: Mapping[str, Any],
) -> dict[str, Any]:
    alignments: list[dict[str, Any]] = []
    output_fields = [str(item) for item in output_detection.get("candidate_output_fields", [])]
    raw_arrays = dict((data_summary.get("raw_data") or {}).get("arrays") or {})
    if "nu_axis" in output_fields and "nu_axis" in raw_arrays:
        alignments.append(
            {
                "output_path": DEFAULT_OUTPUT_PATH,
                "field": "nu_axis",
                "source_path": "data/raw_data.npz",
                "source_field": "nu_axis",
                "mode": "allclose",
                "rtol": 1e-6,
                "atol": 1e-6,
            }
        )
    if "reconstructed_spectrum" in output_fields and "measurements" in raw_arrays:
        alignments.append(
            {
                "output_path": DEFAULT_OUTPUT_PATH,
                "field": "reconstructed_spectrum",
                "source_path": "data/raw_data.npz",
                "source_field": "measurements",
                "mode": "shape",
            }
        )
    return {"alignments": alignments}


def _infer_same_shape_fields(output_fields: Sequence[str]) -> list[str]:
    same_shape_fields: list[str] = []
    for field in output_fields:
        normalized = field.lower()
        if any(token in normalized for token in SCALAR_FIELD_TOKENS) or normalized.endswith("_k"):
            continue
        if field.startswith("TODO_"):
            continue
        same_shape_fields.append(field)
    return same_shape_fields


def _infer_metric_names(
    *,
    metrics_summary: Mapping[str, Any],
    readme_info: Mapping[str, Any],
    output_detection: Mapping[str, Any],
    task_id: str,
) -> list[str]:
    boundary_keys = sorted(
        key for key in metrics_summary.keys() if str(key).endswith("_boundary")
    )
    metrics_text = json.dumps(metrics_summary, sort_keys=True).lower()
    domain_text = " ".join(
        str(item or "")
        for item in (
            task_id,
            readme_info.get("title", ""),
            readme_info.get("domain", ""),
        )
    ).lower()
    output_fields = [str(item) for item in output_detection.get("candidate_output_fields", [])]
    if (
        "ptychograph" in domain_text
        and "object" in output_fields
        and "phase" in metrics_text
    ):
        phase_metrics: list[str] = []
        if "ncc_boundary" in boundary_keys:
            phase_metrics.append("phase_ncc")
        if "nrmse_boundary" in boundary_keys:
            phase_metrics.append("phase_nrmse")
        if phase_metrics:
            return phase_metrics
    metric_names: list[str] = []
    for key in boundary_keys:
        metric_names.append(str(key)[: -len("_boundary")])
    return metric_names


def _infer_family(readme_info: Mapping[str, Any]) -> str:
    domain = str(readme_info.get("domain", "") or "").strip()
    if not domain:
        return "TODO_family"
    normalized = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
    return normalized or "TODO_family"


def _build_manifest_template(
    *,
    task_id: str,
    family: str,
    source_task_dir: str,
    public_policy: Mapping[str, Any],
    output_fields: Sequence[str],
    same_shape_fields: Sequence[str],
    public_eval_spec: Mapping[str, Any],
    judge_metric_names: Sequence[str],
    metrics_config_exists: bool,
) -> dict[str, Any]:
    normalized_output_fields = list(output_fields) if output_fields else ["TODO_field_name"]
    return {
        "task_id": task_id,
        "family": family,
        "source_task_dir": source_task_dir,
        "public_policy": dict(public_policy),
        "runtime_layout": dict(DEFAULT_RUNTIME_LAYOUT),
        "runtime_policy": dict(DEFAULT_RUNTIME_POLICY),
        "output_contract": {
            "required_outputs": [
                {
                    "path": DEFAULT_OUTPUT_PATH,
                    "format": "npz",
                    "required_fields": normalized_output_fields,
                    "numeric_fields": normalized_output_fields,
                    "same_shape_fields": list(same_shape_fields),
                }
            ]
        },
        "public_eval_spec": dict(public_eval_spec),
        "proxy_spec": {
            "primary_output": DEFAULT_OUTPUT_PATH,
            "output_dtype": "npz",
            "required_fields": normalized_output_fields,
            "numeric_fields": normalized_output_fields,
            "same_shape_fields": list(same_shape_fields),
        },
        "judge_spec": {
            "adapter_path": DEFAULT_JUDGE_ADAPTER_PATH,
            "callable": "evaluate_run",
            "metrics_config_path": DEFAULT_METRICS_CONFIG_PATH if metrics_config_exists else "",
            "metrics": list(judge_metric_names),
            "required_fields": normalized_output_fields,
        },
    }


def _build_field_classification(
    *,
    family: str,
    output_fields: Sequence[str],
    metrics_config_exists: bool,
) -> tuple[dict[str, Any], list[str]]:
    auto_fields = {
        "task_id": True,
        "source_task_dir": True,
        "runtime_layout": True,
        "evaluation.metrics_config_path": metrics_config_exists,
        "output_contract.required_outputs[0].path": True,
        "output_contract.required_outputs[0].format": True,
    }
    manual_fields = [
        "public_policy",
        "public_eval_spec",
        "judge_spec",
        "runtime_policy",
    ]
    if family == "TODO_family":
        manual_fields.insert(0, "family")
    if not output_fields or any(field.startswith("TODO_") for field in output_fields):
        manual_fields.extend(
            [
                "output_contract.required_outputs[0].required_fields",
                "proxy_spec.required_fields",
            ]
        )
    return auto_fields, manual_fields


def _missing_items_from_manual_fields(
    manual_fields: Iterable[str],
    manifest_payload: Mapping[str, Any],
) -> list[str]:
    missing_items: list[str] = []
    for item in manual_fields:
        field_path = str(item)
        exists, value = _resolve_field_path(manifest_payload, field_path)
        if not exists or _value_has_unresolved_placeholder(value):
            missing_items.append(field_path)
    return sorted(dict.fromkeys(missing_items))


def _resolve_field_path(
    payload: Mapping[str, Any],
    field_path: str,
) -> tuple[bool, Any]:
    current: Any = payload
    for raw_part in str(field_path).split("."):
        part = str(raw_part)
        index: Optional[int] = None
        match = re.match(r"^(?P<name>[^\[]+)\[(?P<index>\d+)\]$", part)
        if match:
            part = str(match.group("name"))
            index = int(match.group("index"))
        if part:
            if not isinstance(current, Mapping) or part not in current:
                return False, None
            current = current[part]
        if index is not None:
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)):
                return False, None
            if index >= len(current):
                return False, None
            current = current[index]
    return True, current


def _value_has_unresolved_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return not stripped or "TODO" in stripped
    if isinstance(value, Mapping):
        if not value:
            return True
        return any(_contains_placeholder_token(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return True
        return any(_contains_placeholder_token(item) for item in value)
    return False


def _contains_placeholder_token(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return not stripped or "TODO" in stripped
    if isinstance(value, Mapping):
        return any(_contains_placeholder_token(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_placeholder_token(item) for item in value)
    return False


def _recommended_hidden_paths(
    task_root: Path,
    data_summary: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
) -> list[str]:
    hidden_paths = {
        "main.py",
        "src/",
        "notebooks/",
        "plan/",
    }
    for path in data_summary.get("files", []) or []:
        normalized = str(path)
        if any(token in Path(normalized).name.lower() for token in HIDDEN_DATA_TOKENS):
            hidden_paths.add(f"data/{Path(normalized).name}")
    if evaluation_summary.get("reference_outputs_exists", False):
        hidden_paths.add("evaluation/reference_outputs/")
        for item in evaluation_summary.get("reference_outputs_files", []) or []:
            hidden_paths.add(f"evaluation/{item}")
    return sorted(hidden_paths)


def _build_heuristic_llm_suggestions(
    *,
    task_id: str,
    public_policy: Mapping[str, Any],
    output_fields: Sequence[str],
    same_shape_fields: Sequence[str],
    public_eval_spec: Mapping[str, Any],
    judge_metric_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "mode": "heuristic_fallback",
        "public_policy": dict(public_policy),
        "output_contract": {
            "required_outputs": [
                {
                    "path": DEFAULT_OUTPUT_PATH,
                    "format": "npz",
                    "required_fields": list(output_fields),
                    "numeric_fields": list(output_fields),
                    "same_shape_fields": list(same_shape_fields),
                }
            ]
        },
        "public_eval_spec": dict(public_eval_spec),
        "judge_stub_comments": [
            "TODO: compute task-specific hidden metrics from output/reconstruction.npz and private references.",
            (
                "Suggested metrics: " + ", ".join(judge_metric_names)
                if judge_metric_names
                else "Suggested metrics: confirm metric names from evaluation/metrics.json."
            ),
        ],
    }


def _render_judge_stub(
    *,
    output_fields: Sequence[str],
    judge_metric_names: Sequence[str],
) -> str:
    required_fields = list(output_fields) if output_fields else ["TODO_field_name"]
    metrics_list = list(judge_metric_names)
    return (
        '"""Task-local hidden judge adapter stub generated by manifest_bootstrap."""\n\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "from pathlib import Path\n"
        "from typing import Any, Mapping\n\n"
        "import numpy as np\n\n"
        "from myevoskill.judging import HiddenJudge, MetricRequirement\n"
        "from myevoskill.models import JudgeResult, RunRecord\n"
        "from myevoskill.task_runtime import resolve_primary_output_path\n\n\n"
        "def _default_requirements(\n"
        "    metrics_cfg: Mapping[str, Any],\n"
        "    explicit_metric_names: Sequence[str],\n"
        ") -> list[MetricRequirement]:\n"
        "    requirements: list[MetricRequirement] = []\n"
        "    boundary_items = [\n"
        "        (str(key), value)\n"
        "        for key, value in sorted(metrics_cfg.items())\n"
        "        if str(key).endswith('_boundary')\n"
        "    ]\n"
        "    explicit_names = list(explicit_metric_names)\n"
        "    use_explicit_names = bool(explicit_names) and len(explicit_names) == len(boundary_items)\n"
        "    for index, (key, value) in enumerate(boundary_items):\n"
        "        metric_name = explicit_names[index] if use_explicit_names else str(key)[:-len('_boundary')]\n"
        "        operator = '<=' if any(token in metric_name for token in ('error', 'rmse', 'loss')) else '>='\n"
        "        try:\n"
        "            threshold = float(value)\n"
        "        except (TypeError, ValueError):\n"
        "            continue\n"
        "        requirements.append(MetricRequirement(metric_name, threshold, operator))\n"
        "    return requirements\n\n\n"
        "def evaluate_run(\n"
        "    task_root: Path,\n"
        "    run_record: RunRecord,\n"
        "    manifest: Mapping[str, Any],\n"
        ") -> JudgeResult:\n"
        "    \"\"\"Evaluate one run using task-private references.\n\n"
        "    TODO:\n"
        "    - load output/reconstruction.npz from the run workspace\n"
        "    - read evaluation/metrics.json thresholds\n"
        "    - compute task-specific hidden metrics\n"
        "    - populate metrics_actual below before returning HiddenJudge.evaluate(...)\n"
        "    \"\"\"\n"
        "    task_root = Path(task_root)\n"
        "    judge_spec = dict(manifest.get('judge_spec') or {})\n"
        "    required_fields = list(judge_spec.get('required_fields') or " + repr(required_fields) + ")\n"
        "    metric_names = list(judge_spec.get('metrics') or " + repr(metrics_list) + ")\n"
        "    metrics_config_path = task_root / str(judge_spec.get('metrics_config_path') or 'evaluation/metrics.json')\n"
        "    metrics_cfg = {}\n"
        "    if metrics_config_path.exists():\n"
        "        try:\n"
        "            metrics_cfg = json.loads(metrics_config_path.read_text(encoding='utf-8'))\n"
        "        except json.JSONDecodeError:\n"
        "            metrics_cfg = {}\n"
        "    requirements = _default_requirements(metrics_cfg, metric_names)\n"
        "    if not requirements:\n"
        "        fallback_name = metric_names[0] if metric_names else 'todo_hidden_metric'\n"
        "        operator = '<=' if any(token in fallback_name for token in ('error', 'rmse', 'loss')) else '>='\n"
        "        requirements = [MetricRequirement(fallback_name, 0.0, operator)]\n"
        "    output_path = resolve_primary_output_path(run_record.workspace_root, manifest)\n"
        "    if not output_path.exists():\n"
        "        return HiddenJudge().evaluate(\n"
        "            task_id=run_record.task_id,\n"
        "            metrics_actual={},\n"
        "            requirements=requirements,\n"
        "            failure_tags=['missing_output'],\n"
        "        )\n"
        "    try:\n"
        "        with np.load(output_path, allow_pickle=False) as payload:\n"
        "            missing_fields = [field for field in required_fields if field not in payload.files]\n"
        "            if missing_fields:\n"
        "                return HiddenJudge().evaluate(\n"
        "                    task_id=run_record.task_id,\n"
        "                    metrics_actual={},\n"
        "                    requirements=requirements,\n"
        "                    failure_tags=['missing_required_field'],\n"
        "                )\n"
        "            # TODO: compute task-specific metrics here.\n"
        "            metrics_actual: dict[str, float] = {}\n"
        "    except Exception:\n"
        "        return HiddenJudge().evaluate(\n"
        "            task_id=run_record.task_id,\n"
        "            metrics_actual={},\n"
        "            requirements=requirements,\n"
        "            failure_tags=['invalid_output_file'],\n"
        "        )\n"
        "    return HiddenJudge().evaluate(\n"
        "        task_id=run_record.task_id,\n"
        "        metrics_actual=metrics_actual,\n"
        "        requirements=requirements,\n"
        "        failure_tags=['judge_not_implemented'" + (
            ", 'suggested_metrics:" + ",".join(metrics_list) + "'" if metrics_list else ""
        ) + "],\n"
        "    )\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for task registration."""

    parser = argparse.ArgumentParser(
        description=(
            "Register a raw scientific task into registry/tasks and generate the "
            "initial judge adapter stub."
        )
    )
    parser.add_argument("--task-root", required=True, help="Path to the raw task directory")
    parser.add_argument("--task-id", default="", help="Optional task id override")
    parser.add_argument(
        "--output-root",
        default="",
        help="Project root where registry/tasks should be written (defaults to cwd)",
    )
    parser.add_argument(
        "--use-claude",
        action="store_true",
        help="Write a separate heuristic suggestion file without overwriting the manifest",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = _bootstrap_task(
        task_root=Path(args.task_root),
        output_root=_resolve_output_root(Path(args.output_root) if args.output_root else None),
        use_llm=bool(args.use_claude),
        task_id_override=str(args.task_id or "").strip() or None,
    )
    print(
        json.dumps(
            {
                "task_id": result.task_id,
                "manifest_path": str(result.manifest_path),
                "judge_stub_path": str(result.judge_stub_path),
                "notes_path": str(result.notes_path),
                "missing_items": list(result.missing_items),
                "warnings": list(result.warnings),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
