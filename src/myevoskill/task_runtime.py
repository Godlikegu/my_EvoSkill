"""Manifest-driven runtime layout, path, and timeout helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

from .models import (
    EffectiveRuntimePolicy,
    ExecutorSessionConfig,
    ModelConfig,
    RunPaths,
    TaskBundle,
)
from .task_contract import task_contract_primary_output_path


DEFAULT_RUNTIME_LAYOUT: Dict[str, str] = {
    "data_dir": "data",
    "work_dir": "work",
    "output_dir": "output",
    "checkpoints_dir": "checkpoints",
    "public_bundle_dir": "public_bundle",
}

DEFAULT_RUNTIME_POLICY: Dict[str, int] = {
    "model_timeout_seconds": 240,
    "execution_budget_seconds": 900,
}


def coerce_runtime_layout(runtime_layout: Mapping[str, Any] | None = None) -> Dict[str, str]:
    """Merge a manifest runtime layout with stable defaults."""

    layout = dict(DEFAULT_RUNTIME_LAYOUT)
    for key, value in dict(runtime_layout or {}).items():
        if value:
            layout[str(key)] = str(value)
    return layout


def coerce_runtime_policy(runtime_policy: Mapping[str, Any] | None = None) -> Dict[str, int]:
    """Merge a manifest runtime policy with safe defaults."""

    policy = dict(DEFAULT_RUNTIME_POLICY)
    incoming = dict(runtime_policy or {})
    for key in ("model_timeout_seconds", "execution_budget_seconds"):
        value = incoming.get(key)
        if value is None:
            continue
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            policy[key] = numeric
    return policy


def load_task_spec(task_bundle: TaskBundle) -> Dict[str, Any]:
    """Load the compiled task spec if it exists."""

    path = Path(task_bundle.task_spec_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_runtime_paths(
    workspace_root: Path,
    runtime_layout: Mapping[str, Any] | None = None,
) -> Dict[str, Path]:
    """Resolve standard runtime directories under one workspace root."""

    layout = coerce_runtime_layout(runtime_layout)
    root = Path(workspace_root)
    return {
        "runtime_root": root,
        "data_dir": root / layout["data_dir"],
        "work_dir": root / layout["work_dir"],
        "output_dir": root / layout["output_dir"],
        "checkpoints_dir": root / layout["checkpoints_dir"],
        "public_bundle_dir": root / layout["public_bundle_dir"],
    }


def resolve_run_paths(repo_root: Path, task_id: str, run_id: str) -> RunPaths:
    """Resolve persistent workspace and log roots inside the project."""

    base = Path(repo_root)
    workspace_root = base / "artifacts" / "workspaces" / task_id / run_id
    log_root = base / "artifacts" / "logs" / task_id / run_id
    return RunPaths(
        repo_root=base,
        task_id=task_id,
        run_id=run_id,
        workspace_root=workspace_root,
        log_root=log_root,
    )


def ensure_clean_run_directory(run_root: Path) -> Path:
    """Reset one run directory without touching sibling runs."""

    def on_rm_error(function, target, exc_info):
        try:
            target_path = Path(target)
            if target_path.exists():
                os.chmod(target_path, 0o755)
            parent = target_path.parent
            while parent.exists() and parent != path.parent:
                os.chmod(parent, 0o755)
                if parent == path:
                    break
                parent = parent.parent
        except OSError:
            pass
        function(target)

    path = Path(run_root)
    if path.exists():
        shutil.rmtree(path, onerror=on_rm_error)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_runtime_policy(
    task_spec: Mapping[str, Any] | None,
    session_config: ExecutorSessionConfig,
    model_config: ModelConfig | None,
) -> EffectiveRuntimePolicy:
    """Resolve effective timeouts using session overrides first, then manifest."""

    task_spec = dict(task_spec or {})
    manifest_policy = coerce_runtime_policy(task_spec.get("runtime_policy"))

    execution_budget = (
        int(session_config.budget_seconds)
        if int(session_config.budget_seconds or 0) > 0
        else int(manifest_policy["execution_budget_seconds"])
    )

    requested_model_timeout = 0
    if model_config is not None:
        requested_model_timeout = int(model_config.timeout or 0)
    model_timeout = (
        requested_model_timeout
        if requested_model_timeout > 0
        else int(manifest_policy["model_timeout_seconds"])
    )

    return EffectiveRuntimePolicy(
        model_timeout_seconds=model_timeout,
        execution_budget_seconds=execution_budget,
    )


def primary_output_relative_path(task_spec: Mapping[str, Any] | None = None) -> str:
    """Resolve the primary output path from a manifest-derived task spec."""

    spec = dict(task_spec or {})
    primary_output = str(spec.get("primary_output_path", "") or "").strip()
    if primary_output:
        return primary_output

    task_contract = dict(spec.get("task_contract") or {})
    if task_contract:
        return task_contract_primary_output_path(task_contract)

    proxy_spec = dict(spec.get("proxy_spec") or {})
    proxy_primary_output = proxy_spec.get("primary_output")
    if proxy_primary_output:
        return str(proxy_primary_output)

    output_contract = dict(spec.get("output_contract") or {})
    required_outputs = list(output_contract.get("required_outputs") or [])
    if required_outputs:
        first_output = required_outputs[0]
        if isinstance(first_output, Mapping) and first_output.get("path"):
            return str(first_output["path"])
        if first_output:
            return str(first_output)

    layout = coerce_runtime_layout(spec.get("runtime_layout"))
    output_name = spec.get("proxy_output_name") or "reconstruction.npz"
    output_name = str(output_name)
    if "/" in output_name:
        return output_name
    return f"{layout['output_dir']}/{output_name}"


def resolve_primary_output_path(
    workspace_root: Path,
    task_spec: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve the concrete primary output path for one run workspace."""

    return Path(workspace_root) / primary_output_relative_path(task_spec)
