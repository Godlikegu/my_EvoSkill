"""Helpers for model-aware artifact paths."""

from __future__ import annotations

import re
from pathlib import Path


ARTIFACT_LAYOUT_VERSION = "model_task_run_v1"
DEFAULT_MODEL_SLUG = "default_model"
_MAX_SLUG_LENGTH = 120
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_UNDERSCORES = re.compile(r"_+")


def model_slug(value: str | None) -> str:
    """Return a filesystem-safe slug for a model identifier."""

    raw = str(value or "").strip()
    if not raw:
        return DEFAULT_MODEL_SLUG
    slug = _UNSAFE_CHARS.sub("_", raw)
    slug = _UNDERSCORES.sub("_", slug).strip("._-")
    if not slug:
        return DEFAULT_MODEL_SLUG
    return slug[:_MAX_SLUG_LENGTH]


def model_artifact_root(
    repo_root: Path,
    kind: str,
    model_slug_value: str | None,
    task_id: str,
    run_id: str,
) -> Path:
    """Return ``artifacts/<kind>/<model_slug>/<task_id>/<run_id>``."""

    return (
        Path(repo_root)
        / "artifacts"
        / kind
        / model_slug(model_slug_value)
        / task_id
        / run_id
    )


def resolve_workspace_output_path(
    *,
    repo_root: Path,
    task_id: str,
    run_id: str,
    filename: str | None = None,
    model_slug_value: str | None = None,
) -> Path:
    """Find a workspace output path using new paths first, then legacy paths."""

    root = Path(repo_root)
    suffix = Path("output") / filename if filename else Path("output")
    candidates: list[Path] = []
    if model_slug_value:
        candidates.append(
            root
            / "artifacts"
            / "workspaces"
            / model_slug(model_slug_value)
            / task_id
            / run_id
            / suffix
        )
    else:
        model_root = root / "artifacts" / "workspaces"
        if model_root.exists():
            candidates.extend(
                sorted(model_root.glob(f"*/{task_id}/{run_id}/{suffix.as_posix()}"))
            )
    candidates.append(root / "artifacts" / "workspaces" / task_id / run_id / suffix)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
