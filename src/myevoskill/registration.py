"""Deterministic task registration.

This module turns a single task directory under ``tasks/<task_id>/`` into a
manifest at ``registry/tasks/<task_id>.json`` consumed by the harness. There
are no LLM calls and no agent involvement: the task author is expected to
publish a v2 ``evaluation/task_contract.json`` (per-task contract) which
*declares* everything needed for registration and judging:

* ``files[]``       - file ids, paths, ``visibility`` (public/private),
                      ``role`` (input_data / metadata / metric_helper / ...)
* ``execution``     - ``readable_files`` ids, ``writable_paths``,
                      ``entrypoint``
* ``output.path``   - the canonical primary output path
* ``metrics[]``     - the judge will read these directly via the existing
                      ``judging`` / ``task_contract`` modules

The manifest produced here is small and stable; it is the *only* contract
between :mod:`myevoskill.cli` and :mod:`myevoskill.workspace` /
:mod:`myevoskill.judge`. Older v1 contracts (``registration_contract.json``)
are not supported by this module - the user is asked to migrate the task
to a v2 ``task_contract.json`` first. (All 57 tasks shipped in this repo
are already v2.)

The module is intentionally pure / side-effect-free except for the final
``write_manifest`` step, so it is straightforward to unit-test.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)


CONTRACT_REL = "evaluation/task_contract.json"
CONTRACT_PUBLIC_REL = "evaluation/task_contract.public.json"
JUDGE_ADAPTER_REL = "evaluation/judge_adapter.py"


# Files matching any of these are *never* shipped to the agent, regardless of
# what the task contract says.  They correspond to the reference solution and
# author plans which would let the agent trivially "hack" the task.
_ALWAYS_HIDDEN_PREFIXES: tuple[str, ...] = (
    "src/",
    "notebooks/",
    "plan/",
)
_ALWAYS_HIDDEN_SUFFIXES: tuple[str, ...] = (
    "/main.py",
    ".ipynb",
)


def _is_always_hidden(path: str) -> bool:
    """Return True iff *path* must never be exposed to the agent."""

    if not path:
        return False
    norm = path.strip().lstrip("./").replace("\\", "/").lower()
    if norm == "main.py":
        return True
    for prefix in _ALWAYS_HIDDEN_PREFIXES:
        if norm.startswith(prefix):
            return True
    for suffix in _ALWAYS_HIDDEN_SUFFIXES:
        if norm.endswith(suffix):
            return True
    return False



# --------------------------------------------------------------------- errors


class RegistrationError(Exception):
    """Raised when a task cannot be registered deterministically."""


# --------------------------------------------------------------------- result


@dataclass(frozen=True)
class RegistrationResult:
    """Outcome of a successful registration."""

    task_id: str
    manifest_path: Path
    manifest: Mapping[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------- helpers


def _load_contract(task_root: Path) -> dict[str, Any]:
    contract_path = task_root / CONTRACT_REL
    if not contract_path.exists():
        raise RegistrationError(
            f"task contract not found: {contract_path}\n"
            f"Each task must publish a v2 contract at {CONTRACT_REL}. "
            f"See doc/harness_design.md for the schema."
        )
    try:
        return json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistrationError(
            f"invalid JSON in {contract_path}: {exc}"
        ) from exc


def _require(value: Any, name: str, ctx: str) -> Any:
    if value in (None, ""):
        raise RegistrationError(f"contract field {name!r} is required ({ctx})")
    return value


def _build_public_policy(
    contract: Mapping[str, Any],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    """Derive the public_policy block from the v2 contract.

    * ``public_data_allowlist`` = paths of every public file *except* the
      README (which the workspace builder handles separately).
    * ``public_data_denylist``  = paths of every private file, plus the
      hard-coded set of judge-only assets the harness denylists by default.
    * ``readme_policy``         = a sane default that strips references to
      hidden assets but preserves the structured task-description sections.
    """

    files = contract.get("files") or []
    if not isinstance(files, list):
        raise RegistrationError("contract.files must be a list")

    allowlist: list[str] = []
    denylist: list[str] = ["ground_truth.npz", "reference_outputs"]

    for entry in files:
        if not isinstance(entry, Mapping):
            warnings.append(f"skipping non-mapping file entry: {entry!r}")
            continue
        path = str(entry.get("path") or "").strip()
        if not path:
            continue
        visibility = str(entry.get("visibility") or "public").strip().lower()
        role = str(entry.get("role") or "").strip().lower()
        if visibility == "public":
            # Hard rule first: reference solution / notebooks / author plan
            # are NEVER allowed to be public, even if the contract says so
            # (and even if the author miscategorised them as
            # ``task_description``). Downgrade them to the denylist and emit
            # a loud warning so the task author can fix the contract.
            if _is_always_hidden(path):
                warnings.append(
                    f"file {path!r} declared public but matches always-hidden "
                    f"prefix/suffix (src/, notebooks/, plan/, main.py, .ipynb); "
                    f"forced to private."
                )
                denylist.append(path)
                continue
            # The README is read by the workspace builder via _read_readme.
            # Don't put it in the data allowlist.
            if role == "task_description" or path.lower().endswith("readme.md"):
                continue
            # requirements.txt is materialised by the workspace builder too;
            # we still allowlist it because the builder uses the manifest
            # allowlist to decide what to copy into the workspace.
            allowlist.append(path)
        elif visibility == "private":
            denylist.append(path)
        else:
            warnings.append(
                f"file {path!r} has unknown visibility={visibility!r}; treating as private"
            )
            denylist.append(path)

    # Deduplicate while preserving order.
    def _dedupe(items: Sequence[str]) -> list[str]:
        seen: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.append(item)
        return seen

    readme_policy = {
        "preserve_sections": [
            "Method Hints",
            "References",
            "Data Description",
            "Problem Description",
            "Background",
        ],
        "preserve_user_eval_notes": True,
        "remove_path_patterns": [
            r"(?i)data/ground_truth\.",
            r"(?i)data/baseline_reference\.",
            r"(?i)data/simu\.",
            r"(?i)evaluation/reference_outputs/",
            r"(?i)evaluation/tests/",
            r"(?i)\bmain\.py\b",
            r"(?i)\bsrc/",
            r"(?i)\bnotebooks/",
            r"(?i)\bplan/",
        ],
        "remove_sections": [],
    }

    return {
        "public_data_allowlist": _dedupe(allowlist),
        "public_data_denylist": _dedupe(denylist),
        "readme_policy": readme_policy,
    }


def _build_runtime_layout(contract: Mapping[str, Any]) -> dict[str, str]:
    """Pick directory names for the live workspace.

    The v2 contract's ``execution.writable_paths`` lists the directories the
    agent may write to, in priority order. We map the first occurrence of
    each well-known directory to the runtime_layout used by the workspace
    builder.
    """

    writable = list((contract.get("execution") or {}).get("writable_paths") or [])
    norm = [str(p).strip().rstrip("/").rstrip("\\") for p in writable if p]
    layout = {
        "data_dir": "data",
        "work_dir": "work",
        "output_dir": "output",
        "checkpoints_dir": "checkpoints",
        "public_bundle_dir": "public_bundle",
    }
    # Honour overrides only when the contract explicitly lists them.
    for slot, default in (
        ("work_dir", "work"),
        ("output_dir", "output"),
        ("checkpoints_dir", "checkpoints"),
    ):
        for entry in norm:
            stripped = entry.split("/")[0]
            if stripped == default:
                layout[slot] = stripped
                break
    return layout


def _primary_output_path(contract: Mapping[str, Any]) -> str:
    out = contract.get("output") or {}
    path = str(out.get("path") or "").strip()
    if not path:
        raise RegistrationError("contract.output.path is required")
    return path


SETUP_STATE_REL = "runtime_logs/setup/{task_id}.json"


def _load_task_env_state(repo_root: Path, task_id: str) -> dict[str, Any] | None:
    """Read the per-task venv state file written by setup_task_env.sh.

    Returns the parsed dict, or None when the state file does not exist
    (in that case the task has *not* been provisioned through the
    standard bash workflow). Malformed JSON is treated as "not ready"
    rather than raising, so a stale file from a crashed setup never
    blocks registration silently.
    """

    state_path = repo_root / SETUP_STATE_REL.format(task_id=task_id)
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ready": False, "error": "state file is not valid JSON"}


def _detect_runtime_env(
    *,
    repo_root: Path,
    task_root: Path,
    task_id: str,
    require_task_env: bool = False,
) -> dict[str, Any]:
    """Return the runtime_env block for the manifest.

    Resolution order:

    1. If ``runtime_logs/setup/<task_id>.json`` is present and its
       ``ready`` flag is true, the per-task venv path is forwarded to
       the harness (and the judge bridge) via ``python_executable``.
    2. If the state file is missing or ``ready=false`` and
       ``require_task_env`` is True, registration *refuses* with a
       :class:`RegistrationError` telling the user to run
       ``scripts/setup_task_env.sh <task_id>``.
    3. If ``require_task_env`` is False (default for backwards-compat
       and CI), we fall back to a "ready, no per-task venv" block --
       the harness/judge will then re-use ``sys.executable``.

    The returned dict is purely declarative; running the venv setup is
    explicitly *not* a side effect of registration.
    """

    requirements = task_root / "requirements.txt"
    state = _load_task_env_state(repo_root, task_id)

    if state is not None and bool(state.get("ready")):
        return {
            "backend": "per_task_venv",
            "ready": True,
            "python_executable": str(state.get("python_executable") or ""),
            "requirements_path": state.get("requirements_path")
            or (str(requirements.resolve()) if requirements.exists() else None),
            "requirements_sha256": state.get("requirements_sha256"),
            "filtered_requirements_path": state.get("filtered_requirements_path"),
            "skipped_requirements": state.get("skipped_requirements"),
            "shared_torch_env": state.get("shared_torch_env"),
            "torch_info": state.get("torch_info"),
            "cupy_info": state.get("cupy_info"),
            "setup_state_path": str(
                (repo_root / SETUP_STATE_REL.format(task_id=task_id)).resolve()
            ),
            "created_at_unix": state.get("created_at_unix"),
        }

    if require_task_env:
        detail = (
            f"missing"
            if state is None
            else f"not ready: {state.get('error') or 'see runtime_logs/setup/'}"
        )
        raise RegistrationError(
            f"per-task venv state for task_id={task_id!r} is {detail}.\n"
            f"Run:  bash scripts/setup_task_env.sh {task_id}\n"
            f"and then re-run register-task. To bypass this check pass "
            f"--no-require-task-env."
        )

    env_block: dict[str, Any] = {
        "backend": "harness_python_fallback",
        "ready": True,
        "python_executable": "",
    }
    if requirements.exists():
        env_block["requirements_path"] = str(requirements.resolve())
    return env_block


# --------------------------------------------------------------------- main


def register_task(
    *,
    repo_root: Path,
    task_id: str,
    tasks_root: Path | None = None,
    force: bool = False,
    require_task_env: bool = False,
) -> RegistrationResult:
    """Build / refresh the registry manifest for a single task.

    Parameters
    ----------
    repo_root:
        Root of the MyEvoSkill checkout (the directory that contains
        ``registry/`` and ``src/myevoskill/``).
    task_id:
        Folder name under ``tasks_root``.
    tasks_root:
        Where the task source directories live. Defaults to
        ``repo_root.parent / "tasks"``.
    force:
        If False and a manifest already exists, the existing manifest is
        loaded and returned unchanged (after a basic sanity check). If True,
        the manifest is regenerated.
    """

    repo_root = Path(repo_root).resolve()
    if tasks_root is None:
        tasks_root = repo_root.parent / "tasks"
    tasks_root = Path(tasks_root).resolve()

    task_root = tasks_root / task_id
    if not task_root.is_dir():
        raise RegistrationError(f"task directory not found: {task_root}")

    manifest_dir = repo_root / "registry" / "tasks"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{task_id}.json"

    if manifest_path.exists() and not force:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "existing manifest at %s is unreadable, regenerating", manifest_path
            )
        else:
            if existing.get("task_id") == task_id and existing.get("ready"):
                return RegistrationResult(
                    task_id=task_id,
                    manifest_path=manifest_path,
                    manifest=existing,
                    warnings=("manifest already present; pass --force to regenerate",),
                )

    contract = _load_contract(task_root)
    declared_id = str(contract.get("task_id") or "").strip()
    if declared_id and declared_id != task_id:
        raise RegistrationError(
            f"contract task_id={declared_id!r} does not match folder name {task_id!r}"
        )

    judge_adapter_path = task_root / JUDGE_ADAPTER_REL
    if not judge_adapter_path.exists():
        raise RegistrationError(
            f"judge adapter not found: {judge_adapter_path}\n"
            f"Tasks must ship a generated judge adapter at {JUDGE_ADAPTER_REL}."
        )

    warnings: list[str] = []
    public_policy = _build_public_policy(contract, warnings=warnings)

    # Belt-and-braces: even if a future bug allowed an always-hidden path
    # to slip through ``_build_public_policy``, fail loudly here instead of
    # silently exposing the reference solution.
    leaks = [p for p in public_policy["public_data_allowlist"] if _is_always_hidden(p)]
    if leaks:
        raise RegistrationError(
            "internal invariant violated: the following paths must never "
            "appear in public_data_allowlist (they expose the reference "
            f"solution): {leaks}"
        )
    runtime_layout = _build_runtime_layout(contract)
    primary_output = _primary_output_path(contract)
    runtime_env = _detect_runtime_env(
        repo_root=repo_root,
        task_root=task_root,
        task_id=task_id,
        require_task_env=require_task_env,
    )

    # source_task_dir is stored as a path relative to repo_root (so the
    # manifest is portable across machines that mirror the same layout).
    try:
        source_task_dir_rel = os.path.relpath(task_root, repo_root)
    except ValueError:
        # Different drives on Windows: fall back to absolute path.
        source_task_dir_rel = str(task_root)
    # Always use forward slashes in the manifest for consistency.
    source_task_dir_rel = source_task_dir_rel.replace("\\", "/")

    manifest: dict[str, Any] = {
        "task_id": task_id,
        "family": contract.get("family") or "unknown",
        "ready": True,
        "source_task_dir": source_task_dir_rel,
        "primary_output_path": primary_output,
        "task_contract_path": CONTRACT_REL,
        "task_contract_public_path": CONTRACT_PUBLIC_REL,
        "judge_adapter_path": JUDGE_ADAPTER_REL,
        "public_policy": public_policy,
        "runtime_layout": runtime_layout,
        "runtime_policy": {
            "execution_budget_seconds": 7200,
            "model_timeout_seconds": 240,
        },
        "runtime_env": runtime_env,
        "public_eval_spec": {"alignments": []},
    }

    # Write atomically.
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(manifest_path)

    return RegistrationResult(
        task_id=task_id,
        manifest_path=manifest_path,
        manifest=manifest,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------- CLI shim


def main(argv: Sequence[str] | None = None) -> int:
    """Tiny argparse front-end so ``python -m myevoskill.registration`` works."""

    import argparse

    parser = argparse.ArgumentParser(prog="myevoskill.registration")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--tasks-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-task-env",
        action="store_true",
        help=(
            "refuse to register the task unless runtime_logs/setup/<task>.json "
            "is present and ready=true (use scripts/setup_task_env.sh first)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        result = register_task(
            repo_root=Path(args.repo_root),
            task_id=args.task_id,
            tasks_root=Path(args.tasks_root) if args.tasks_root else None,
            force=bool(args.force),
            require_task_env=bool(args.require_task_env),
        )
    except RegistrationError as exc:
        print(f"registration failed: {exc}", file=__import__("sys").stderr)
        return 2

    print(f"registered: {result.manifest_path}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
