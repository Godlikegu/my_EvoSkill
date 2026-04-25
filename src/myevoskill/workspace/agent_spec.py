"""Build the agent-visible ``agent_task_spec.json`` from a v2 task contract.

The spec is the *machine-readable* IO contract the agent receives. It is
strictly more restrictive than the full ``task_contract.json``:

* It does **not** mention metric names, thresholds, helper callables,
  ``ground_truth`` paths, ``evaluation/`` paths, or any private file id.
* It does **not** reveal which output field is compared against which
  reference array.
* It **does** expose the output schema (path, format, required keys with
  dtype + shape) so the agent can produce a file the judge will accept --
  this is the whole reason we ship a spec at all.

The result is consumed by ``workspace.builder`` (which copies it into the
agent workspace as ``agent_task_spec.json``) and by ``harness.prompts``
(which inlines a one-paragraph summary into the first-round user turn).

This module is pure and side-effect-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Substrings that must never appear (case-insensitive) anywhere in the emitted
# JSON.  We assert this in tests so a future refactor cannot accidentally
# leak threshold / metric-name / ground-truth information through the spec.
#
# We deliberately use *anchored* substrings rather than bare words like
# "judge" or "evaluation", because the spec legitimately contains tokens
# such as ``judge_verdict`` (the feedback channel name) and
# ``evaluation_protocol`` (the section name).  Only the *path-shaped* and
# *implementation-detail* leak forms are forbidden.
# NOTE: We deliberately do *not* list bare words like "threshold" or
# "metric": the leak guard checks ``json.dumps(spec)``, which legitimately
# contains a help-text sentence such as "...pass thresholds are
# intentionally hidden." The numeric thresholds and concrete metric names
# are covered by a parametrized leak test (see ``tests/test_agent_spec.py``)
# rather than by a substring sweep, so the spec text can stay friendly.
#
# We also do *not* list ``.ipynb`` here even though it is a forbidden
# substring at runtime: the spec advertises ``*.ipynb`` under
# ``forbidden.paths`` precisely so the agent knows notebooks are off-limits.
LEAK_WORDS: tuple[str, ...] = (
    "ground_truth",
    "judge_adapter",
    "evaluation/",
    "evaluation\\",
    "metric_helper",
    "task_contract",
    "registration_contract",
    "reference_outputs",
)


def _coerce_int_list(shape: Any) -> list[Any]:
    """Pass ``shape`` through verbatim, but reject anything weird."""

    if shape is None:
        return []
    if isinstance(shape, (list, tuple)):
        out: list[Any] = []
        for dim in shape:
            if isinstance(dim, bool):
                continue
            if isinstance(dim, int) or (isinstance(dim, str) and dim):
                out.append(dim)
        return out
    return []


def _normalise_input_files(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Filter ``contract.files`` to the agent-visible inputs only.

    We keep ``visibility == "public"`` files that play an *input* role:
    ``input_data``, ``metadata``, ``runtime_dependencies`` -- the things
    the agent must read to solve the task.  We deliberately drop the
    ``task_description`` (README, surfaced separately) so the spec stays
    purely structured.
    """

    keep_roles = {"input_data", "metadata", "runtime_dependencies", "auxiliary"}
    out: list[dict[str, Any]] = []
    for entry in contract.get("files") or []:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("visibility") or "").lower() != "public":
            continue
        role = str(entry.get("role") or "").lower()
        path = str(entry.get("path") or "").strip()
        if not path:
            continue
        if role not in keep_roles:
            continue
        item: dict[str, Any] = {
            "path": path.replace("\\", "/"),
            "role": role,
        }
        fmt = entry.get("format")
        if fmt:
            item["format"] = str(fmt)
        elif path.lower().endswith(".npz"):
            item["format"] = "npz"
        elif path.lower().endswith(".json"):
            item["format"] = "json"
        elif path.lower().endswith(".csv"):
            item["format"] = "csv"
        elif path.lower().endswith(".txt"):
            item["format"] = "txt"
        fields = entry.get("fields")
        if isinstance(fields, Mapping) and fields:
            item["fields"] = {
                str(k): {
                    "dtype": str(v.get("dtype")) if isinstance(v, Mapping) else None,
                    "shape": _coerce_int_list(v.get("shape")) if isinstance(v, Mapping) else [],
                }
                for k, v in fields.items()
            }
        out.append(item)
    return out


def _normalise_output(contract: Mapping[str, Any]) -> dict[str, Any]:
    out_block = contract.get("output") or {}
    path = str(out_block.get("path") or "").strip().replace("\\", "/")
    fmt = str(out_block.get("format") or "").strip().lower()
    if not fmt:
        if path.lower().endswith(".npz"):
            fmt = "npz"
        elif path.lower().endswith(".npy"):
            fmt = "npy"
        elif path.lower().endswith(".json"):
            fmt = "json"

    required: list[dict[str, Any]] = []
    for entry in out_block.get("fields") or []:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        required.append(
            {
                "name": name,
                "dtype": str(entry.get("dtype") or ""),
                "shape": _coerce_int_list(entry.get("shape")),
                "description": str(entry.get("semantics") or "").strip(),
            }
        )

    return {
        "path": path,
        "format": fmt,
        "required_keys": required,
        "optional_keys": [],
    }


def derive_agent_task_spec(
    *,
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce the agent-visible spec dict.

    Parameters
    ----------
    contract:
        The full ``evaluation/task_contract.json`` (v2) loaded as dict.
    manifest:
        The registry manifest, used only for ``runtime_layout`` and
        ``runtime_policy``; *no* private fields from it are leaked.
    """

    runtime_layout = dict(manifest.get("runtime_layout") or {})
    runtime_policy = dict(manifest.get("runtime_policy") or {})
    execution = dict(contract.get("execution") or {})

    writable_paths = [
        str(p).replace("\\", "/")
        for p in (execution.get("writable_paths") or ["work/", "output/", "checkpoints/"])
    ]
    # The agent's plan.md is also writable -- expose this so the agent
    # knows it is allowed to update the plan file in-place.
    if "plan.md" not in writable_paths:
        writable_paths.append("plan.md")

    spec: dict[str, Any] = {
        "schema_version": 1,
        "task_id": str(contract.get("task_id") or manifest.get("task_id") or ""),
        "family": str(contract.get("family") or manifest.get("family") or "unknown"),
        "inputs": {
            "readme": "README.md",
            "meta_data": "meta_data.json",
            "files": _normalise_input_files(contract),
        },
        "output": _normalise_output(contract),
        "runtime": {
            "entrypoint": str(execution.get("entrypoint") or "work/main.py"),
            "writable_paths": writable_paths,
            "wall_clock_seconds": int(
                runtime_policy.get("execution_budget_seconds") or 7200
            ),
        },
        "evaluation_protocol": {
            "mode": "pass_fail",
            "feedback_channel": "judge_verdict",
            "note": (
                "After your code finishes, the harness will judge the "
                "primary output file. You will receive PASS / FAIL / INVALID. "
                "Numeric metric values, metric names, and pass thresholds "
                "are intentionally hidden."
            ),
        },
        "forbidden": {
            "paths": [
                "../",
                "src/",
                "notebooks/",
                "*.ipynb",
            ],
            "actions": [
                "network",
                "package_install",
                "writes_outside_writable_paths",
                "absolute_paths_in_bash",
                "cd_outside_workspace",
            ],
        },
    }

    return spec


def assert_spec_has_no_leaks(spec: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if the spec contains any forbidden leak word.

    This is a defensive check called from ``builder.build_workspace`` after
    deriving the spec, so a future contract refactor can never accidentally
    smuggle a metric name or ground-truth path into the agent's view.
    """

    blob = json.dumps(spec, ensure_ascii=False).lower()
    leaks = [w for w in LEAK_WORDS if w.lower() in blob]
    if leaks:
        raise ValueError(
            "agent_task_spec.json contains forbidden leak words: "
            f"{leaks!r}. This is a programming error; the spec must not "
            "expose threshold/metric/ground-truth information."
        )


def render_summary(spec: Mapping[str, Any]) -> str:
    """One-paragraph summary inlined into the first-round user prompt.

    Keep it short; the agent can read the full ``agent_task_spec.json``
    on disk.
    """

    out_block = spec.get("output") or {}
    out_path = out_block.get("path") or "(missing)"
    fmt = (out_block.get("format") or "").upper() or "?"
    keys = out_block.get("required_keys") or []
    if keys:
        key_lines = []
        for k in keys:
            shape = k.get("shape") or []
            shape_s = "x".join(str(s) for s in shape) or "scalar"
            key_lines.append(
                f"      - `{k.get('name')}` ({k.get('dtype') or '?'}, shape {shape_s})"
            )
        keys_block = "\n".join(key_lines)
    else:
        keys_block = "      (no required output keys declared in the spec)"

    runtime = spec.get("runtime") or {}
    return (
        "Output schema (excerpted from `agent_task_spec.json`):\n"
        f"  * path:   `{out_path}`\n"
        f"  * format: {fmt}\n"
        "  * required keys:\n"
        f"{keys_block}\n"
        "  * writable paths: "
        + ", ".join(f"`{p}`" for p in runtime.get("writable_paths") or [])
    )
