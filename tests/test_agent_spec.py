"""Unit tests for the agent-visible task spec derivation.

These tests pin down two invariants that are easy to break in a refactor:

    1. ``derive_agent_task_spec`` exposes the *output schema* (path, format,
       required keys with dtype + shape) and the *runtime budget*, so an
       agent can produce a file the judge will accept.
    2. The emitted spec contains **no** ground-truth path, judge helper
       reference, threshold, or metric name -- a future contract refactor
       must not be able to silently leak those.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from myevoskill.workspace.agent_spec import (
    LEAK_WORDS,
    assert_spec_has_no_leaks,
    derive_agent_task_spec,
    render_summary,
)


# Minimal v2-shaped contract that mirrors what
# ``contract_generation.derive_contract`` writes to disk.
_CONTRACT: dict[str, Any] = {
    "schema_version": 2,
    "task_id": "demo_task",
    "family": "demo_family",
    "files": [
        {
            "path": "data/measurements.npz",
            "role": "input_data",
            "visibility": "public",
            "format": "npz",
            "fields": {
                "y": {"dtype": "complex64", "shape": [128, 128]},
                "kx": {"dtype": "float32", "shape": [128]},
            },
        },
        {
            "path": "data/meta_data.json",
            "role": "metadata",
            "visibility": "public",
            "format": "json",
        },
        # These three MUST be filtered out: they are private.
        {
            "path": "evaluation/ground_truth.npz",
            "role": "ground_truth",
            "visibility": "private",
            "format": "npz",
        },
        {
            "path": "evaluation/task_contract.json",
            "role": "task_contract",
            "visibility": "private",
        },
        {
            "path": "src/visualization.py",
            "role": "reference_solution",
            "visibility": "private",
        },
    ],
    "output": {
        "path": "output/reconstruction.npz",
        "format": "npz",
        "fields": [
            {
                "name": "image",
                "dtype": "float32",
                "shape": [256, 256],
                "semantics": "Reconstructed image, real-valued.",
            },
            {
                "name": "phase",
                "dtype": "float32",
                "shape": [256, 256],
                "semantics": "Reconstructed phase in radians.",
            },
        ],
    },
    "evaluation": {
        # These thresholds and metric names MUST NOT appear in the spec.
        "metrics": [
            {"name": "ncc", "threshold": 0.92, "direction": "higher_is_better"},
            {"name": "nrmse", "threshold": 0.15, "direction": "lower_is_better"},
        ],
        "judge_adapter": "evaluation/judge_adapter.py",
    },
    "execution": {
        "entrypoint": "work/main.py",
        "writable_paths": ["work/", "output/"],
    },
}

_MANIFEST: dict[str, Any] = {
    "task_id": "demo_task",
    "family": "demo_family",
    "runtime_layout": {"data_dir": "data", "work_dir": "work", "output_dir": "output"},
    "runtime_policy": {"execution_budget_seconds": 3600},
}


def test_spec_exposes_output_schema() -> None:
    spec = derive_agent_task_spec(contract=_CONTRACT, manifest=_MANIFEST)
    out = spec["output"]
    assert out["path"] == "output/reconstruction.npz"
    assert out["format"] == "npz"
    names = [k["name"] for k in out["required_keys"]]
    assert names == ["image", "phase"]
    image = out["required_keys"][0]
    assert image["dtype"] == "float32"
    assert image["shape"] == [256, 256]


def test_spec_exposes_public_inputs_only() -> None:
    spec = derive_agent_task_spec(contract=_CONTRACT, manifest=_MANIFEST)
    paths = [f["path"] for f in spec["inputs"]["files"]]
    assert "data/measurements.npz" in paths
    assert "data/meta_data.json" in paths
    # Private files MUST be filtered out, regardless of role.
    assert not any("evaluation/" in p for p in paths)
    assert not any("src/" in p for p in paths)
    assert not any("ground_truth" in p.lower() for p in paths)


def test_spec_exposes_runtime_budget() -> None:
    spec = derive_agent_task_spec(contract=_CONTRACT, manifest=_MANIFEST)
    assert spec["runtime"]["wall_clock_seconds"] == 3600
    assert spec["runtime"]["entrypoint"] == "work/main.py"
    assert "work/" in spec["runtime"]["writable_paths"]
    assert "plan.md" in spec["runtime"]["writable_paths"]


@pytest.mark.parametrize("leak", ["ncc", "nrmse", "0.92", "0.15"])
def test_spec_does_not_leak_metric_names_or_thresholds(leak: str) -> None:
    spec = derive_agent_task_spec(contract=_CONTRACT, manifest=_MANIFEST)
    blob = json.dumps(spec).lower()
    assert leak.lower() not in blob, f"spec leaked {leak!r}: {blob}"


def test_assert_spec_has_no_leaks_passes_on_clean_spec() -> None:
    spec = derive_agent_task_spec(contract=_CONTRACT, manifest=_MANIFEST)
    # Should not raise.
    assert_spec_has_no_leaks(spec)


def test_assert_spec_has_no_leaks_rejects_dirty_spec() -> None:
    dirty = {
        "inputs": {"files": [{"path": "evaluation/judge_adapter.py"}]},
    }
    with pytest.raises(ValueError) as excinfo:
        assert_spec_has_no_leaks(dirty)
    msg = str(excinfo.value).lower()
    assert "judge_adapter" in msg or "evaluation/" in msg


def test_render_summary_mentions_output_path_and_keys() -> None:
    spec = derive_agent_task_spec(contract=_CONTRACT, manifest=_MANIFEST)
    summary = render_summary(spec)
    assert "output/reconstruction.npz" in summary
    assert "image" in summary
    assert "phase" in summary
    assert "256x256" in summary  # shape rendered as compact form
    # The summary is what gets inlined in the user prompt -- it must also
    # be free of leak words.
    blob = summary.lower()
    for leak in ("ncc", "nrmse", "threshold", "ground_truth"):
        assert leak not in blob


def test_leak_words_does_not_include_friendly_tokens() -> None:
    # Sanity: we deliberately allow tokens like "judge_verdict" and
    # "evaluation_protocol" inside the spec, even though they share a
    # prefix with private items. Make sure the leak list never grows to
    # cover them, otherwise the leak guard will false-positive.
    forbidden_substrings = {w.lower() for w in LEAK_WORDS}
    assert "judge_verdict" not in forbidden_substrings
    assert "evaluation_protocol" not in forbidden_substrings
    assert "judge" not in forbidden_substrings  # bare word would over-match
