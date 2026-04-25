"""Tests for the deterministic v2 task registration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myevoskill.registration import (
    RegistrationError,
    register_task,
)


# --------------------------------------------------------------------- fixtures


def _write_v2_task(tasks_root: Path, task_id: str, **overrides) -> Path:
    """Materialise a minimal but valid v2 task on disk."""
    task_root = tasks_root / task_id
    (task_root / "data").mkdir(parents=True, exist_ok=True)
    (task_root / "evaluation").mkdir(parents=True, exist_ok=True)
    (task_root / "src").mkdir(parents=True, exist_ok=True)

    (task_root / "README.md").write_text("# task readme\n", encoding="utf-8")
    (task_root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (task_root / "data" / "raw_data.npz").write_bytes(b"\x00")
    (task_root / "data" / "meta_data.json").write_text("{}", encoding="utf-8")
    (task_root / "data" / "ground_truth.npz").write_bytes(b"\x00")
    (task_root / "src" / "visualization.py").write_text("def compute_metrics(): pass\n", encoding="utf-8")
    (task_root / "evaluation" / "judge_adapter.py").write_text("# generated\n", encoding="utf-8")

    contract = {
        "version": 2,
        "task_id": task_id,
        "family": "medicine",
        "files": [
            {"id": "readme", "path": "README.md", "visibility": "public", "role": "task_description"},
            {"id": "requirements", "path": "requirements.txt", "visibility": "public", "role": "runtime_dependencies"},
            {"id": "raw_data", "path": "data/raw_data.npz", "visibility": "public", "role": "input_data"},
            {"id": "meta_data", "path": "data/meta_data.json", "visibility": "public", "role": "metadata"},
            {"id": "ground_truth", "path": "data/ground_truth.npz", "visibility": "private", "role": "reference_data"},
            {"id": "metric_helper", "path": "src/visualization.py", "visibility": "private", "role": "metric_helper"},
        ],
        "execution": {
            "readable_files": ["readme", "raw_data", "meta_data", "requirements"],
            "entrypoint": "work/main.py",
            "writable_paths": ["work/", "output/", "checkpoints/"],
        },
        "output": {"path": "output/reconstruction.npz", "format": "npz"},
        "metrics": [],
    }
    contract.update(overrides)
    (task_root / "evaluation" / "task_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    return task_root


# --------------------------------------------------------------------- tests


def test_register_writes_manifest(tmp_path: Path):
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    _write_v2_task(tasks_root, "demo")

    result = register_task(repo_root=repo_root, task_id="demo", tasks_root=tasks_root)

    assert result.manifest_path.exists()
    m = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert m["task_id"] == "demo"
    assert m["family"] == "medicine"
    assert m["ready"] is True
    assert m["primary_output_path"] == "output/reconstruction.npz"
    assert m["judge_adapter_path"] == "evaluation/judge_adapter.py"
    assert m["task_contract_path"] == "evaluation/task_contract.json"


def test_public_policy_separates_public_and_private(tmp_path: Path):
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    _write_v2_task(tasks_root, "demo")

    result = register_task(repo_root=repo_root, task_id="demo", tasks_root=tasks_root)
    pp = result.manifest["public_policy"]

    # README should NOT be in the data allowlist (workspace builder handles it).
    assert "README.md" not in pp["public_data_allowlist"]
    # public input data should be allowlisted.
    assert "data/raw_data.npz" in pp["public_data_allowlist"]
    assert "data/meta_data.json" in pp["public_data_allowlist"]
    assert "requirements.txt" in pp["public_data_allowlist"]
    # private files must be denied.
    assert "data/ground_truth.npz" in pp["public_data_denylist"]
    assert "src/visualization.py" in pp["public_data_denylist"]


def test_missing_contract_raises(tmp_path: Path):
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    (tasks_root / "broken").mkdir()

    with pytest.raises(RegistrationError, match="task contract not found"):
        register_task(repo_root=repo_root, task_id="broken", tasks_root=tasks_root)


def test_missing_judge_adapter_raises(tmp_path: Path):
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    task_root = _write_v2_task(tasks_root, "demo")
    (task_root / "evaluation" / "judge_adapter.py").unlink()

    with pytest.raises(RegistrationError, match="judge adapter not found"):
        register_task(repo_root=repo_root, task_id="demo", tasks_root=tasks_root)


def test_task_id_mismatch_raises(tmp_path: Path):
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    task_root = _write_v2_task(tasks_root, "demo")
    # Corrupt the contract so its task_id no longer matches the folder name.
    contract_path = task_root / "evaluation" / "task_contract.json"
    data = json.loads(contract_path.read_text(encoding="utf-8"))
    data["task_id"] = "other_id"
    contract_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with pytest.raises(RegistrationError, match="does not match folder name"):
        register_task(repo_root=repo_root, task_id="demo", tasks_root=tasks_root)


def test_idempotent_without_force(tmp_path: Path):
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    _write_v2_task(tasks_root, "demo")

    first = register_task(repo_root=repo_root, task_id="demo", tasks_root=tasks_root)
    mtime_first = first.manifest_path.stat().st_mtime_ns

    second = register_task(repo_root=repo_root, task_id="demo", tasks_root=tasks_root)
    # No regeneration without --force.
    assert second.manifest_path.stat().st_mtime_ns == mtime_first
    assert any("already present" in w for w in second.warnings)

    third = register_task(repo_root=repo_root, task_id="demo", tasks_root=tasks_root, force=True)
    assert third.manifest_path.stat().st_mtime_ns >= mtime_first


def test_missing_output_path_raises(tmp_path: Path):
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    _write_v2_task(tasks_root, "demo", output={})

    with pytest.raises(RegistrationError, match="contract.output.path is required"):
        register_task(repo_root=repo_root, task_id="demo", tasks_root=tasks_root)
