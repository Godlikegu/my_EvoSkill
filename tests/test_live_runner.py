from __future__ import annotations

import json
from pathlib import Path

import pytest

from myevoskill.live_runner import (
    evaluate_manifest_run,
    load_registered_manifest,
    resolve_registered_task_root,
)
from myevoskill.registration_contract import ensure_live_ready_manifest
from myevoskill.models import JudgeResult, RunRecord


def test_load_registered_manifest_reads_registry_task(tmp_path):
    project_root = tmp_path / "project"
    registry_root = project_root / "registry" / "tasks"
    registry_root.mkdir(parents=True, exist_ok=True)
    manifest_path = registry_root / "demo_task.json"
    manifest_path.write_text(
        json.dumps({"task_id": "demo_task", "source_task_dir": "../tasks/demo_task"}),
        encoding="utf-8",
    )

    manifest = load_registered_manifest("demo_task", project_root=project_root)

    assert manifest["task_id"] == "demo_task"


def test_resolve_registered_task_root_uses_project_relative_source_dir(tmp_path):
    project_root = tmp_path / "project"
    task_root = tmp_path / "tasks" / "demo_task"
    task_root.mkdir(parents=True, exist_ok=True)

    resolved = resolve_registered_task_root(
        {"task_id": "demo_task", "source_task_dir": "../tasks/demo_task"},
        project_root=project_root,
    )

    assert resolved == task_root.resolve()


def test_evaluate_manifest_run_loads_task_local_judge_adapter(tmp_path):
    task_root = tmp_path / "task_root"
    adapter_dir = task_root / "evaluation"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "judge_adapter.py").write_text(
        "\n".join(
            [
                "from myevoskill.models import JudgeResult",
                "",
                "def evaluate_run(task_root, run_record, manifest):",
                "    return JudgeResult(",
                "        task_id=run_record.task_id,",
                "        all_metrics_passed=True,",
                "        metrics_actual={'demo_metric': 1.0},",
                "        failed_metrics=[],",
                "        failure_tags=[],",
                "    )",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "task_id": "demo_task",
        "judge_spec": {
            "adapter_path": "evaluation/judge_adapter.py",
            "callable": "evaluate_run",
        },
    }
    run_record = RunRecord(
        run_id="run-1",
        task_id="demo_task",
        provider="local",
        env_hash="env-1",
        skills_active=(),
        workspace_root=tmp_path / "workspace",
    )

    result = evaluate_manifest_run(task_root, run_record, manifest)

    assert isinstance(result, JudgeResult)
    assert result.all_metrics_passed is True
    assert result.metrics_actual["demo_metric"] == 1.0


def test_live_runner_gate_rejects_manifest_without_ready_judge(tmp_path):
    task_root = tmp_path / "task_root"
    (task_root / "evaluation").mkdir(parents=True, exist_ok=True)
    contract_path = task_root / "evaluation" / "registration_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "task_id": "demo_task",
                "family": "optics",
                "resources": [
                    {
                        "path": "README.md",
                        "role": "task_description",
                        "visibility": "public",
                        "semantics": "Task description.",
                        "authority": "authoritative",
                    }
                ],
                "output_contract": {
                    "path": "output/reconstruction.npz",
                    "format": "npz",
                    "required_fields": ["signal"],
                    "numeric_fields": ["signal"],
                    "same_shape_fields": ["signal"],
                },
                "judge_contract": {
                    "metrics": [
                        {
                            "name": "score",
                            "kind": "standard",
                            "description": "Demo score.",
                            "mode": "ncc",
                            "output_field": "signal",
                            "reference_resource_path": "data/raw_data.npz",
                            "reference_field": "signal",
                            "pass_condition": {"operator": ">=", "threshold": 0.9},
                        }
                    ]
                },
                "execution_conventions": {
                    "read_first": ["README_public.md"],
                    "readable_paths": ["README_public.md"],
                    "writable_paths": ["work", "output", "checkpoints"],
                    "entrypoint": "work/main.py",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = {
        "task_id": "demo_task",
        "judge_spec": {
            "ready": False,
            "registration_contract_path": "evaluation/registration_contract.json",
        },
    }

    with pytest.raises(RuntimeError, match="judge_spec.ready is false"):
        ensure_live_ready_manifest(manifest, task_root=task_root)
