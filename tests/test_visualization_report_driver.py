from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_driver():
    path = ROOT / "scripts" / "run_model_visualize.py"
    spec = importlib.util.spec_from_file_location("run_model_visualize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_common():
    path = ROOT / "scripts" / "_visualize_common.py"
    spec = importlib.util.spec_from_file_location("_visualize_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_last_json_line_ignores_progress_text():
    driver = _load_driver()
    payload = driver._parse_last_json_line('hello\n{"ok": false}\nprogress\n{"ok": true}\n')
    assert payload == {"ok": True}


def test_select_run_prefers_latest_pass(tmp_path: Path):
    driver = _load_driver()
    repo = tmp_path
    task_id = "demo"
    slug = "model-a"
    workspace = repo / "artifacts" / "workspaces" / slug / task_id / "run-pass"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "reconstruction.npz").write_bytes(b"placeholder")

    log_root = repo / "artifacts" / "logs" / slug / task_id
    fail_summary = log_root / "run-fail" / "run_summary.json"
    pass_summary = log_root / "run-pass" / "run_summary.json"
    fail_summary.parent.mkdir(parents=True)
    pass_summary.parent.mkdir(parents=True)
    fail_summary.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "run_id": "run-fail",
                "verdict": "FAIL",
                "workspace_root": str(repo / "missing"),
            }
        ),
        encoding="utf-8",
    )
    pass_summary.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "run_id": "run-pass",
                "verdict": "PASS",
                "workspace_root": str(workspace),
            }
        ),
        encoding="utf-8",
    )

    selected = driver._select_run(
        repo_root=repo,
        model_slug_value=slug,
        task_id=task_id,
        manifest={"primary_output_path": "output/reconstruction.npz"},
        prefer_verdict="PASS",
    )

    assert selected is not None
    assert selected.run_id == "run-pass"
    assert selected.selected_reason == "latest_pass"


def test_render_image_triplet_writes_png(tmp_path: Path):
    common = _load_common()
    gt = np.eye(8)
    baseline = gt * 0.5
    agent = gt.copy()
    dest = tmp_path / "comparison.png"

    metrics = common.render_image_triplet(
        gt=gt,
        baseline=baseline,
        agent=agent,
        dest=dest,
        title="demo",
    )

    assert dest.exists()
    assert dest.stat().st_size > 0
    assert metrics["agent_nrmse"] == 0.0
    assert metrics["agent_nrmse"] < metrics["baseline_nrmse"]
