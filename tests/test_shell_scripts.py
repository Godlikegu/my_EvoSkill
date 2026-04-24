from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_create_dev_env_script_targets_named_conda_env():
    script = (ROOT / "scripts" / "create_dev_env.sh").read_text(encoding="utf-8")
    assert 'conda env create -n "${ENV_NAME}"' in script
    assert 'conda env update -n "${ENV_NAME}"' in script
    assert "./.conda_env" not in script


def test_run_task_live_script_sets_pythonpath_and_accepts_help():
    env = dict(os.environ)
    env["CONDA_DEFAULT_ENV"] = "myevoskill"
    completed = subprocess.run(
        [str(ROOT / "scripts" / "run_task_live.sh"), "--help"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0
    assert "--executor" in completed.stdout


def test_run_task_register_script_sets_pythonpath_and_accepts_help():
    env = dict(os.environ)
    env["CONDA_DEFAULT_ENV"] = "myevoskill"
    completed = subprocess.run(
        [str(ROOT / "scripts" / "run_task_register.sh"), "--help"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0
    assert "--task-root" in completed.stdout
