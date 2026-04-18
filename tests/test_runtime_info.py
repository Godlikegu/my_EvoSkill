import json
import os
import subprocess
from pathlib import Path

from myevoskill.models import RuntimeEnvironmentPolicy
from myevoskill.runtime_info import detect_dev_environment


def test_detect_dev_environment_returns_policy_values():
    info = detect_dev_environment(
        RuntimeEnvironmentPolicy(
            dev_environment_name="myevoskill",
            allow_bridge_execution=False,
            allow_task_env_build=True,
        )
    )
    assert isinstance(info.conda_env_name, str)
    assert info.allow_bridge_execution is False
    assert info.allow_task_env_build is True


def test_print_env_info_script_outputs_expected_keys():
    completed = subprocess.run(
        ["python", str(Path(__file__).resolve().parents[1] / "scripts" / "print_env_info.py")],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    payload = json.loads(completed.stdout)
    assert "conda_env_name" in payload
    assert "python_executable" in payload


