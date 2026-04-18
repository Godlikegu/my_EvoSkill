"""Development/runtime environment helpers."""

from __future__ import annotations

import os
import sys

from .models import DevEnvironmentInfo, RuntimeEnvironmentPolicy


def detect_dev_environment(
    policy: RuntimeEnvironmentPolicy | None = None,
) -> DevEnvironmentInfo:
    policy = policy or RuntimeEnvironmentPolicy()
    conda_env_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    return DevEnvironmentInfo(
        conda_env_name=conda_env_name,
        conda_prefix=conda_prefix,
        python_executable=sys.executable,
        allow_bridge_execution=policy.allow_bridge_execution,
        allow_task_env_build=policy.allow_task_env_build,
    )
