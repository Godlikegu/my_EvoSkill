#!/usr/bin/env python3
"""Print current MyEvoSkill development environment information."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from myevoskill.runtime_info import detect_dev_environment


def main() -> None:
    info = detect_dev_environment()
    print(
        json.dumps(
            {
                "conda_env_name": info.conda_env_name,
                "conda_prefix": info.conda_prefix,
                "python_executable": info.python_executable,
                "allow_bridge_execution": info.allow_bridge_execution,
                "allow_task_env_build": info.allow_task_env_build,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
