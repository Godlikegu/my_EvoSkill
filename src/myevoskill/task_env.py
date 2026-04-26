"""Cross-platform per-task Python environment setup.

This is the Python equivalent of ``scripts/setup_task_env.sh``.  It creates a
task-local virtualenv under ``.venvs/<task_id>``, installs the task
``requirements.txt`` when present, and writes the state consumed by
``register-task --require-task-env``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SETUP_STATE_REL = "runtime_logs/setup/{task_id}.json"


class TaskEnvSetupError(RuntimeError):
    """Raised when a per-task environment cannot be prepared."""


@dataclass(frozen=True)
class TaskEnvSetupResult:
    task_id: str
    ready: bool
    state_path: Path
    venv_dir: Path
    python_executable: Path
    requirements_path: Path | None
    requirements_sha256: str | None
    pip_log: Path


def setup_task_env(
    *,
    repo_root: Path,
    task_id: str,
    tasks_root: Path | None = None,
    force: bool = False,
    base_python: Path | None = None,
) -> TaskEnvSetupResult:
    """Create/update the per-task virtualenv and write its setup state."""

    repo_root = Path(repo_root).resolve()
    if tasks_root is None:
        tasks_root = repo_root.parent / "tasks"
    tasks_root = Path(tasks_root).resolve()
    task_dir = tasks_root / task_id
    if not task_dir.is_dir():
        raise TaskEnvSetupError(f"task directory not found: {task_dir}")

    base_python = Path(base_python).resolve() if base_python else Path(sys.executable).resolve()
    _ensure_supported_python(base_python)

    venv_dir = repo_root / ".venvs" / task_id
    setup_dir = repo_root / "runtime_logs" / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    state_path = setup_dir / f"{task_id}.json"
    pip_log = setup_dir / f"{task_id}.pip.log"
    requirements_path = task_dir / "requirements.txt"
    req_hash = _sha256_file(requirements_path) if requirements_path.exists() else None

    if force and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        subprocess.run(
            [str(base_python), "-m", "venv", str(venv_dir)],
            check=True,
            cwd=str(repo_root),
        )

    venv_python = _find_venv_python(venv_dir)
    if requirements_path.exists():
        try:
            with pip_log.open("ab") as log:
                subprocess.run(
                    [
                        str(venv_python),
                        "-m",
                        "pip",
                        "install",
                        "--retries",
                        "10",
                        "--timeout",
                        "120",
                        "-r",
                        str(requirements_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=str(repo_root),
                )
        except subprocess.CalledProcessError as exc:
            _write_state(
                state_path,
                {
                    "task_id": task_id,
                    "ready": False,
                    "error": f"pip install failed: {exc}",
                    "python_executable": str(venv_python.resolve()),
                    "requirements_path": str(requirements_path.resolve()),
                    "requirements_sha256": req_hash,
                    "created_at_unix": int(time.time()),
                    "pip_log": str(pip_log.resolve()),
                },
            )
            raise TaskEnvSetupError(f"pip install failed; see {pip_log}") from exc

    resolved_py = _resolved_python(venv_python)
    state = {
        "task_id": task_id,
        "ready": True,
        "python_executable": str(resolved_py),
        "requirements_path": str(requirements_path.resolve()) if requirements_path.exists() else None,
        "requirements_sha256": req_hash,
        "created_at_unix": int(time.time()),
        "pip_log": str(pip_log.resolve()),
    }
    _write_state(state_path, state)

    return TaskEnvSetupResult(
        task_id=task_id,
        ready=True,
        state_path=state_path,
        venv_dir=venv_dir,
        python_executable=resolved_py,
        requirements_path=requirements_path.resolve() if requirements_path.exists() else None,
        requirements_sha256=req_hash,
        pip_log=pip_log,
    )


def _ensure_supported_python(python: Path) -> None:
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise TaskEnvSetupError(f"{python} is older than Python 3.9 or cannot run")


def _find_venv_python(venv_dir: Path) -> Path:
    candidates = [
        venv_dir / "Scripts" / "python.exe",
        venv_dir / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise TaskEnvSetupError(f"venv python not found inside {venv_dir}")


def _resolved_python(python: Path) -> Path:
    completed = subprocess.run(
        [str(python), "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(completed.stdout.strip()).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
