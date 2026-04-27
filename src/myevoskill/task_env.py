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
DEFAULT_TORCH_VERSION = "2.5.1"
DEFAULT_TORCHVISION_VERSION = "0.20.1"
DEFAULT_TORCHAUDIO_VERSION = "2.5.1"
DEFAULT_TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu118"
DEFAULT_SHARED_TORCH_ENV_REL = ".venvs/_torch-cu118-py310"
TORCH_REQUIREMENT_NAMES = {"torch", "torchvision", "torchaudio"}
NOTEBOOK_REQUIREMENT_NAMES = {
    "ipykernel",
    "ipywidgets",
    "jupyter",
    "jupyter-console",
    "jupyterlab",
    "nbconvert",
    "notebook",
}


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
    filtered_requirements_path: Path | None = None
    skipped_requirements: list[str] | None = None
    shared_torch_env: Path | None = None
    torch_info: dict[str, Any] | None = None
    cupy_info: dict[str, Any] | None = None


def setup_task_env(
    *,
    repo_root: Path,
    task_id: str,
    tasks_root: Path | None = None,
    force: bool = False,
    base_python: Path | None = None,
    shared_torch_env: Path | None = None,
    torch_cuda_index_url: str = DEFAULT_TORCH_CUDA_INDEX_URL,
    torch_version: str = DEFAULT_TORCH_VERSION,
    require_gpu_torch: bool = False,
    skip_notebook_packages: bool = True,
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
    requirements_lines = (
        requirements_path.read_text(encoding="utf-8").splitlines()
        if requirements_path.exists()
        else []
    )
    has_torch = _has_requirement(requirements_lines, TORCH_REQUIREMENT_NAMES)
    has_cupy = _has_requirement_prefix(requirements_lines, ("cupy", "cupy-cuda"))
    filtered_requirements_path: Path | None = None
    skipped_requirements: list[str] = []
    torch_info: dict[str, Any] | None = None
    cupy_info: dict[str, Any] | None = None

    if force and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        subprocess.run(
            [str(base_python), "-m", "venv", str(venv_dir)],
            check=True,
            cwd=str(repo_root),
        )

    venv_python = _find_venv_python(venv_dir)

    if has_torch:
        if shared_torch_env is None:
            shared_torch_env = repo_root / DEFAULT_SHARED_TORCH_ENV_REL
        else:
            shared_torch_env = Path(shared_torch_env)
            if not shared_torch_env.is_absolute():
                shared_torch_env = repo_root / shared_torch_env
        shared_torch_env = shared_torch_env.resolve()
        torch_info = _ensure_shared_torch_env(
            repo_root=repo_root,
            shared_env=shared_torch_env,
            base_python=base_python,
            torch_cuda_index_url=torch_cuda_index_url,
            torch_version=torch_version,
            require_gpu_torch=require_gpu_torch,
            pip_log=pip_log,
        )
        _link_shared_site_packages(
            task_python=venv_python,
            shared_python=_find_venv_python(shared_torch_env),
            name="myevoskill_shared_torch.pth",
        )

    if requirements_path.exists():
        install_requirements_path = requirements_path
        skip_names: set[str] = set()
        if has_torch:
            skip_names.update(TORCH_REQUIREMENT_NAMES)
        if skip_notebook_packages:
            skip_names.update(NOTEBOOK_REQUIREMENT_NAMES)
        if skip_names:
            filtered_requirements_path = setup_dir / f"{task_id}.requirements.filtered.txt"
            filtered, skipped_requirements = _filter_requirements(requirements_lines, skip_names)
            filtered_requirements_path.write_text(
                "\n".join(filtered) + ("\n" if filtered else ""),
                encoding="utf-8",
            )
            install_requirements_path = filtered_requirements_path
        try:
            with pip_log.open("ab") as log:
                if install_requirements_path.stat().st_size > 0:
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
                            str(install_requirements_path),
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
                    "filtered_requirements_path": str(filtered_requirements_path.resolve()) if filtered_requirements_path else None,
                    "skipped_requirements": skipped_requirements,
                    "shared_torch_env": str(shared_torch_env.resolve()) if has_torch and shared_torch_env else None,
                    "torch_info": torch_info,
                    "created_at_unix": int(time.time()),
                    "pip_log": str(pip_log.resolve()),
                },
            )
            raise TaskEnvSetupError(f"pip install failed; see {pip_log}") from exc

    resolved_py = _resolved_python(venv_python)
    if has_torch:
        torch_info = _verify_torch(
            resolved_py,
            require_gpu=require_gpu_torch,
            label="task venv torch import",
        )
    if has_cupy:
        cupy_info = _verify_cupy(resolved_py)
    state = {
        "task_id": task_id,
        "ready": True,
        "python_executable": str(resolved_py),
        "requirements_path": str(requirements_path.resolve()) if requirements_path.exists() else None,
        "requirements_sha256": req_hash,
        "filtered_requirements_path": str(filtered_requirements_path.resolve()) if filtered_requirements_path else None,
        "skipped_requirements": skipped_requirements,
        "shared_torch_env": str(shared_torch_env.resolve()) if has_torch and shared_torch_env else None,
        "torch_info": torch_info,
        "cupy_info": cupy_info,
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
        filtered_requirements_path=filtered_requirements_path.resolve() if filtered_requirements_path else None,
        skipped_requirements=skipped_requirements,
        shared_torch_env=shared_torch_env.resolve() if has_torch and shared_torch_env else None,
        torch_info=torch_info,
        cupy_info=cupy_info,
    )


def setup_shared_torch_env(
    *,
    repo_root: Path,
    force: bool = False,
    base_python: Path | None = None,
    shared_torch_env: Path | None = None,
    torch_cuda_index_url: str = DEFAULT_TORCH_CUDA_INDEX_URL,
    torch_version: str = DEFAULT_TORCH_VERSION,
    require_gpu_torch: bool = True,
) -> dict[str, Any]:
    """Create/update the shared CUDA Torch venv and verify GPU visibility."""

    repo_root = Path(repo_root).resolve()
    base_python = Path(base_python).resolve() if base_python else Path(sys.executable).resolve()
    _ensure_supported_python(base_python)
    shared_env = Path(shared_torch_env) if shared_torch_env else repo_root / DEFAULT_SHARED_TORCH_ENV_REL
    if not shared_env.is_absolute():
        shared_env = repo_root / shared_env
    shared_env = shared_env.resolve()
    setup_dir = repo_root / "runtime_logs" / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    pip_log = setup_dir / "_shared_torch.pip.log"
    if force and shared_env.exists():
        shutil.rmtree(shared_env)
    return _ensure_shared_torch_env(
        repo_root=repo_root,
        shared_env=shared_env,
        base_python=base_python,
        torch_cuda_index_url=torch_cuda_index_url,
        torch_version=torch_version,
        require_gpu_torch=require_gpu_torch,
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


def _site_packages(python: Path) -> Path:
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(completed.stdout.strip()).resolve()


def _ensure_shared_torch_env(
    *,
    repo_root: Path,
    shared_env: Path,
    base_python: Path,
    torch_cuda_index_url: str,
    torch_version: str,
    require_gpu_torch: bool,
    pip_log: Path,
) -> dict[str, Any]:
    if not shared_env.exists():
        shared_env.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(base_python), "-m", "venv", str(shared_env)],
            check=True,
            cwd=str(repo_root),
        )
    shared_python = _find_venv_python(shared_env)
    install_cmd = [
        str(shared_python),
        "-m",
        "pip",
        "install",
        "--retries",
        "10",
        "--timeout",
        "120",
        f"torch=={torch_version}",
        f"torchvision=={DEFAULT_TORCHVISION_VERSION}",
        f"torchaudio=={DEFAULT_TORCHAUDIO_VERSION}",
        "--index-url",
        torch_cuda_index_url,
    ]
    try:
        with pip_log.open("ab") as log:
            subprocess.run(
                install_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=str(repo_root),
            )
    except subprocess.CalledProcessError as exc:
        raise TaskEnvSetupError(f"shared torch install failed; see {pip_log}") from exc
    info = _verify_torch(
        _resolved_python(shared_python),
        require_gpu=require_gpu_torch,
        label="shared torch env",
    )
    info.update(
        {
            "shared_torch_env": str(shared_env),
            "python_executable": str(_resolved_python(shared_python)),
            "site_packages": str(_site_packages(shared_python)),
            "torch_cuda_index_url": torch_cuda_index_url,
        }
    )
    return info


def _link_shared_site_packages(*, task_python: Path, shared_python: Path, name: str) -> None:
    task_site = _site_packages(task_python)
    shared_site = _site_packages(shared_python)
    task_site.mkdir(parents=True, exist_ok=True)
    pth = task_site / name
    pth.write_text(str(shared_site) + "\n", encoding="utf-8")


def _verify_torch(python: Path, *, require_gpu: bool, label: str) -> dict[str, Any]:
    code = (
        "import json, torch\n"
        "payload = {\n"
        "  'torch_version': torch.__version__,\n"
        "  'torch_cuda': torch.version.cuda,\n"
        "  'cuda_available': bool(torch.cuda.is_available()),\n"
        "  'device_count': int(torch.cuda.device_count()),\n"
        "  'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
        "}\n"
        "print(json.dumps(payload))\n"
        f"raise SystemExit(0 if (payload['cuda_available'] or {not require_gpu!r}) else 3)\n"
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise TaskEnvSetupError(
            f"{label} failed GPU torch verification: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout.strip())


def _verify_cupy(python: Path) -> dict[str, Any]:
    code = (
        "import json, cupy as cp\n"
        "payload = {'device_count': int(cp.cuda.runtime.getDeviceCount())}\n"
        "print(json.dumps(payload))\n"
        "raise SystemExit(0 if payload['device_count'] > 0 else 3)\n"
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise TaskEnvSetupError(
            f"cupy GPU verification failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout.strip())


def _requirement_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith(("-", "http:", "https:", "git+")):
        return None
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", ";", " "):
        if sep in stripped:
            stripped = stripped.split(sep, 1)[0]
    name = stripped.strip().lower().replace("_", "-")
    return name or None


def _has_requirement(lines: list[str], names: set[str]) -> bool:
    normal = {name.lower().replace("_", "-") for name in names}
    return any((_requirement_name(line) or "") in normal for line in lines)


def _has_requirement_prefix(lines: list[str], prefixes: tuple[str, ...]) -> bool:
    return any(
        any((name := (_requirement_name(line) or "")).startswith(prefix) for prefix in prefixes)
        for line in lines
    )


def _filter_requirements(lines: list[str], names: set[str]) -> tuple[list[str], list[str]]:
    normal = {name.lower().replace("_", "-") for name in names}
    kept: list[str] = []
    skipped: list[str] = []
    for line in lines:
        if (_requirement_name(line) or "") in normal:
            skipped.append(line)
        else:
            kept.append(line)
    return kept, skipped


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
