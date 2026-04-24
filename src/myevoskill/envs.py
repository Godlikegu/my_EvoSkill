"""Environment hashing, build, and cache management."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .models import EnvCacheRecord

DEFAULT_ENV_BUILD_TIMEOUT_SECONDS = 30 * 60


@dataclass(frozen=True)
class EnvSpec:
    """Stable environment specification used for cache reuse."""

    python_version: str
    requirements: List[str]
    python_executable: str = ""
    backend: str = "venv_pip"
    task_id: str = ""
    task_family: str = ""
    system_packages: List[str] = field(default_factory=list)
    compute_profile: str = "mixed"
    cuda: str = ""
    container_image: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


def normalize_requirements_lines(lines: Sequence[str] | str) -> list[str]:
    """Normalize requirements.txt contents into stable cache inputs."""

    if isinstance(lines, str):
        raw_lines = lines.splitlines()
    else:
        raw_lines = [str(item) for item in lines]
    normalized: list[str] = []
    for raw_line in raw_lines:
        line = str(raw_line).strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
        if line:
            normalized.append(line)
    return normalized


def load_requirements_lines(requirements_path: Path) -> list[str]:
    """Load and normalize one task requirements.txt file."""

    path = Path(requirements_path)
    if not path.exists():
        raise FileNotFoundError(f"requirements.txt not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"requirements.txt is not a file: {path}")
    return normalize_requirements_lines(path.read_text(encoding="utf-8"))


def _python_major_minor(python_executable: Path | str) -> str:
    executable = Path(python_executable)
    completed = subprocess.run(
        [str(executable), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to query python version for interpreter {executable}: "
            f"{completed.stderr.strip() or '<empty>'}"
        )
    return (completed.stdout or "").strip()


def resolve_python_executable(
    *,
    project_root: Path | None = None,
    python_executable: Path | str | None = None,
) -> Path:
    """Resolve the preferred builder interpreter for task environments."""

    candidates: list[Path] = []
    if python_executable is not None:
        candidates.append(Path(python_executable))

    env_override = str(os.environ.get("MYEVOSKILL_ENV_PYTHON", "") or "").strip()
    if env_override:
        candidates.append(Path(env_override))

    if project_root is not None:
        root = Path(project_root).resolve()
        if os.name == "nt":
            candidates.extend(
                [
                    root / ".conda_env" / "python.exe",
                    root / ".conda_env" / "Scripts" / "python.exe",
                    root / ".venv" / "Scripts" / "python.exe",
                    root / "venv" / "Scripts" / "python.exe",
                ]
            )
        else:
            candidates.extend(
                [
                    root / ".conda_env" / "bin" / "python",
                    root / ".venv" / "bin" / "python",
                    root / "venv" / "bin" / "python",
                ]
            )

    candidates.append(Path(sys.executable))
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.normpath(str(candidate)))
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def build_task_env_spec(
    *,
    task_id: str,
    family: str,
    requirements_path: Path,
    project_root: Path | None = None,
    python_version: str | None = None,
    python_executable: Path | str | None = None,
) -> EnvSpec:
    """Construct one task runtime environment specification."""

    resolved_python_executable = resolve_python_executable(
        project_root=project_root,
        python_executable=python_executable,
    )
    normalized_python_version = (
        str(python_version).strip()
        if python_version is not None
        else _python_major_minor(resolved_python_executable)
    )
    if not normalized_python_version:
        normalized_python_version = _python_major_minor(resolved_python_executable)
    return EnvSpec(
        python_version=normalized_python_version,
        requirements=load_requirements_lines(requirements_path),
        python_executable=str(resolved_python_executable),
        task_id=str(task_id or "").strip(),
        task_family=str(family or "").strip(),
    )


def resolve_env_cache_root(project_root: Path) -> Path:
    """Resolve the shared task environment cache root for one project."""

    return Path(project_root).resolve() / "artifacts" / "env_cache"


def python_executable_path_entries(python_executable: Path | str) -> list[str]:
    """Return PATH entries that make one Python environment the default launcher."""

    executable = Path(python_executable).resolve()
    parent = executable.parent
    env_root = parent.parent if parent.name.lower() in {"scripts", "bin"} else parent
    candidates: list[Path] = [executable.parent, env_root]
    windows_style_layout = bool(
        executable.suffix.lower() == ".exe"
        or (env_root / "Scripts").exists()
        or (env_root / "Library" / "bin").exists()
        or (env_root / "DLLs").exists()
    )
    if windows_style_layout:
        candidates.extend(
            [
                env_root / "Scripts",
                env_root / "Library" / "bin",
                env_root / "Library" / "usr" / "bin",
                env_root / "DLLs",
            ]
        )
    else:
        candidates.append(env_root / "bin")
    entries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if not candidate.exists():
            continue
        normalized = os.path.normcase(os.path.normpath(candidate_str))
        if normalized in seen:
            continue
        seen.add(normalized)
        entries.append(candidate_str)
    return entries


class EnvManager:
    """Manage reusable task environments and per-run workspace resets."""

    def __init__(
        self,
        cache_root: Path,
        *,
        build_timeout_seconds: int = DEFAULT_ENV_BUILD_TIMEOUT_SECONDS,
    ):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.base_image_cache_root = self.cache_root / "base_images"
        self.task_env_cache_root = self.cache_root / "task_envs"
        self.venv_storage_root = self._default_venv_storage_root()
        self.dataset_cache_root = self.cache_root / "datasets"
        self.artifact_cache_root = self.cache_root / "artifacts"
        self.checkpoint_cache_root = self.cache_root / "checkpoints"
        self.build_timeout_seconds = max(1, int(build_timeout_seconds))
        for path in (
            self.base_image_cache_root,
            self.task_env_cache_root,
            self.venv_storage_root,
            self.dataset_cache_root,
            self.artifact_cache_root,
            self.checkpoint_cache_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def compute_env_hash(self, spec: EnvSpec) -> str:
        payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def ensure_env(self, spec: EnvSpec) -> EnvCacheRecord:
        env_hash = self.compute_env_hash(spec)
        base_dir = self.base_image_cache_root / (
            spec.container_image.replace("/", "_").replace(":", "_") or "default"
        )
        env_dir = self.task_env_cache_root / env_hash
        dataset_dir = self.dataset_cache_root / env_hash
        artifact_dir = self.artifact_cache_root / env_hash
        checkpoint_dir = self.checkpoint_cache_root / env_hash
        for path in (base_dir, env_dir, dataset_dir, artifact_dir, checkpoint_dir):
            path.mkdir(parents=True, exist_ok=True)
        env_spec_path = env_dir / "env_spec.json"
        build_requirements_path = env_dir / "requirements.txt"
        install_report_path = env_dir / "install_report.json"
        build_log_path = env_dir / "build.log"
        freeze_path = env_dir / "pip_freeze.txt"
        venv_dir = env_dir / "venv"
        materialized_venv_dir = self._materialized_venv_dir(env_hash=env_hash, env_dir=env_dir)
        python_executable = self._venv_python_executable(materialized_venv_dir)

        env_spec_path.write_text(
            json.dumps(asdict(spec), indent=2, sort_keys=True), encoding="utf-8"
        )
        build_requirements_path.write_text(
            ("\n".join(spec.requirements) + "\n") if spec.requirements else "",
            encoding="utf-8",
        )

        cached_report = self._load_install_report(install_report_path)
        if (
            cached_report.get("success") is True
            and python_executable.exists()
            and freeze_path.exists()
        ):
            self._ensure_venv_link(venv_dir=venv_dir, materialized_venv_dir=materialized_venv_dir)
            return self._build_cache_record(
                env_hash=env_hash,
                env_dir=env_dir,
                base_dir=base_dir,
                dataset_dir=dataset_dir,
                artifact_dir=artifact_dir,
                checkpoint_dir=checkpoint_dir,
                python_executable=python_executable,
                install_report_path=install_report_path,
                build_log_path=build_log_path,
                ready=True,
            )

        self._remove_link_or_directory(venv_dir)
        if materialized_venv_dir.exists():
            shutil.rmtree(materialized_venv_dir, ignore_errors=True)
        if freeze_path.exists():
            freeze_path.unlink()
        build_log_path.write_text("", encoding="utf-8")

        commands_run = [
            f"{spec.python_executable or sys.executable} -m venv {materialized_venv_dir}",
            f"{python_executable} -m pip install --upgrade pip setuptools wheel",
        ]
        if spec.requirements:
            commands_run.append(f"{python_executable} -m pip install -r {build_requirements_path}")
        commands_run.append(f"{python_executable} -m pip freeze")

        started_at = time.time()
        try:
            self._append_build_log(
                build_log_path,
                f"[create_venv] {spec.python_executable or sys.executable} -m venv {materialized_venv_dir}\n",
            )
            self._create_venv(
                materialized_venv_dir,
                python_executable=spec.python_executable or sys.executable,
            )
            self._ensure_venv_link(venv_dir=venv_dir, materialized_venv_dir=materialized_venv_dir)
            self._run_logged_command(
                [str(python_executable), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                log_path=build_log_path,
                cwd=env_dir,
            )
            if spec.requirements:
                self._run_logged_command(
                    [str(python_executable), "-m", "pip", "install", "-r", str(build_requirements_path)],
                    log_path=build_log_path,
                    cwd=env_dir,
                )
            freeze_completed = self._run_logged_command(
                [str(python_executable), "-m", "pip", "freeze"],
                log_path=build_log_path,
                cwd=env_dir,
            )
            freeze_path.write_text(freeze_completed.stdout or "", encoding="utf-8")
        except Exception as exc:
            report = {
                "success": False,
                "backend": spec.backend,
                "env_hash": env_hash,
                "task_id": spec.task_id,
                "task_family": spec.task_family,
                "python_version": spec.python_version,
                "python_executable": str(python_executable),
                "venv_path": str(venv_dir),
                "venv_storage_path": str(materialized_venv_dir),
                "requirements_path": str(build_requirements_path),
                "build_log_path": str(build_log_path),
                "commands_run": commands_run,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
            install_report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            raise RuntimeError(
                f"task runtime environment build failed for env_hash={env_hash}; "
                f"see {build_log_path}"
            ) from exc

        report = {
            "success": True,
            "backend": spec.backend,
            "env_hash": env_hash,
            "task_id": spec.task_id,
            "task_family": spec.task_family,
            "python_version": spec.python_version,
            "python_executable": str(python_executable),
            "venv_path": str(venv_dir),
            "venv_storage_path": str(materialized_venv_dir),
            "requirements_path": str(build_requirements_path),
            "build_log_path": str(build_log_path),
            "pip_freeze_path": str(freeze_path),
            "commands_run": commands_run,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        install_report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return self._build_cache_record(
            env_hash=env_hash,
            env_dir=env_dir,
            base_dir=base_dir,
            dataset_dir=dataset_dir,
            artifact_dir=artifact_dir,
            checkpoint_dir=checkpoint_dir,
            python_executable=python_executable,
            install_report_path=install_report_path,
            build_log_path=build_log_path,
            ready=True,
        )

    def reset_run_workspace(self, run_root: Path) -> None:
        run_root = Path(run_root)
        for name in ("work", "output"):
            target = run_root / name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

    def stage_checkpoint(
        self, cache_record: EnvCacheRecord, checkpoint_name: str, content: str
    ) -> Path:
        path = cache_record.checkpoint_cache_dir / checkpoint_name
        path.write_text(content, encoding="utf-8")
        return path

    def restore_checkpoint(
        self, cache_record: EnvCacheRecord, run_root: Path, checkpoint_name: str
    ) -> Path:
        source = cache_record.checkpoint_cache_dir / checkpoint_name
        if not source.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_name}")
        target_dir = Path(run_root) / "checkpoints"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / checkpoint_name
        shutil.copy2(source, target)
        return target

    def _build_cache_record(
        self,
        *,
        env_hash: str,
        env_dir: Path,
        base_dir: Path,
        dataset_dir: Path,
        artifact_dir: Path,
        checkpoint_dir: Path,
        python_executable: Path,
        install_report_path: Path,
        build_log_path: Path,
        ready: bool,
    ) -> EnvCacheRecord:
        return EnvCacheRecord(
            env_hash=env_hash,
            env_dir=env_dir,
            python_executable=python_executable,
            install_report_path=install_report_path,
            build_log_path=build_log_path,
            ready=ready,
            base_image_cache_dir=base_dir,
            task_env_cache_dir=env_dir,
            dataset_cache_dir=dataset_dir,
            artifact_cache_dir=artifact_dir,
            checkpoint_cache_dir=checkpoint_dir,
        )

    def _create_venv(self, venv_dir: Path, *, python_executable: Path | str) -> None:
        completed = subprocess.run(
            [str(python_executable), "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
            timeout=self.build_timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"failed to create venv with interpreter {python_executable}: "
                f"{completed.stderr.strip() or '<empty>'}"
            )

    def _default_venv_storage_root(self) -> Path:
        if os.name != "nt":
            return self.cache_root / "v"
        try:
            project_root = self.cache_root.parents[1]
        except IndexError:
            project_root = self.cache_root
        return project_root / ".v"

    def _materialized_venv_dir(self, *, env_hash: str, env_dir: Path) -> Path:
        if os.name != "nt":
            return env_dir / "venv"
        return self.venv_storage_root / env_hash

    def _ensure_venv_link(self, *, venv_dir: Path, materialized_venv_dir: Path) -> None:
        if venv_dir == materialized_venv_dir:
            return
        if venv_dir.exists():
            return
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "cmd",
            "/c",
            "mklink",
            "/J",
            str(venv_dir),
            str(materialized_venv_dir),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to create task env junction: "
                f"link={venv_dir} target={materialized_venv_dir} stderr={completed.stderr.strip() or '<empty>'}"
            )

    def _remove_link_or_directory(self, path: Path) -> None:
        if not path.exists():
            return
        if os.name == "nt":
            completed = subprocess.run(
                ["cmd", "/c", "rmdir", str(path)],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    def _run_logged_command(
        self,
        command: Sequence[str],
        *,
        log_path: Path,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        self._append_build_log(log_path, f"[command] {' '.join(command)}\n")
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=self.build_timeout_seconds,
        )
        if completed.stdout:
            self._append_build_log(log_path, completed.stdout)
        if completed.stderr:
            self._append_build_log(log_path, completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"command failed with returncode={completed.returncode}: {' '.join(command)}"
            )
        return completed

    def _append_build_log(self, log_path: Path, text: str) -> None:
        if not text:
            return
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")

    def _load_install_report(self, report_path: Path) -> dict[str, object]:
        if not report_path.exists():
            return {}
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _venv_python_executable(self, venv_dir: Path) -> Path:
        if os.name == "nt":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"
