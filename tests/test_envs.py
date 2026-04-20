import json
import subprocess
import sys
from pathlib import Path

import pytest

from myevoskill.envs import EnvManager, EnvSpec, build_task_env_spec


def _patch_env_build(monkeypatch):
    def _fake_create_venv(self, venv_dir, *, python_executable):
        python_executable = self._venv_python_executable(Path(venv_dir))
        python_executable.parent.mkdir(parents=True, exist_ok=True)
        python_executable.write_text("", encoding="utf-8")

    def _fake_run_logged_command(self, command, *, log_path, cwd):
        stdout = "demo-package==1.0\n" if list(command)[-1:] == ["freeze"] else ""
        return subprocess.CompletedProcess(list(command), 0, stdout, "")

    monkeypatch.setattr(EnvManager, "_create_venv", _fake_create_venv)
    monkeypatch.setattr(EnvManager, "_run_logged_command", _fake_run_logged_command)


def test_env_manager_reuses_same_hash(tmp_path, monkeypatch):
    _patch_env_build(monkeypatch)
    manager = EnvManager(tmp_path / "env_cache")
    spec = EnvSpec(
        python_version="3.9",
        requirements=["numpy==1.26.0", "pytest==7.1.1"],
        system_packages=["git"],
    )
    first = manager.ensure_env(spec)
    second = manager.ensure_env(spec)
    assert first.env_hash == second.env_hash
    assert first.env_dir == second.env_dir
    assert first.ready is True
    assert first.python_executable.exists()
    assert first.install_report_path.exists()
    assert (first.env_dir / "pip_freeze.txt").exists()
    assert (first.env_dir / "env_spec.json").exists()


def test_env_manager_resets_only_work_and_output(tmp_path):
    manager = EnvManager(tmp_path / "env_cache")
    run_root = tmp_path / "run-1"
    (run_root / "work").mkdir(parents=True)
    (run_root / "output").mkdir(parents=True)
    (run_root / "keep").mkdir(parents=True)
    (run_root / "work" / "temp.txt").write_text("x", encoding="utf-8")
    (run_root / "output" / "result.txt").write_text("y", encoding="utf-8")
    (run_root / "keep" / "saved.txt").write_text("z", encoding="utf-8")

    manager.reset_run_workspace(run_root)

    assert list((run_root / "work").iterdir()) == []
    assert list((run_root / "output").iterdir()) == []
    assert (run_root / "keep" / "saved.txt").exists()


def test_checkpoint_restore_does_not_rebuild_env(tmp_path, monkeypatch):
    _patch_env_build(monkeypatch)
    manager = EnvManager(tmp_path / "env_cache")
    spec = EnvSpec(
        python_version="3.9",
        requirements=["torch==2.0.0"],
        compute_profile="mixed",
        cuda="12.1",
    )
    cache_record = manager.ensure_env(spec)
    manager.stage_checkpoint(cache_record, "epoch_1.ckpt", "checkpoint-data")
    run_root = tmp_path / "run-2"
    manager.reset_run_workspace(run_root)
    restored = manager.restore_checkpoint(cache_record, run_root, "epoch_1.ckpt")
    assert restored.read_text(encoding="utf-8") == "checkpoint-data"
    assert cache_record.env_dir.exists()


def test_env_manager_writes_install_report_on_build_failure(tmp_path, monkeypatch):
    manager = EnvManager(tmp_path / "env_cache")

    def _fake_create_venv(self, venv_dir, *, python_executable):
        python_executable = self._venv_python_executable(Path(venv_dir))
        python_executable.parent.mkdir(parents=True, exist_ok=True)
        python_executable.write_text("", encoding="utf-8")

    def _failing_run_logged_command(self, command, *, log_path, cwd):
        raise RuntimeError(f"command failed with returncode=1: {' '.join(command)}")

    monkeypatch.setattr(EnvManager, "_create_venv", _fake_create_venv)
    monkeypatch.setattr(EnvManager, "_run_logged_command", _failing_run_logged_command)

    spec = EnvSpec(
        python_version="3.11",
        requirements=["numpy==1.26.0"],
        task_id="demo-task",
        task_family="optics",
    )

    with pytest.raises(RuntimeError, match="task runtime environment build failed"):
        manager.ensure_env(spec)

    env_hash = manager.compute_env_hash(spec)
    install_report_path = manager.task_env_cache_root / env_hash / "install_report.json"
    payload = json.loads(install_report_path.read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["task_id"] == "demo-task"
    assert "build.log" in payload["build_log_path"]
    assert "pip install" in " ".join(payload["commands_run"])


def test_dev_environment_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "environment.yml").exists()
    assert (root / "scripts" / "create_dev_env.sh").exists()
    assert (root / "scripts" / "print_env_info.py").exists()


def test_build_task_env_spec_uses_explicit_python_executable(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==1.26.0\n", encoding="utf-8")

    spec = build_task_env_spec(
        task_id="demo-task",
        family="demo",
        requirements_path=requirements,
        python_executable=Path(sys.executable),
    )

    assert spec.python_executable == str(Path(sys.executable).resolve())
    assert spec.python_version == f"{sys.version_info.major}.{sys.version_info.minor}"


