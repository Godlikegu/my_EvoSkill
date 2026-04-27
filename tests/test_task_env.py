from __future__ import annotations

import json
from pathlib import Path

from myevoskill.registration import register_task
import myevoskill.task_env as task_env_mod
from myevoskill.task_env import setup_task_env


def _write_minimal_task(tasks_root: Path, task_id: str) -> None:
    task_root = tasks_root / task_id
    (task_root / "data").mkdir(parents=True)
    (task_root / "evaluation").mkdir()
    (task_root / "README.md").write_text("# demo\n", encoding="utf-8")
    (task_root / "data" / "raw_data.npz").write_bytes(b"not a real npz")
    (task_root / "data" / "meta_data.json").write_text("{}", encoding="utf-8")
    (task_root / "evaluation" / "judge_adapter.py").write_text("# judge\n", encoding="utf-8")
    contract = {
        "version": 2,
        "task_id": task_id,
        "family": "demo",
        "files": [
            {"id": "readme", "path": "README.md", "visibility": "public", "role": "task_description"},
            {"id": "raw_data", "path": "data/raw_data.npz", "visibility": "public", "role": "input_data"},
            {"id": "meta_data", "path": "data/meta_data.json", "visibility": "public", "role": "metadata"},
        ],
        "execution": {
            "readable_files": ["readme", "raw_data", "meta_data"],
            "entrypoint": "work/main.py",
            "writable_paths": ["work/", "output/", "checkpoints/"],
        },
        "output": {
            "path": "output/reconstruction.npz",
            "format": "npz",
            "fields": [{"name": "x", "dtype": "float64", "shape": [1]}],
        },
        "metrics": [],
    }
    (task_root / "evaluation" / "task_contract.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )


def test_setup_task_env_writes_ready_state_and_registration_uses_it(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    _write_minimal_task(tasks_root, "demo")

    result = setup_task_env(repo_root=repo_root, task_id="demo", tasks_root=tasks_root)

    assert result.ready is True
    assert result.state_path.exists()
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert state["ready"] is True
    assert Path(state["python_executable"]).exists()
    assert state["requirements_path"] is None

    reg = register_task(
        repo_root=repo_root,
        task_id="demo",
        tasks_root=tasks_root,
        force=True,
        require_task_env=True,
    )
    assert reg.manifest["runtime_env"]["backend"] == "per_task_venv"
    assert reg.manifest["runtime_env"]["python_executable"] == state["python_executable"]


def test_setup_task_env_reuses_shared_torch_and_filters_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    tasks_root = tmp_path / "tasks"
    repo_root.mkdir()
    tasks_root.mkdir()
    _write_minimal_task(tasks_root, "torch_demo")
    (tasks_root / "torch_demo" / "requirements.txt").write_text(
        "torch>=2.0\n"
        "torchvision>=0.15\n"
        "torchaudio\n",
        encoding="utf-8",
    )

    linked: list[tuple[Path, Path]] = []

    def fake_ensure_shared_torch_env(**kwargs):
        shared = kwargs["shared_env"]
        (shared / "Scripts").mkdir(parents=True)
        (shared / "Scripts" / "python.exe").write_text("", encoding="utf-8")
        return {
            "torch_version": "2.5.1+cu118",
            "torch_cuda": "11.8",
            "cuda_available": True,
            "device_name": "Fake GPU",
        }

    def fake_verify_torch(python: Path, *, require_gpu: bool, label: str):
        return {
            "torch_version": "2.5.1+cu118",
            "torch_cuda": "11.8",
            "cuda_available": True,
            "device_name": "Fake GPU",
        }

    def fake_link_shared_site_packages(*, task_python: Path, shared_python: Path, name: str):
        linked.append((task_python, shared_python))

    monkeypatch.setattr(task_env_mod, "_ensure_shared_torch_env", fake_ensure_shared_torch_env)
    monkeypatch.setattr(task_env_mod, "_verify_torch", fake_verify_torch)
    monkeypatch.setattr(task_env_mod, "_link_shared_site_packages", fake_link_shared_site_packages)

    result = setup_task_env(
        repo_root=repo_root,
        task_id="torch_demo",
        tasks_root=tasks_root,
        require_gpu_torch=True,
    )

    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert result.shared_torch_env == (repo_root / ".venvs" / "_torch-cu118-py310").resolve()
    assert linked
    assert state["shared_torch_env"] == str(result.shared_torch_env)
    assert state["torch_info"]["cuda_available"] is True
    filtered = Path(state["filtered_requirements_path"]).read_text(encoding="utf-8")
    assert "torch" not in filtered.lower()

    reg = register_task(
        repo_root=repo_root,
        task_id="torch_demo",
        tasks_root=tasks_root,
        force=True,
        require_task_env=True,
    )
    runtime_env = reg.manifest["runtime_env"]
    assert runtime_env["shared_torch_env"] == str(result.shared_torch_env)
    assert runtime_env["torch_info"]["device_name"] == "Fake GPU"
