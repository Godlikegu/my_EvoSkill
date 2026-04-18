from pathlib import Path

from myevoskill.envs import EnvManager, EnvSpec


def test_env_manager_reuses_same_hash(tmp_path):
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


def test_checkpoint_restore_does_not_rebuild_env(tmp_path):
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


def test_dev_environment_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "environment.yml").exists()
    assert (root / "scripts" / "create_dev_env.sh").exists()
    assert (root / "scripts" / "print_env_info.py").exists()


