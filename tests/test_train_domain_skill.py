from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_domain_skill.py"
    spec = importlib.util.spec_from_file_location("train_domain_skill_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_split(repo: Path, train: list[str], valid: list[str]) -> Path:
    split = repo / "registry" / "splits" / "wave.json"
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps({"model_slug": "Model", "train": train, "valid": valid}),
        encoding="utf-8",
    )
    return split


def _write_summary(repo: Path, task_id: str, verdict: str) -> None:
    run = repo / "artifacts" / "logs" / "Model" / task_id / "run-1"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run_summary.json").write_text(
        json.dumps({"run_id": "run-1", "verdict": verdict}),
        encoding="utf-8",
    )


def test_epoch_one_runs_full_train_before_distill(tmp_path, monkeypatch) -> None:
    mod = _load_script_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    split = _write_split(repo, ["T1", "T2"], ["V1"])
    _write_summary(repo, "T1", "PASS")
    _write_summary(repo, "T2", "PASS")
    events: list[tuple[str, str | None]] = []

    def fake_run_task(**kwargs):
        events.append(("run", kwargs["task_id"]))
        return 0

    def fake_distill(**kwargs):
        events.append(("distill", None))
        return 0

    monkeypatch.setattr(mod, "_invoke_run_task", fake_run_task)
    monkeypatch.setattr(mod, "_invoke_distill_skill", fake_distill)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_domain_skill.py",
            "--repo-root",
            str(repo),
            "--split",
            str(split),
            "--skill-id",
            "wave_optics_recon_v1",
            "--max-epochs",
            "1",
            "--skip-validate",
        ],
    )

    assert mod.main() == 0
    assert events == [
        ("run", "T1"),
        ("run", "T2"),
        ("distill", None),
        ("distill", None),
    ]


def test_validation_skipped_until_all_train_pass(tmp_path, monkeypatch) -> None:
    mod = _load_script_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    split = _write_split(repo, ["T1", "T2"], ["V1"])
    validate_calls: list[str | None] = []

    def fake_run_task(**kwargs):
        _write_summary(repo, kwargs["task_id"], "FAIL")
        return 1

    monkeypatch.setattr(mod, "_invoke_run_task", fake_run_task)
    monkeypatch.setattr(mod, "_invoke_distill_skill", lambda **kwargs: 0)
    monkeypatch.setattr(
        mod,
        "_invoke_validate_skill",
        lambda **kwargs: validate_calls.append(kwargs.get("valid_task_id")) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_domain_skill.py",
            "--repo-root",
            str(repo),
            "--split",
            str(split),
            "--skill-id",
            "wave_optics_recon_v1",
            "--max-epochs",
            "1",
        ],
    )

    assert mod.main() == 1
    assert validate_calls == []


def test_validation_uses_one_valid_task_after_train_pass(tmp_path, monkeypatch) -> None:
    mod = _load_script_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    split = _write_split(repo, ["T1"], ["V1", "V2"])
    _write_summary(repo, "T1", "PASS")
    validate_calls: list[str | None] = []

    monkeypatch.setattr(mod, "_invoke_run_task", lambda **kwargs: 0)
    monkeypatch.setattr(mod, "_invoke_distill_skill", lambda **kwargs: 0)
    monkeypatch.setattr(
        mod,
        "_invoke_validate_skill",
        lambda **kwargs: validate_calls.append(kwargs.get("valid_task_id")) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_domain_skill.py",
            "--repo-root",
            str(repo),
            "--split",
            str(split),
            "--skill-id",
            "wave_optics_recon_v1",
            "--max-epochs",
            "1",
            "--valid-task-id",
            "V2",
        ],
    )

    assert mod.main() == 0
    assert validate_calls == ["V2"]


def test_require_all_train_pass_reruns_only_unpassed_on_first_epoch(tmp_path, monkeypatch) -> None:
    mod = _load_script_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    split = _write_split(repo, ["T1", "T2"], ["V1"])
    _write_summary(repo, "T1", "PASS")
    _write_summary(repo, "T2", "FAIL")
    skill_dir = repo / "artifacts" / "skills" / "wave-optics-recon-v1"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: wave-optics-recon-v1\ndescription: test\n---\n",
        encoding="utf-8",
    )
    events: list[tuple[str, str | None]] = []
    injected: list[Path | None] = []

    def fake_run_task(**kwargs):
        events.append(("run", kwargs["task_id"]))
        injected.append(kwargs.get("skill_pack_dir"))
        _write_summary(repo, kwargs["task_id"], "FAIL")
        return 1

    def fake_distill(**kwargs):
        events.append(("distill", None))
        return 0

    monkeypatch.setattr(mod, "_invoke_run_task", fake_run_task)
    monkeypatch.setattr(mod, "_invoke_distill_skill", fake_distill)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_domain_skill.py",
            "--repo-root",
            str(repo),
            "--split",
            str(split),
            "--skill-id",
            "wave_optics_recon_v1",
            "--max-epochs",
            "1",
            "--skip-validate",
            "--require-all-train-pass",
        ],
    )

    assert mod.main() == 1
    assert events == [("run", "T2"), ("distill", None)]
    assert injected == [skill_dir.resolve()]


def test_all_train_pass_at_start_still_final_distills(tmp_path, monkeypatch) -> None:
    mod = _load_script_module()
    repo = tmp_path / "repo"
    repo.mkdir()
    split = _write_split(repo, ["T1", "T2"], ["V1"])
    _write_summary(repo, "T1", "PASS")
    _write_summary(repo, "T2", "PASS")
    skill_dir = repo / "artifacts" / "skills" / "wave-optics-recon-v1"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: wave-optics-recon-v1\ndescription: test\n---\n",
        encoding="utf-8",
    )
    events: list[tuple[str, str | None]] = []
    validate_calls: list[str | None] = []

    monkeypatch.setattr(mod, "_invoke_run_task", lambda **kwargs: events.append(("run", kwargs["task_id"])) or 0)
    monkeypatch.setattr(mod, "_invoke_distill_skill", lambda **kwargs: events.append(("distill", None)) or 0)
    monkeypatch.setattr(
        mod,
        "_invoke_validate_skill",
        lambda **kwargs: validate_calls.append(kwargs.get("valid_task_id")) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_domain_skill.py",
            "--repo-root",
            str(repo),
            "--split",
            str(split),
            "--skill-id",
            "wave_optics_recon_v1",
            "--max-epochs",
            "1",
            "--require-all-train-pass",
        ],
    )

    assert mod.main() == 0
    assert events == [("distill", None)]
    assert validate_calls == ["V1"]
