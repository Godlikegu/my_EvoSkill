from __future__ import annotations

import myevoskill.task_live as task_live_mod
import myevoskill.task_register as task_register_mod
import myevoskill.task_registration_draft as task_registration_draft_mod


def test_task_registration_draft_module_forwards_to_contract_draft_main(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_contract_draft_main(argv=None):
        captured["argv"] = list(argv or [])
        return 17

    monkeypatch.setattr(
        task_registration_draft_mod,
        "contract_draft_main",
        _fake_contract_draft_main,
    )

    exit_code = task_registration_draft_mod.main(["--task-root", "demo-task"])

    assert exit_code == 17
    assert captured["argv"] == ["--task-root", "demo-task"]


def test_task_register_module_forwards_to_register_main(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_register_main(argv=None):
        captured["argv"] = list(argv or [])
        return 23

    monkeypatch.setattr(
        task_register_mod,
        "register_main",
        _fake_register_main,
    )

    exit_code = task_register_mod.main(["--task-root", "demo-task"])

    assert exit_code == 23
    assert captured["argv"] == ["--task-root", "demo-task"]


def test_task_live_module_forwards_to_live_runner_main(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_live_main(argv=None):
        captured["argv"] = list(argv or [])
        return 31

    monkeypatch.setattr(
        task_live_mod,
        "live_main",
        _fake_live_main,
    )

    exit_code = task_live_mod.main(["--task-id", "demo-task"])

    assert exit_code == 31
    assert captured["argv"] == ["--task-id", "demo-task"]
