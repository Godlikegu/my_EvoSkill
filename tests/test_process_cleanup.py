from __future__ import annotations

import asyncio
from pathlib import Path

from claude_agent_sdk import TaskNotificationMessage, TaskStartedMessage

import myevoskill.harness.process_reaper as process_reaper
import myevoskill.harness.runner as runner
from myevoskill.harness.process_reaper import ProcessInfo, ReapResult
from myevoskill.harness.runner import (
    ActiveTaskRegistry,
    _cleanup_agent_processes,
    _record_message,
)
from myevoskill.harness.trajectory import TrajectoryWriter


def test_task_started_and_notification_update_registry(tmp_path: Path) -> None:
    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    registry = ActiveTaskRegistry()

    started = TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id="task-1",
        description="python work/main.py",
        uuid="u1",
        session_id="default",
        tool_use_id="tool-1",
        task_type="local_bash",
    )
    _record_message(traj, 1, started, task_registry=registry)
    assert [t.task_id for t in registry.active()] == ["task-1"]

    stopped = TaskNotificationMessage(
        subtype="task_notification",
        data={},
        task_id="task-1",
        status="stopped",
        output_file="",
        summary="stopped",
        uuid="u2",
        session_id="default",
        tool_use_id="tool-1",
    )
    _record_message(traj, 1, stopped, task_registry=registry)
    assert registry.active() == []

    text = traj.path.read_text(encoding="utf-8")
    assert "task_started" in text
    assert "task_stopped" in text


def test_process_reaper_leaf_first_and_scoped(monkeypatch) -> None:
    table = [
        ProcessInfo(100, 1, "claude.exe", "claude run-x"),
        ProcessInfo(101, 100, "bash.exe", "bash run-abc"),
        ProcessInfo(102, 101, "python.exe", "python work/main.py"),
        ProcessInfo(103, 100, "python.exe", "python unrelated.py"),
        ProcessInfo(104, 1, "python.exe", "python work/main.py run-abc"),
    ]
    remaining = {p.pid: p for p in table}
    killed: list[int] = []

    def fake_list_processes():
        return list(remaining.values())

    def fake_kill(pid: int, sig: int) -> None:
        killed.append(pid)
        remaining.pop(pid, None)

    monkeypatch.setattr(process_reaper, "list_processes", fake_list_processes)
    monkeypatch.setattr(process_reaper.os, "kill", fake_kill)
    monkeypatch.setattr(process_reaper.time, "sleep", lambda _: None)

    result = process_reaper.reap_descendant_processes(
        root_pid=100,
        markers=["run-abc"],
        kill=True,
    )

    assert result.ok
    assert killed == [102, 101]
    assert 103 in remaining
    assert 104 in remaining


class _FakeClient:
    def __init__(self, messages=None) -> None:
        self.stopped: list[str] = []
        self._messages = list(messages or [])

    async def stop_task(self, task_id: str) -> None:
        self.stopped.append(task_id)

    async def receive_messages(self):
        while self._messages:
            yield self._messages.pop(0)


def test_cleanup_barrier_stops_active_task_before_judge(tmp_path: Path, monkeypatch) -> None:
    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    registry = ActiveTaskRegistry()
    registry.start(
        TaskStartedMessage(
            subtype="task_started",
            data={},
            task_id="task-1",
            description="python work/main.py",
            uuid="u1",
            session_id="default",
            tool_use_id="tool-1",
            task_type="local_bash",
        )
    )
    notification = TaskNotificationMessage(
        subtype="task_notification",
        data={},
        task_id="task-1",
        status="stopped",
        output_file="",
        summary="stopped",
        uuid="u2",
        session_id="default",
        tool_use_id="tool-1",
    )
    client = _FakeClient(messages=[notification])

    monkeypatch.setattr(runner, "_reap_run_processes", lambda **_: ReapResult())

    result = asyncio.run(
        _cleanup_agent_processes(
            client=client,
            task_registry=registry,
            trajectory=traj,
            round_index=1,
            claude_pid=100,
            run_id="run-1",
            workspace_root=tmp_path / "ws",
            sandbox_root=tmp_path / "sandbox",
            phase="before_judge",
            require_clean=True,
        )
    )

    assert result["ok"] is True
    assert client.stopped == ["task-1"]
    assert registry.active() == []


def test_cleanup_failure_reports_error(tmp_path: Path, monkeypatch) -> None:
    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    registry = ActiveTaskRegistry()
    client = _FakeClient()
    failed = ReapResult(
        remaining=[ProcessInfo(102, 101, "python.exe", "python work/main.py run-1")]
    )
    monkeypatch.setattr(runner, "_reap_run_processes", lambda **_: failed)

    result = asyncio.run(
        _cleanup_agent_processes(
            client=client,
            task_registry=registry,
            trajectory=traj,
            round_index=1,
            claude_pid=100,
            run_id="run-1",
            workspace_root=tmp_path / "ws",
            sandbox_root=tmp_path / "sandbox",
            phase="before_judge",
            require_clean=True,
        )
    )

    assert result["ok"] is False
    assert "process_cleanup_failed" in traj.path.read_text(encoding="utf-8")
