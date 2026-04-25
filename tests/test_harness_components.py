"""Unit tests for the new clean harness components.

These tests deliberately do **not** call the Claude API. They cover:
    * ``WorkspacePolicy`` path / substring / bash checks.
    * ``PlanGuard`` freshness logic.
    * ``TrajectoryWriter`` JSONL contract.
    * ``hooks.make_pre_tool_use_hook`` end-to-end deny / allow decisions.
    * ``JudgeRunner._make_feedback`` verdict translation.

The end-to-end harness is exercised separately by ``run_smoke_three.sh``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from myevoskill.harness.hooks import make_pre_tool_use_hook
from myevoskill.harness.plan_guard import PLAN_FILENAME, PlanGuard
from myevoskill.harness.trajectory import TrajectoryWriter
from myevoskill.judge.bridge import FAIL, INVALID, PASS, JudgeRunner
from myevoskill.workspace.policy import WorkspacePolicy


# --------------------------------------------------------------------- policy


def _policy(tmp_path: Path) -> WorkspacePolicy:
    root = tmp_path / "ws"
    (root / "work").mkdir(parents=True)
    (root / "output").mkdir(parents=True)
    return WorkspacePolicy(
        agent_root=root.resolve(),
        primary_output_rel="output/result.npz",
    )


def test_policy_blocks_forbidden_substring(tmp_path: Path) -> None:
    p = _policy(tmp_path)
    assert p.find_forbidden("/some/path/ground_truth/x.npy") is not None
    assert p.find_forbidden("evaluation/judge.py") is not None
    assert p.find_forbidden("workspace/work/output.npy") is None


def test_policy_is_inside_writable(tmp_path: Path) -> None:
    p = _policy(tmp_path)
    inside = p.agent_root / "work" / "main.py"
    outside = tmp_path / "elsewhere" / "x.py"
    assert p.is_inside(inside)
    assert not p.is_inside(outside)
    assert p.is_writable(inside)
    assert not p.is_writable(p.agent_root / "README.md")


@pytest.mark.parametrize(
    "cmd",
    [
        "sudo rm -rf /",
        "curl http://example.com",
        "pip install evil-pkg",
        "cd / && ls",
        "cd C:\\Windows",
        "cat /etc/passwd",
        "cd ../..",
    ],
)
def test_policy_dangerous_bash_blocks(tmp_path: Path, cmd: str) -> None:
    p = _policy(tmp_path)
    assert p.find_dangerous_bash(cmd) is not None, f"should block: {cmd!r}"


@pytest.mark.parametrize(
    "cmd",
    [
        "python work/main.py",
        "ls work",
        "python -m pytest",
        "echo hello",
    ],
)
def test_policy_dangerous_bash_allows(tmp_path: Path, cmd: str) -> None:
    p = _policy(tmp_path)
    assert p.find_dangerous_bash(cmd) is None, f"should allow: {cmd!r}"


# ------------------------------------------------------------------ plan_guard


def test_plan_guard_seeds_plan(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    g = PlanGuard(root)
    assert (root / PLAN_FILENAME).exists()
    assert g.is_plan_fresh()  # nothing edited yet


def test_plan_guard_blocks_stale(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    g = PlanGuard(root)
    # Force plan.md mtime backwards so _last_code_mtime is strictly newer.
    old_t = time.time() - 60
    os.utime(root / "plan.md", (old_t, old_t))
    g.note_code_modification()
    reason = g.should_block("Write", {"file_path": str(root / "work" / "main.py")})

    assert reason is not None
    assert "plan.md" in reason
    # Editing plan.md itself is always allowed.
    assert g.should_block("Write", {"file_path": str(root / "plan.md")}) is None


def test_plan_guard_allows_after_plan_refresh(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    g = PlanGuard(root)
    g.note_code_modification()
    time.sleep(0.01)
    (root / "plan.md").write_text("# Plan\n## Round 2\nrefreshed", encoding="utf-8")
    # Touch the file mtime forward, just in case.
    new_t = time.time() + 1
    os.utime(root / "plan.md", (new_t, new_t))
    assert g.is_plan_fresh()
    assert g.should_block("Write", {"file_path": str(root / "work" / "x.py")}) is None


# ------------------------------------------------------------------ trajectory


def test_trajectory_writes_jsonl(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    tw = TrajectoryWriter(log_root, run_id="r1")
    tw.assistant_text(1, "hello")
    tw.tool_call(1, "Read", {"file_path": "README.md"}, tool_use_id="t1")
    tw.tool_result(1, "t1", "file contents", is_error=False)
    tw.env_feedback(1, "policy_deny", {"reason": "x"})
    tw.round_marker(1, "judged", {"verdict": "FAIL"})

    lines = (log_root / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    kinds = [json.loads(l)["kind"] for l in lines]
    assert kinds == [
        "assistant_text",
        "tool_call",
        "tool_result",
        "env_feedback",
        "round_marker",
    ]
    assert all(json.loads(l)["run_id"] == "r1" for l in lines)


# ------------------------------------------------------------------- pre-hook


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_pre_hook_denies_outside_path(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    plan_guard = PlanGuard(policy.agent_root)
    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    hook = make_pre_tool_use_hook(
        policy=policy,
        plan_guard=plan_guard,
        trajectory=traj,
        round_index_getter=lambda: 1,
    )
    out = asyncio.run(
        hook(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": str(tmp_path / "outside.txt")},
            },
            "use1",
            None,
        )
    )
    decision = out["hookSpecificOutput"]["permissionDecision"]
    assert decision == "deny"
    assert "outside" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_pre_hook_denies_forbidden_substring(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    plan_guard = PlanGuard(policy.agent_root)
    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    hook = make_pre_tool_use_hook(
        policy=policy,
        plan_guard=plan_guard,
        trajectory=traj,
        round_index_getter=lambda: 1,
    )
    # Path inside the workspace but mentions ground_truth.
    target = policy.agent_root / "work" / "ground_truth.npy"
    out = asyncio.run(
        hook(
            {"tool_name": "Read", "tool_input": {"file_path": str(target)}},
            "use2",
            None,
        )
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pre_hook_denies_dangerous_bash(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    plan_guard = PlanGuard(policy.agent_root)
    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    hook = make_pre_tool_use_hook(
        policy=policy,
        plan_guard=plan_guard,
        trajectory=traj,
        round_index_getter=lambda: 1,
    )
    out = asyncio.run(
        hook(
            {"tool_name": "Bash", "tool_input": {"command": "curl http://x"}},
            "use3",
            None,
        )
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pre_hook_allows_inside_read(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    plan_guard = PlanGuard(policy.agent_root)
    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    hook = make_pre_tool_use_hook(
        policy=policy,
        plan_guard=plan_guard,
        trajectory=traj,
        round_index_getter=lambda: 1,
    )
    target = policy.agent_root / "README.md"
    target.write_text("hello", encoding="utf-8")
    out = asyncio.run(
        hook(
            {"tool_name": "Read", "tool_input": {"file_path": str(target)}},
            "use4",
            None,
        )
    )
    # No hookSpecificOutput == allow.
    assert "hookSpecificOutput" not in out


def test_pre_hook_blocks_code_when_plan_stale(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    plan_guard = PlanGuard(policy.agent_root)
    # Force plan.md mtime backwards so _last_code_mtime is strictly newer.
    old_t = time.time() - 60
    os.utime(policy.agent_root / "plan.md", (old_t, old_t))
    plan_guard.note_code_modification()

    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    hook = make_pre_tool_use_hook(
        policy=policy,
        plan_guard=plan_guard,
        trajectory=traj,
        round_index_getter=lambda: 2,
    )
    out = asyncio.run(
        hook(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(policy.agent_root / "work" / "main.py"),
                    "content": "print(1)",
                },
            },
            "use5",
            None,
        )
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "plan.md" in out["hookSpecificOutput"]["permissionDecisionReason"]


# --------------------------------------------------------------- judge bridge


def _make_runner(tmp_path: Path) -> JudgeRunner:
    return JudgeRunner(
        repo_root=tmp_path,
        manifest={"task_id": "fake", "source_task_dir": "fake"},
        log_root=tmp_path / "logs",
    )


def test_judge_feedback_pass(tmp_path: Path) -> None:
    r = _make_runner(tmp_path)
    fb = r._make_feedback(
        {
            "all_metrics_passed": True,
            "metrics_actual": {"ncc": 0.99, "nrmse": 0.05},
            "failed_metrics": [],
            "failure_tags": [],
        }
    )
    assert fb.verdict == PASS
    assert fb.metric_status == {"ncc": True, "nrmse": True}
    assert fb.failure_tags == ()


def test_judge_feedback_fail(tmp_path: Path) -> None:
    r = _make_runner(tmp_path)
    fb = r._make_feedback(
        {
            "all_metrics_passed": False,
            "metrics_actual": {"ncc": 0.5, "nrmse": 0.4},
            "failed_metrics": ["ncc"],
            "failure_tags": ["metric:ncc"],
        }
    )
    assert fb.verdict == FAIL
    assert fb.metric_status == {"ncc": False, "nrmse": True}


def test_judge_feedback_invalid(tmp_path: Path) -> None:
    r = _make_runner(tmp_path)
    fb = r._make_feedback(
        {
            "all_metrics_passed": False,
            "metrics_actual": {},
            "failed_metrics": [],
            "failure_tags": ["missing_output"],
        }
    )
    assert fb.verdict == INVALID
    assert "missing_output" in fb.failure_tags


def test_judge_invalid_helper(tmp_path: Path) -> None:
    r = _make_runner(tmp_path)
    res = r._invalid("missing_judge_adapter", "no adapter")
    assert res.feedback.verdict == INVALID
    assert res.feedback.failure_tags == ("missing_judge_adapter",)
    assert res.success is False
