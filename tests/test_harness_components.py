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
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ToolUseBlock

from myevoskill.harness.hooks import make_pre_tool_use_hook
from myevoskill.harness.plan_guard import PLAN_FILENAME, PlanGuard
from myevoskill.harness.prompts import feedback_user_prompt, initial_user_prompt
from myevoskill.harness.runner import _record_message, agent_runtime_env_overrides
from myevoskill.harness.trajectory import TrajectoryWriter
from myevoskill.judge.bridge import FAIL, INVALID, PASS, JudgeFeedback, JudgeRunner
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


def test_policy_blocks_reference_solution_substrings(tmp_path: Path) -> None:
    """The denylist must cover the entire reference solution: ``src/``,
    ``notebooks/``, ``plan/`` (author plan, distinct from agent ``plan.md``),
    ``main.py`` and ``*.ipynb``. These were previously only enforced at
    registration time; we now enforce them at runtime too in case a stale
    workspace ever contained one.
    """
    p = _policy(tmp_path)
    # Reference Python source.
    assert p.find_forbidden("../tasks/demo/src/visualization.py") is not None
    assert p.find_forbidden("/abs/tasks/demo/src/main.py") is not None
    # Reference notebooks.
    assert p.find_forbidden("tasks/demo/notebooks/walkthrough.ipynb") is not None
    assert p.find_forbidden("anything.ipynb") is not None
    # Author plan/ folder.
    assert p.find_forbidden("tasks/demo/plan/approach.md") is not None
    # Reference entrypoint outside the workspace will be rejected by the
    # ``is_inside`` boundary check, NOT by a substring match -- the agent's
    # own ``work/main.py`` entrypoint must remain writable.
    assert p.find_forbidden("tasks/demo/main.py") is None
    assert p.find_forbidden("work/main.py") is None
    # The agent's own plan.md is NOT a reference notebook -- different path.
    assert p.find_forbidden("plan.md") is None


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
    assert (root / PLAN_FILENAME).read_text(encoding="utf-8") == "# Plan\n\n"
    assert not g.is_plan_fresh(round_index=1)
    reason = g.should_block(
        "Write",
        {"file_path": str(root / "work" / "main.py")},
        round_index=1,
    )
    assert reason is not None
    assert "Round 1" in reason


def test_plan_guard_allows_repeated_code_actions_within_same_round(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "work").mkdir()
    g = PlanGuard(root)
    (root / "plan.md").write_text(
        "# Plan\n\n"
        "## Round 1 - initial solver\n"
        "**Hypothesis:** use the public measurements.\n"
        "**Change:** create work/main.py.\n"
        "**Verification:** output shape check.\n",
        encoding="utf-8",
    )
    assert g.is_plan_fresh(round_index=1)
    assert (
        g.should_block(
            "Write",
            {"file_path": str(root / "work" / "main.py")},
            round_index=1,
        )
        is None
    )
    g.note_code_modification()
    assert (
        g.should_block(
            "Edit",
            {"file_path": str(root / "work" / "main.py")},
            round_index=1,
        )
        is None
    )
    assert g.should_block("Bash", {"command": "python work/main.py"}, round_index=1) is None
    assert (
        g.should_block(
            "Bash",
            {"command": f'cd "{root}" && python -c "print(1)"'},
            round_index=1,
        )
        is None
    )


def test_plan_guard_blocks_chained_python_before_round_one_plan(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    g = PlanGuard(root)
    reason = g.should_block(
        "Bash",
        {"command": f'cd "{root}" && python -c "print(1)"'},
        round_index=1,
    )
    assert reason is not None
    assert "Round 1" in reason


def test_plan_guard_requires_new_heading_after_round_advances(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    g = PlanGuard(root)
    (root / "plan.md").write_text(
        "# Plan\n\n## Round 1 - first\nready\n",
        encoding="utf-8",
    )
    assert g.should_block("Write", {"file_path": str(root / "work" / "main.py")}, round_index=1) is None
    reason = g.should_block(
        "Write",
        {"file_path": str(root / "work" / "main.py")},
        round_index=2,
    )

    assert reason is not None
    assert "Round 2" in reason
    # Editing plan.md itself is always allowed.
    assert g.should_block("Write", {"file_path": str(root / "plan.md")}, round_index=2) is None


def test_plan_guard_allows_round_two_after_round_two_heading(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    g = PlanGuard(root)
    (root / "plan.md").write_text(
        "# Plan\n\n## Round 1\ninitial\n\n## Round 2\nrefreshed\n",
        encoding="utf-8",
    )
    assert g.is_plan_fresh(round_index=2)
    assert (
        g.should_block(
            "Write",
            {"file_path": str(root / "work" / "x.py")},
            round_index=2,
        )
        is None
    )


def test_plan_guard_ignores_indented_round_example(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    g = PlanGuard(root)
    (root / "plan.md").write_text(
        "# Plan\n\n    ## Round 1 - example only\n",
        encoding="utf-8",
    )
    reason = g.should_block(
        "Write",
        {"file_path": str(root / "work" / "x.py")},
        round_index=1,
    )
    assert reason is not None


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


def test_record_message_does_not_duplicate_tool_calls(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    tw = TrajectoryWriter(log_root, run_id="r1")
    msg = AssistantMessage(
        content=[ToolUseBlock(id="tool1", name="Read", input={"file_path": "README.md"})],
        model="test",
    )
    _record_message(tw, 1, msg)
    lines = (log_root / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines == []


def test_clean_trajectory_filters_thinking_and_duplicate_tool_calls(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    tw = TrajectoryWriter(log_root, run_id="r1")
    tw.assistant_thinking(1, "private scratch")
    tw.tool_call(1, "Read", {"file_path": "README.md"}, tool_use_id="tool1")
    tw.tool_call(1, "Read", {"file_path": "README.md"}, tool_use_id="tool1")
    tw.tool_result(1, "tool1", "ok", is_error=False)

    events = tw.read_clean_events()
    assert [e["kind"] for e in events] == ["tool_call", "tool_result"]
    assert len([e for e in events if e["kind"] == "tool_call"]) == 1


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


def test_pre_hook_blocks_code_when_round_plan_missing(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    plan_guard = PlanGuard(policy.agent_root)

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
                    "file_path": str(policy.agent_root / "work" / "solver.py"),
                    "content": "print(1)",
                },
            },
            "use5",
            None,
        )
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "Round 2" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_pre_hook_allows_repeated_code_in_same_round_after_plan(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    plan_guard = PlanGuard(policy.agent_root)
    (policy.agent_root / "plan.md").write_text(
        "# Plan\n\n## Round 1 - implement\nready\n",
        encoding="utf-8",
    )
    traj = TrajectoryWriter(tmp_path / "logs", run_id="r1")
    hook = make_pre_tool_use_hook(
        policy=policy,
        plan_guard=plan_guard,
        trajectory=traj,
        round_index_getter=lambda: 1,
    )

    write_out = asyncio.run(
        hook(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(policy.agent_root / "work" / "solver.py"),
                    "content": "print(1)",
                },
            },
            "use6",
            None,
        )
    )
    edit_out = asyncio.run(
        hook(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": str(policy.agent_root / "work" / "solver.py"),
                    "old_string": "1",
                    "new_string": "2",
                },
            },
            "use7",
            None,
        )
    )
    run_out = asyncio.run(
        hook(
            {"tool_name": "Bash", "tool_input": {"command": "python work/solver.py"}},
            "use8",
            None,
        )
    )

    assert "hookSpecificOutput" not in write_out
    assert "hookSpecificOutput" not in edit_out
    assert "hookSpecificOutput" not in run_out


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
    res = r._invalid(3, "missing_judge_adapter", "no adapter")
    assert res.feedback.verdict == INVALID
    assert res.feedback.failure_tags == ("missing_judge_adapter",)
    assert res.success is False
    persisted = tmp_path / "logs" / "judge_round_03.json"
    assert persisted.exists()
    data = json.loads(persisted.read_text(encoding="utf-8"))
    assert data["judge_result"]["_detail"] == "no adapter"
    assert "stdout_tail" in data
    assert "stderr_tail" in data


def test_judge_feedback_identifies_infrastructure_error() -> None:
    fb = JudgeFeedback(verdict=INVALID, failure_tags=("judge_runtime_error",))
    assert fb.is_infrastructure_error is True
    user_fixable = JudgeFeedback(verdict=INVALID, failure_tags=("missing_output",))
    assert user_fixable.is_infrastructure_error is False


def test_agent_runtime_env_uses_manifest_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", "OLDPATH")

    env = agent_runtime_env_overrides(
        {"runtime_env": {"ready": True, "python_executable": str(python)}}
    )

    assert env["PATH"].split(";")[0] == str(scripts)
    assert env["VIRTUAL_ENV"] == str(tmp_path / ".venv")


def test_agent_runtime_env_falls_back_without_ready_python(tmp_path: Path) -> None:
    env = agent_runtime_env_overrides(
        {"runtime_env": {"ready": False, "python_executable": str(tmp_path / "python")}}
    )
    assert env == {}


def test_prompts_prefer_workspace_root_execution(tmp_path: Path) -> None:
    prompt = initial_user_prompt(
        task_id="demo",
        primary_output_rel="output/x.npz",
        workspace_root=tmp_path,
        budget_seconds=60,
    )
    assert "python work/main.py" in prompt
    assert "do not\n        `cd work`" in prompt


def test_feedback_prompt_describes_previous_round() -> None:
    prompt = feedback_user_prompt(
        round_index=2,
        feedback=JudgeFeedback(verdict=FAIL),
        primary_output_rel="output/x.npz",
    )
    assert "Previous round judgement" in prompt
    assert "now starting Round 2" in prompt
