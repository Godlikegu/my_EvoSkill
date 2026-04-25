"""End-to-end harness runner.

A single :func:`run_task_once` call:

    1. Builds the agent workspace from the registered manifest.
    2. Opens a single Claude Agent SDK session.
    3. Loops:
        a. send the round's prompt (initial or feedback)
        b. stream messages, hooks log + enforce policy
        c. wait for the assistant to say it's READY (or budget exhausted)
        d. invoke the hidden judge
        e. if PASS or budget exhausted -> stop, else feed back judge verdict
    4. Writes a final ``run_summary.json`` and full ``trajectory.jsonl``.

We intentionally use *one* SDK session for all rounds so the agent retains its
own working memory of what it has already tried; we just inject a follow-up
user message per round.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UserMessage,
    ResultMessage,
)

from ..judge.bridge import JudgeFeedback, JudgeRunner, FAIL, INVALID, PASS
from ..workspace.builder import WorkspaceBuild, build_workspace
from .hooks import make_post_tool_use_hook, make_pre_tool_use_hook
from .plan_guard import PlanGuard
from .prompts import SYSTEM_PROMPT, feedback_user_prompt, initial_user_prompt
from .sandbox import (
    cleanup_isolated_home,
    env_overrides_for,
    make_isolated_home,
)
from .trajectory import TrajectoryWriter

logger = logging.getLogger(__name__)


# Hard cap on tool turns *inside one round* so a runaway agent cannot burn
# the entire wall-clock budget in a single conversation step.
DEFAULT_MAX_TURNS_PER_ROUND = 60


@dataclass
class HarnessConfig:
    """Static configuration for one task run."""

    repo_root: Path
    manifest: dict[str, Any]
    max_rounds: int = 4
    budget_seconds: int = 7200  # 2h per task
    max_turns_per_round: int = DEFAULT_MAX_TURNS_PER_ROUND
    model: str | None = None  # let SDK pick the default
    judge_python: str | None = None
    show_metric_status: bool = False  # if True, tell agent which metric failed (still no values)
    keep_workspace_on_success: bool = False
    log_root: Path | None = None  # default: artifacts/logs/<task>/<run_id>
    sandbox_root: Path | None = None  # default: artifacts/sandboxes/<task>/<run>/home
    keep_sandbox: bool = False  # opt-out of post-run sandbox cleanup


@dataclass
class HarnessOutcome:
    """Final result of one task run."""

    task_id: str
    run_id: str
    verdict: str  # PASS / FAIL / INVALID / TIMEOUT / ERROR
    rounds_used: int
    runtime_seconds: float
    workspace_root: Path
    log_root: Path
    trajectory_path: Path
    summary_path: Path
    error: str | None = None
    feedback_history: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------- runner


def run_task_once(config: HarnessConfig) -> HarnessOutcome:
    """Synchronous entry point - drives the asyncio loop internally."""

    return asyncio.run(_run_task_async(config))


async def _run_task_async(config: HarnessConfig) -> HarnessOutcome:
    repo_root = Path(config.repo_root).resolve()
    manifest = dict(config.manifest)
    task_id = str(manifest["task_id"])
    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    log_root = (
        Path(config.log_root)
        if config.log_root is not None
        else (repo_root / "artifacts" / "logs" / task_id / run_id)
    )
    log_root.mkdir(parents=True, exist_ok=True)

    # 1. Build the workspace.
    build = build_workspace(repo_root=repo_root, manifest=manifest, run_id=run_id)
    plan_guard = PlanGuard(build.agent_root)
    trajectory = TrajectoryWriter(log_root, run_id)
    judge = JudgeRunner(
        repo_root=repo_root,
        manifest=manifest,
        log_root=log_root,
        python_executable=config.judge_python,
    )

    # Per-run isolated $HOME so the spawned `claude` Node CLI does not write
    # into the operator's real ~/.claude (avoids state leakage between tasks
    # *and* avoids bloating user disk with conversation caches). Cleaned up
    # unconditionally in the finally below unless --keep-sandbox is set.
    sandbox = make_isolated_home(
        repo_root=repo_root,
        task_id=task_id,
        run_id=run_id,
        sandbox_root=Path(config.sandbox_root) if config.sandbox_root else None,
    )
    trajectory.env_feedback(
        0,
        "sandbox_ready",
        {
            "home_root": str(sandbox.home_root),
            "seeded_files": list(sandbox.seeded_files),
        },
    )

    # Mutable round counter exposed to the hooks via closure.
    round_state = {"round": 1}

    def round_getter() -> int:
        return int(round_state["round"])

    pre_hook = make_pre_tool_use_hook(
        policy=build.policy,
        plan_guard=plan_guard,
        trajectory=trajectory,
        round_index_getter=round_getter,
    )
    post_hook = make_post_tool_use_hook(
        trajectory=trajectory, round_index_getter=round_getter
    )

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        cwd=build.agent_root,
        # Restrict the agent to the workspace via add_dirs (positive list) and
        # disallow tools that bypass the sandbox.
        add_dirs=[],
        allowed_tools=[
            "Read",
            "Write",
            "Edit",
            "MultiEdit",
            "Glob",
            "Grep",
            "Bash",
            "TodoWrite",
        ],
        disallowed_tools=["WebSearch", "WebFetch"],
        permission_mode="default",
        max_turns=config.max_turns_per_round,
        model=config.model,
        hooks={
            "PreToolUse": [HookMatcher(matcher="*", hooks=[pre_hook])],
            "PostToolUse": [HookMatcher(matcher="*", hooks=[post_hook])],
        },
        env=env_overrides_for(sandbox),
        setting_sources=None,  # don't pick up user-level CLAUDE.md etc.
    )

    started = time.time()
    feedback_history: list[dict[str, Any]] = []
    final_verdict: str | None = None
    error_message: str | None = None

    try:
      async with ClaudeSDKClient(options=options) as client:
       for round_index in range(1, config.max_rounds + 1):
            round_state["round"] = round_index
            elapsed = time.time() - started
            remaining = max(1, config.budget_seconds - int(elapsed))
            trajectory.round_marker(
                round_index,
                "start",
                {"elapsed_seconds": elapsed, "budget_remaining": remaining},
            )

            if round_index == 1:
                user_msg = initial_user_prompt(
                    task_id=task_id,
                    primary_output_rel=build.policy.primary_output_rel,
                    workspace_root=build.agent_root,
                    task_spec_summary=build.agent_task_spec_summary,
                    budget_seconds=config.budget_seconds,
                )
            else:
                last = feedback_history[-1]["feedback"]
                user_msg = feedback_user_prompt(
                    round_index=round_index,
                    feedback=JudgeFeedback(**last),
                    primary_output_rel=build.policy.primary_output_rel,
                    show_metric_status=config.show_metric_status,
                )

            # Send the round prompt and let the agent run until it stops or the
            # per-round turn cap fires.
            try:
                await asyncio.wait_for(
                    client.query(user_msg),
                    timeout=remaining,
                )
                async for message in _bounded_receive(client, deadline=started + config.budget_seconds):
                    _record_message(trajectory, round_index, message)
                    if isinstance(message, ResultMessage):
                        break
            except asyncio.TimeoutError:
                trajectory.round_marker(round_index, "timeout", {"phase": "agent"})
                final_verdict = "TIMEOUT"
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent round failed")
                trajectory.round_marker(
                    round_index, "error", {"phase": "agent", "error": repr(exc)}
                )
                error_message = repr(exc)
                final_verdict = "ERROR"
                break

            # Round complete: invoke the judge.
            judge_run = judge.run(
                round_index=round_index,
                run_id=run_id,
                workspace_root=build.agent_root,
            )
            entry = {
                "round": round_index,
                "feedback": asdict(judge_run.feedback),
                "judge_runtime_seconds": judge_run.runtime_seconds,
                "judge_success": judge_run.success,
            }
            feedback_history.append(entry)
            trajectory.round_marker(round_index, "judged", entry)
            trajectory.env_feedback(
                round_index,
                "judge_verdict",
                {"verdict": judge_run.feedback.verdict, "failure_tags": list(judge_run.feedback.failure_tags)},
            )

            if judge_run.feedback.verdict == PASS:
                final_verdict = PASS
                break

            # Budget check before next round.
            if (time.time() - started) >= config.budget_seconds:
                final_verdict = "TIMEOUT"
                break
       else:
            # max_rounds exhausted without PASS
            final_verdict = feedback_history[-1]["feedback"]["verdict"] if feedback_history else INVALID
    finally:
        # Always wipe (or keep, with --keep-sandbox) the per-run isolated
        # $HOME so we never leak state between tasks. We do this in finally
        # so a crashed agent or SDK exception still cleans up.
        cleanup_isolated_home(sandbox, keep=bool(config.keep_sandbox))

    runtime = time.time() - started
    summary = {
        "task_id": task_id,
        "run_id": run_id,
        "verdict": final_verdict,
        "rounds_used": len(feedback_history) if feedback_history else 0,
        "runtime_seconds": runtime,
        "workspace_root": str(build.agent_root),
        "log_root": str(log_root),
        "feedback_history": feedback_history,
        "policy": {
            "agent_root": str(build.policy.agent_root),
            "primary_output_rel": build.policy.primary_output_rel,
            "forbidden_substrings": list(build.policy.all_forbidden_substrings()),
        },
        "copied_files": list(build.copied_files),
        "skipped_files": list(build.skipped_files),
        "error": error_message,
    }
    summary_path = log_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Optional cleanup of the workspace on PASS to save disk.
    if final_verdict == PASS and not config.keep_workspace_on_success:
        try:
            shutil.rmtree(build.agent_root, ignore_errors=True)
        except OSError:
            pass

    return HarnessOutcome(
        task_id=task_id,
        run_id=run_id,
        verdict=final_verdict or "ERROR",
        rounds_used=len(feedback_history),
        runtime_seconds=runtime,
        workspace_root=build.agent_root,
        log_root=log_root,
        trajectory_path=trajectory.path,
        summary_path=summary_path,
        error=error_message,
        feedback_history=feedback_history,
    )


# --------------------------------------------------------------------- helpers


async def _bounded_receive(client: ClaudeSDKClient, *, deadline: float):
    """Yield messages but stop if the wall-clock deadline is reached."""

    async for message in client.receive_response():
        yield message
        if time.time() >= deadline:
            return


def _record_message(
    trajectory: TrajectoryWriter, round_index: int, message: Any
) -> None:
    """Translate an SDK message into trajectory events."""

    if isinstance(message, AssistantMessage):
        for block in getattr(message, "content", []) or []:
            if isinstance(block, TextBlock):
                trajectory.assistant_text(round_index, block.text)
            elif isinstance(block, ThinkingBlock):
                trajectory.assistant_thinking(round_index, getattr(block, "thinking", "") or "")
            elif isinstance(block, ToolUseBlock):
                # tool_use events are also captured by the PreToolUse hook,
                # but we additionally emit them here for trajectories where
                # the hook didn't fire (e.g. denied at PermissionRequest).
                trajectory.tool_call(
                    round_index,
                    getattr(block, "name", "") or "",
                    dict(getattr(block, "input", {}) or {}),
                    tool_use_id=getattr(block, "id", None),
                )
    elif isinstance(message, SystemMessage):
        # We don't surface system messages to the user; just record their
        # subtype for debugging.
        sub = getattr(message, "subtype", "") or ""
        if sub:
            trajectory.env_feedback(round_index, "system", {"subtype": sub})
    elif isinstance(message, ResultMessage):
        payload: dict[str, Any] = {}
        for attr in ("subtype", "is_error", "duration_ms", "duration_api_ms", "num_turns", "total_cost_usd"):
            v = getattr(message, attr, None)
            if v is not None:
                payload[attr] = v
        trajectory.round_marker(round_index, "agent_done", payload)
