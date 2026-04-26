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
    UserMessage,
    ResultMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
)

from ..judge.bridge import JudgeFeedback, JudgeRunner, FAIL, INVALID, PASS
from ..workspace.builder import WorkspaceBuild, build_workspace
from .hooks import make_post_tool_use_hook, make_pre_tool_use_hook
from .plan_guard import PlanGuard
from .plan_history import PlanHistoryRecorder
from .process_reaper import ReapResult, make_run_markers, reap_descendant_processes
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


@dataclass
class ActiveTask:
    """Claude Code background task tracked by SDK system messages."""

    task_id: str
    description: str = ""
    tool_use_id: str | None = None
    task_type: str | None = None
    status: str = "started"


class ActiveTaskRegistry:
    """Small state holder for Claude Code local Bash task lifecycle."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveTask] = {}

    def start(self, message: TaskStartedMessage) -> ActiveTask:
        task = ActiveTask(
            task_id=message.task_id,
            description=message.description,
            tool_use_id=message.tool_use_id,
            task_type=message.task_type,
            status="started",
        )
        self._active[task.task_id] = task
        return task

    def progress(self, message: TaskProgressMessage) -> ActiveTask | None:
        task = self._active.get(message.task_id)
        if task is not None:
            task.status = "running"
        return task

    def finish(self, message: TaskNotificationMessage) -> ActiveTask:
        task = self._active.pop(
            message.task_id,
            ActiveTask(
                task_id=message.task_id,
                description=message.summary,
                tool_use_id=message.tool_use_id,
                status=message.status,
            ),
        )
        task.status = message.status
        return task

    def active(self) -> list[ActiveTask]:
        return list(self._active.values())

    def clear(self) -> None:
        self._active.clear()

    def __bool__(self) -> bool:
        return bool(self._active)


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
    plan_history = PlanHistoryRecorder(
        workspace_root=build.agent_root, log_root=log_root
    )
    trajectory = TrajectoryWriter(log_root, run_id)
    task_registry = ActiveTaskRegistry()
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

    agent_env = env_overrides_for(sandbox)
    venv_env = agent_runtime_env_overrides(manifest)
    if venv_env:
        agent_env.update(venv_env)
    else:
        trajectory.env_feedback(
            0,
            "runtime_env_warning",
            {
                "reason": (
                    "manifest has no ready runtime_env.python_executable; "
                    "agent Bash will use inherited PATH"
                )
            },
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
        env=agent_env,
        setting_sources=None,  # don't pick up user-level CLAUDE.md etc.
    )

    started = time.time()
    feedback_history: list[dict[str, Any]] = []
    final_verdict: str | None = None
    error_message: str | None = None
    claude_pid: int | None = None
    cleanup_errors: list[dict[str, Any]] = []

    try:
      async with ClaudeSDKClient(options=options) as client:
       claude_pid = _client_transport_pid(client)
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
                    _record_message(
                        trajectory,
                        round_index,
                        message,
                        task_registry=task_registry,
                    )
                    if isinstance(message, ResultMessage):
                        break
            except asyncio.TimeoutError:
                trajectory.round_marker(round_index, "timeout", {"phase": "agent"})
                clean = await _cleanup_agent_processes(
                    client=client,
                    task_registry=task_registry,
                    trajectory=trajectory,
                    round_index=round_index,
                    claude_pid=claude_pid,
                    run_id=run_id,
                    workspace_root=build.agent_root,
                    sandbox_root=sandbox.home_root,
                    phase="agent_timeout",
                    require_clean=True,
                )
                cleanup_errors.extend(clean.get("errors", []))
                if clean["ok"]:
                    final_verdict = "TIMEOUT"
                else:
                    final_verdict = "ERROR"
                    error_message = "process cleanup failed after agent timeout"
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent round failed")
                trajectory.round_marker(
                    round_index, "error", {"phase": "agent", "error": repr(exc)}
                )
                clean = await _cleanup_agent_processes(
                    client=client,
                    task_registry=task_registry,
                    trajectory=trajectory,
                    round_index=round_index,
                    claude_pid=claude_pid,
                    run_id=run_id,
                    workspace_root=build.agent_root,
                    sandbox_root=sandbox.home_root,
                    phase="agent_error",
                    require_clean=True,
                )
                cleanup_errors.extend(clean.get("errors", []))
                error_message = (
                    repr(exc)
                    if clean["ok"]
                    else f"{repr(exc)}; process cleanup failed after agent error"
                )
                final_verdict = "ERROR"
                break

            clean = await _cleanup_agent_processes(
                client=client,
                task_registry=task_registry,
                trajectory=trajectory,
                round_index=round_index,
                claude_pid=claude_pid,
                run_id=run_id,
                workspace_root=build.agent_root,
                sandbox_root=sandbox.home_root,
                phase="before_judge",
                require_clean=True,
            )
            cleanup_errors.extend(clean.get("errors", []))
            if not clean["ok"]:
                final_verdict = "ERROR"
                error_message = "process cleanup failed before judge"
                break

            # Round complete: snapshot plan.md *before* the judge runs so we
            # capture exactly what the agent claimed it was attempting this
            # round, even if the judge is what flips the run into TIMEOUT/
            # ERROR on the very next round.
            plan_snap = plan_history.snapshot(round_index)
            trajectory.env_feedback(
                round_index,
                "plan_snapshot",
                {
                    "snapshot_path": str(plan_snap.snapshot_path),
                    "size_bytes": plan_snap.size_bytes,
                    "sha256": plan_snap.sha256,
                    "diff_lines": plan_snap.diff_lines,
                    "note": plan_snap.note,
                },
            )

            # Now invoke the judge.
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

            if judge_run.feedback.is_infrastructure_error:
                final_verdict = "ERROR"
                error_message = (
                    "judge infrastructure error: "
                    + ",".join(judge_run.feedback.failure_tags)
                )
                break

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
       final_clean = await _cleanup_agent_processes(
            client=client,
            task_registry=task_registry,
            trajectory=trajectory,
            round_index=max(1, len(feedback_history)),
            claude_pid=claude_pid,
            run_id=run_id,
            workspace_root=build.agent_root,
            sandbox_root=sandbox.home_root,
            phase="session_final",
            require_clean=False,
       )
       cleanup_errors.extend(final_clean.get("errors", []))
    finally:
        if claude_pid is not None:
            final_reap = _reap_run_processes(
                claude_pid=claude_pid,
                run_id=run_id,
                workspace_root=build.agent_root,
                sandbox_root=sandbox.home_root,
                kill=True,
            )
            if final_reap.killed or final_reap.remaining or final_reap.errors:
                trajectory.env_feedback(
                    max(1, len(feedback_history)),
                    "process_cleanup_final",
                    final_reap.to_dict(),
                )
                if not final_reap.ok:
                    cleanup_errors.append(final_reap.to_dict())
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
        "plan_history": plan_history.read_history(),
        "process_cleanup_errors": cleanup_errors,
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


async def _cleanup_agent_processes(
    *,
    client: ClaudeSDKClient,
    task_registry: ActiveTaskRegistry,
    trajectory: TrajectoryWriter,
    round_index: int,
    claude_pid: int | None,
    run_id: str,
    workspace_root: Path,
    sandbox_root: Path,
    phase: str,
    require_clean: bool,
) -> dict[str, Any]:
    """Stop Claude background tasks and reap leftover run-scoped children."""

    active = task_registry.active()
    for task in active:
        trajectory.env_feedback(
            round_index,
            "task_stop_requested",
            {
                "phase": phase,
                "task_id": task.task_id,
                "tool_use_id": task.tool_use_id,
                "description": task.description,
            },
        )
        try:
            await client.stop_task(task.task_id)
        except Exception as exc:  # noqa: BLE001
            trajectory.env_feedback(
                round_index,
                "task_stop_error",
                {"phase": phase, "task_id": task.task_id, "error": repr(exc)},
            )

    if active:
        await _drain_task_notifications(
            client=client,
            task_registry=task_registry,
            trajectory=trajectory,
            round_index=round_index,
            timeout_seconds=10.0,
        )

    reap = _reap_run_processes(
        claude_pid=claude_pid,
        run_id=run_id,
        workspace_root=workspace_root,
        sandbox_root=sandbox_root,
        kill=True,
    )
    if reap.killed or reap.remaining or reap.errors or active:
        trajectory.env_feedback(
            round_index,
            "process_cleanup_barrier",
            {
                "phase": phase,
                "active_tasks_before": [asdict(t) for t in active],
                **reap.to_dict(),
            },
        )
    if active and reap.ok and claude_pid is not None:
        # Claude may not emit a stopped notification for tasks killed by the
        # OS-level fallback. Once no matching child processes remain, the
        # registry should not block judge execution.
        task_registry.clear()

    ok = not require_clean or (not task_registry.active() and reap.ok)
    errors = []
    if not ok:
        payload = {
            "phase": phase,
            "active_tasks": [asdict(t) for t in task_registry.active()],
            **reap.to_dict(),
        }
        trajectory.env_feedback(round_index, "process_cleanup_failed", payload)
        errors.append(payload)
    return {"ok": ok, "errors": errors, "reap": reap.to_dict()}


async def _drain_task_notifications(
    *,
    client: ClaudeSDKClient,
    task_registry: ActiveTaskRegistry,
    trajectory: TrajectoryWriter,
    round_index: int,
    timeout_seconds: float,
) -> None:
    deadline = time.time() + timeout_seconds
    messages = client.receive_messages()
    while task_registry.active() and time.time() < deadline:
        remaining = max(0.05, deadline - time.time())
        try:
            message = await asyncio.wait_for(
                messages.__anext__(),
                timeout=remaining,
            )
        except (asyncio.TimeoutError, StopAsyncIteration):
            return
        _record_message(
            trajectory,
            round_index,
            message,
            task_registry=task_registry,
        )


def _client_transport_pid(client: ClaudeSDKClient) -> int | None:
    transport = getattr(client, "_transport", None)
    process = getattr(transport, "_process", None)
    pid = getattr(process, "pid", None)
    return int(pid) if pid is not None else None


def _reap_run_processes(
    *,
    claude_pid: int | None,
    run_id: str,
    workspace_root: Path,
    sandbox_root: Path,
    kill: bool,
) -> ReapResult:
    markers = make_run_markers(
        run_id=run_id,
        workspace_root=workspace_root,
        sandbox_root=sandbox_root,
    )
    return reap_descendant_processes(
        root_pid=claude_pid,
        markers=markers,
        kill=kill,
    )


def _record_message(
    trajectory: TrajectoryWriter,
    round_index: int,
    message: Any,
    *,
    task_registry: ActiveTaskRegistry | None = None,
) -> None:
    """Translate an SDK message into trajectory events."""

    if isinstance(message, AssistantMessage):
        for block in getattr(message, "content", []) or []:
            if isinstance(block, TextBlock):
                trajectory.assistant_text(round_index, block.text)
            elif isinstance(block, ThinkingBlock):
                trajectory.assistant_thinking(round_index, getattr(block, "thinking", "") or "")
    elif isinstance(message, TaskStartedMessage):
        task = task_registry.start(message) if task_registry is not None else None
        trajectory.env_feedback(
            round_index,
            "task_started",
            {
                "task_id": message.task_id,
                "tool_use_id": message.tool_use_id,
                "description": message.description,
                "task_type": message.task_type,
                "active": asdict(task) if task else None,
            },
        )
    elif isinstance(message, TaskProgressMessage):
        task = task_registry.progress(message) if task_registry is not None else None
        trajectory.env_feedback(
            round_index,
            "task_progress",
            {
                "task_id": message.task_id,
                "tool_use_id": message.tool_use_id,
                "description": message.description,
                "last_tool_name": message.last_tool_name,
                "usage": dict(message.usage or {}),
                "active": asdict(task) if task else None,
            },
        )
    elif isinstance(message, TaskNotificationMessage):
        task = task_registry.finish(message) if task_registry is not None else None
        trajectory.env_feedback(
            round_index,
            f"task_{message.status}",
            {
                "task_id": message.task_id,
                "tool_use_id": message.tool_use_id,
                "status": message.status,
                "summary": message.summary,
                "output_file": message.output_file,
                "usage": dict(message.usage or {}),
                "active": asdict(task) if task else None,
            },
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


def agent_runtime_env_overrides(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return env overrides that make agent Bash use the per-task venv.

    The judge may be explicitly overridden with ``--judge-python``; the agent
    environment instead follows the registered task runtime. If no ready
    per-task interpreter exists, callers should use the inherited PATH.
    """

    runtime_env = manifest.get("runtime_env") or {}
    if not isinstance(runtime_env, Mapping):
        return {}
    if not bool(runtime_env.get("ready")):
        return {}
    python_executable = str(runtime_env.get("python_executable") or "").strip()
    if not python_executable:
        return {}
    python_path = Path(python_executable)
    if not python_path.exists():
        return {}

    bin_dir = python_path.parent
    if bin_dir.name.lower() == "scripts":
        venv_root = bin_dir.parent
    elif bin_dir.name == "bin":
        venv_root = bin_dir.parent
    else:
        venv_root = bin_dir.parent

    old_path = os.environ.get("PATH", "")
    return {
        "PATH": str(bin_dir) + os.pathsep + old_path,
        "VIRTUAL_ENV": str(venv_root),
    }
