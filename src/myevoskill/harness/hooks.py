"""PreToolUse / PostToolUse hooks for the Claude Agent SDK.

These hooks are the *only* place we enforce sandbox policy. They:

    1. Reject tool calls that escape the workspace (path is outside agent_root
       or contains a forbidden substring like ``ground_truth``).
    2. Reject Bash commands that match a dangerous pattern (sudo, curl,
       cd /, ...).
    3. Enforce the plan.md freshness rule.
    4. Append every observed tool call / tool result to the trajectory.

Returning ``hookSpecificOutput.permissionDecision = "deny"`` causes the SDK to
*intercept* the tool call - the agent sees a tool result with
``permissionDecisionReason`` and can react, instead of the harness silently
killing the run.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Mapping

from ..workspace.policy import WorkspacePolicy, collect_path_args
from .plan_guard import CODE_MODIFY_TOOLS, PLAN_FILENAME, PlanGuard
from .trajectory import TrajectoryWriter

logger = logging.getLogger(__name__)

HookCallable = Callable[[dict, str | None, Any], Awaitable[dict]]


def make_pre_tool_use_hook(
    *,
    policy: WorkspacePolicy,
    plan_guard: PlanGuard,
    trajectory: TrajectoryWriter,
    round_index_getter: Callable[[], int],
) -> HookCallable:
    """Build the PreToolUse hook closure."""

    async def pre_tool_use(
        input_data: dict, tool_use_id: str | None, context: Any
    ) -> dict:
        tool_name = str(input_data.get("tool_name") or "")
        tool_input: Mapping[str, Any] = dict(input_data.get("tool_input") or {})

        round_idx = round_index_getter()

        # 1. Trajectory: log the *attempt*, even if denied.
        trajectory.tool_call(round_idx, tool_name, tool_input, tool_use_id)

        # 2. Path checks (apply to every tool with path-like args).
        deny_reason = _check_paths(policy, tool_name, tool_input)
        if deny_reason:
            trajectory.env_feedback(
                round_idx,
                "policy_deny",
                {"tool": tool_name, "reason": deny_reason, "kind": "path"},
            )
            return _deny(deny_reason)

        # 3. Bash-specific checks.
        if tool_name == "Bash":
            command = str(tool_input.get("command") or "")
            bash_reason = policy.find_dangerous_bash(command)
            if bash_reason:
                msg = (
                    f"Bash command rejected: it matches the harness denylist"
                    f" pattern `{bash_reason}`. Stay inside the workspace and"
                    f" avoid network / privilege-escalation commands."
                )
                trajectory.env_feedback(
                    round_idx,
                    "policy_deny",
                    {"tool": tool_name, "reason": msg, "kind": "bash"},
                )
                return _deny(msg)

        # 4. plan.md freshness.
        plan_reason = plan_guard.should_block(tool_name, tool_input)
        if plan_reason:
            trajectory.env_feedback(
                round_idx,
                "plan_guard_deny",
                {"tool": tool_name, "reason": plan_reason},
            )
            return _deny(plan_reason)

        # If this call is a code modification, mark a tick *after* the call
        # is allowed; the next plan.md edit will then count as fresher.
        if tool_name in CODE_MODIFY_TOOLS:
            target = _path_value(tool_input)
            if not target.endswith(PLAN_FILENAME):
                plan_guard.note_code_modification()

        return {}

    return pre_tool_use


def make_post_tool_use_hook(
    *,
    trajectory: TrajectoryWriter,
    round_index_getter: Callable[[], int],
) -> HookCallable:
    """Append the tool result to the trajectory."""

    async def post_tool_use(
        input_data: dict, tool_use_id: str | None, context: Any
    ) -> dict:
        tool_response = input_data.get("tool_response")
        is_error = bool(input_data.get("is_error") or False)
        text = _extract_text(tool_response)
        trajectory.tool_result(
            round_index_getter(), tool_use_id, text, is_error
        )
        return {}

    return post_tool_use


# --------------------------------------------------------------------- helpers


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _check_paths(
    policy: WorkspacePolicy, tool_name: str, tool_input: Mapping[str, Any]
) -> str | None:
    # Don't path-check the Bash tool here; we apply the substring check via
    # find_forbidden on the command string just below.
    candidate_paths = collect_path_args(tool_input)

    # 1. Substring deny across all string args.
    for raw in candidate_paths:
        sub = policy.find_forbidden(raw)
        if sub:
            return (
                f"Tool input contains forbidden substring '{sub}'. The"
                f" workspace policy blocks access to hidden assets such as"
                f" ground truth, evaluation/, and reference solutions."
            )

    # 2. For path-shaped tools (Read/Write/Edit/Glob/Grep/...): the file_path
    #    must resolve inside the workspace.
    for key in ("file_path", "path", "notebook_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            if not policy.is_inside(value):
                return (
                    f"Path '{value}' is outside the agent workspace"
                    f" ({policy.agent_root}). All reads/writes must stay"
                    f" inside the workspace."
                )

    # 3. For Glob / Grep, the same applies to their `path`.
    if tool_name in {"Glob", "Grep"}:
        path = tool_input.get("path")
        if isinstance(path, str) and path and not policy.is_inside(path):
            return (
                f"Glob/Grep path '{path}' is outside the workspace"
                f" ({policy.agent_root})."
            )

    return None


def _path_value(tool_input: Mapping[str, Any]) -> str:
    for key in ("file_path", "path", "notebook_path"):
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            return v.replace("\\", "/")
    return ""


def _extract_text(response: Any) -> str:
    """Pull a plain-text representation out of a tool_response."""

    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("output", "stdout", "text", "content"):
            v = response.get(key)
            if isinstance(v, str):
                return v
        return repr(dict(response))
    if isinstance(response, list):
        parts = []
        for item in response:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(response)
