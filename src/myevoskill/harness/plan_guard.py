"""plan.md update enforcement.

Behaviour:
    * Before any code-modifying tool (Write / Edit / Bash that writes a .py)
      or before launching code (Bash python ...), the agent MUST have edited
      ``plan.md`` *more recently* than the last code-modification.
    * On the very first round we seed an empty plan.md the agent then has to
      flesh out.
    * The check is implemented as part of the PreToolUse hook chain.

The point is to keep plan.md and code in lock-step so the trajectory shows
explicit reasoning before each new attempt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


CODE_MODIFY_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
CODE_RUN_TOOLS = {"Bash"}

PLAN_FILENAME = "plan.md"
# The seed file is intentionally a *template only*: it explains the required
# structure but does NOT pre-fill any "Round 1" entry.  The agent must author
# its own Round 1 block before it is allowed to edit/run code (enforced by
# PlanGuard.should_block).  This keeps the trajectory honest -- every plan
# entry the agent produces is authentically its own reasoning, not a fill-in
# of a template that was already on disk.
PLAN_SEED = (
    "# Plan\n\n"
    "Update this file **before** every coding action.  Append a new section\n"
    "for every iteration using the structure below.  The harness refuses to\n"
    "run or edit code while this file is older than your most recent code\n"
    "modification.\n\n"
    "Required structure for every entry:\n\n"
    "    ## Round <N> - <one line summary>\n"
    "    **Hypothesis:** what you think is wrong (or, for Round 1, what you\n"
    "    think the task needs).\n"
    "    **Change:** the *single* concrete edit you will make next.\n"
    "    **Verification:** the signal you will inspect after running\n"
    "    (printed value, output shape, sanity-check on a small slice, ...).\n\n"
    "Write your Round 1 block below before you touch any code.\n"
)


class PlanGuard:
    """Track plan.md vs code mtimes for one workspace."""

    def __init__(self, agent_root: Path) -> None:
        self.agent_root = Path(agent_root)
        self.plan_path = self.agent_root / PLAN_FILENAME
        self._last_code_mtime: float = 0.0
        if not self.plan_path.exists():
            self.plan_path.write_text(PLAN_SEED, encoding="utf-8")

    # ----------------------------------------------------------- predicates

    def _plan_mtime(self) -> float:
        try:
            return self.plan_path.stat().st_mtime
        except OSError:
            return 0.0

    def is_plan_fresh(self) -> bool:
        return self._plan_mtime() >= self._last_code_mtime

    def note_code_modification(self) -> None:
        # Stamp a tick after the agent edited code so the next plan.md edit
        # has to come *after* this point.
        import time

        self._last_code_mtime = max(self._last_code_mtime, time.time())

    # -------------------------------------------------------------- checks

    def should_block(self, tool_name: str, tool_input: Mapping[str, object]) -> str | None:
        """Return a deny reason or None.

        Allow:
            * any non-code tool (Read, Glob, Grep, ...)
            * any tool whose target IS plan.md (the only way to refresh it)
        Deny:
            * code-modifying / code-running tools while plan.md is stale
        """

        target = _tool_path(tool_input)

        if tool_name in CODE_MODIFY_TOOLS:
            if target.endswith(PLAN_FILENAME):
                # Editing plan.md itself is always fine.
                return None
            if not self.is_plan_fresh():
                return self._reason("edit code")
            return None

        if tool_name in CODE_RUN_TOOLS:
            command = str(tool_input.get("command") or "")
            if _is_code_run(command) and not self.is_plan_fresh():
                return self._reason("run code")
        return None

    @staticmethod
    def _reason(action: str) -> str:
        return (
            f"Plan check failed: you must update `plan.md` (with a new"
            f" 'Round N' entry describing your hypothesis, change, and"
            f" verification) before you {action}. Edit plan.md first, then"
            f" retry."
        )


# --------------------------------------------------------------------- helpers


def _tool_path(tool_input: Mapping[str, object]) -> str:
    for key in ("file_path", "path", "notebook_path"):
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            return v.replace("\\", "/")
    return ""


def _is_code_run(command: str) -> bool:
    if not command:
        return False
    head = command.strip().split()
    if not head:
        return False
    first = head[0].lower()
    return first in {
        "python",
        "python3",
        "py",
        "pytest",
        "pip",
        "uv",
        "ipython",
        "jupyter",
    }
