"""plan.md update enforcement.

Behaviour:
    * Before the first code-modifying tool or code run in judge round N, the
      agent MUST have authored a top-level ``## Round N`` section in
      ``plan.md``.
    * Once that round section exists, all code edits and runs within the same
      judge round are allowed. A failed judge result advances the harness to
      the next round, which requires a new ``## Round N+1`` section.
    * The check is implemented as part of the PreToolUse hook chain.

The point is to keep plan.md aligned with judge attempts without forcing a
new plan entry before every single incremental edit inside one attempt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping


CODE_MODIFY_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
CODE_RUN_TOOLS = {"Bash"}

PLAN_FILENAME = "plan.md"
PLAN_SEED = "# Plan\n\n"


class PlanGuard:
    """Enforce one authored plan section per judge round."""

    def __init__(self, agent_root: Path) -> None:
        self.agent_root = Path(agent_root)
        self.plan_path = self.agent_root / PLAN_FILENAME
        if not self.plan_path.exists():
            self.plan_path.write_text(PLAN_SEED, encoding="utf-8")

    # ----------------------------------------------------------- predicates

    def has_round_plan(self, round_index: int) -> bool:
        """Return True iff plan.md contains a top-level ``## Round N``.

        The heading must start at column zero. Indented examples such as
        ``    ## Round 1`` are intentionally ignored.
        """

        if round_index < 1:
            return False
        try:
            text = self.plan_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        pattern = re.compile(rf"(?m)^##\s+Round\s+{round_index}\b")
        return bool(pattern.search(text))

    def is_plan_fresh(self, round_index: int = 1) -> bool:
        """Backwards-compatible predicate for tests/callers.

        Freshness now means "the current judge round has a plan", not "plan.md
        has a newer mtime than the last code edit".
        """

        return self.has_round_plan(round_index)

    def note_code_modification(self) -> None:
        """Deprecated no-op kept for older callers/tests.

        Plan freshness is round-based now; code edits inside a round do not
        make the plan stale.
        """

    # -------------------------------------------------------------- checks

    def should_block(
        self,
        tool_name: str,
        tool_input: Mapping[str, object],
        *,
        round_index: int = 1,
    ) -> str | None:
        """Return a deny reason or None.

        Allow:
            * any non-code tool (Read, Glob, Grep, ...)
            * any tool whose target IS plan.md (the only way to refresh it)
        Deny:
            * code-modifying / code-running tools while this round lacks a
              top-level ``## Round N`` entry
        """

        target = _tool_path(tool_input)

        if tool_name in CODE_MODIFY_TOOLS:
            if target.endswith(PLAN_FILENAME):
                # Editing plan.md itself is always fine.
                return None
            if not self.has_round_plan(round_index):
                return self._reason("edit code", round_index)
            return None

        if tool_name in CODE_RUN_TOOLS:
            command = str(tool_input.get("command") or "")
            if _is_code_run(command) and not self.has_round_plan(round_index):
                return self._reason("run code", round_index)
        return None

    @staticmethod
    def _reason(action: str, round_index: int) -> str:
        return (
            f"Plan check failed: before you {action} in judge round"
            f" {round_index}, update `plan.md` with a top-level"
            f" `## Round {round_index}` entry describing your hypothesis,"
            f" change, and verification. Then retry."
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
    return bool(
        re.search(
            r"(?i)(?:^|[;&|]\s*)"
            r"(?:python|python3|py|pytest|pip|uv|ipython|jupyter)\b",
            command.strip(),
        )
    )
