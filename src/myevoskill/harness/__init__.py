"""Claude-Code harness for MyEvoSkill.

This package contains everything needed to run a registered task with the
Claude Agent SDK:

    * ``hooks``      - PreToolUse/PostToolUse policy enforcement
    * ``plan_guard`` - require an updated ``plan.md`` before code edits
    * ``prompts``    - assemble the system / user prompts (round 1 vs N+1)
    * ``trajectory`` - record assistant text, tool calls, tool results, env
                        feedback into a single jsonl
    * ``runner``     - orchestrate one task run (build workspace, talk to
                        the judge, iterate, aggregate trajectory)

Public entry points:
    * :func:`run_task_once` - run a single task to completion
"""

from .runner import HarnessConfig, HarnessOutcome, run_task_once

__all__ = ["HarnessConfig", "HarnessOutcome", "run_task_once"]
