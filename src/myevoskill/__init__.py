"""MyEvoSkill — Claude-Code-harness agent for computational-imaging tasks.

Public surface (post-cleanup, master @ 41xxxxx onwards):

* ``myevoskill.cli``                -- the ``register-task`` / ``run-task`` /
                                       ``run-tasks-parallel`` entry points.
* ``myevoskill.registration``       -- deterministic v2-only task-manifest builder.
* ``myevoskill.harness``            -- ``HarnessConfig`` and ``run_task_once``.
* ``myevoskill.judge``              -- ``JudgeRunner`` + ``JudgeFeedback``.
* ``myevoskill.workspace``          -- ``WorkspacePolicy`` + ``build_workspace``.
* ``myevoskill.concurrency``        -- ``run_tasks_parallel``.

Modules retained purely as the **judge-adapter API** that every
``tasks/*/evaluation/judge_adapter.py`` imports (do not delete without a
coordinated migration of all task adapters):

* ``myevoskill.models``             -- ``RunRecord``, ``JudgeResult`` and
                                       supporting dataclasses.
* ``myevoskill.judging``            -- ``HiddenJudge`` + ``MetricRequirement``.
* ``myevoskill.task_contract``      -- contract loaders + metric-requirement helpers.
* ``myevoskill.task_runtime``       -- ``resolve_primary_output_path`` etc.
* ``myevoskill.resource_probe``     -- shape/value helpers used by some adapters.
"""

from __future__ import annotations

from typing import Any

# Keep package import lightweight. Judge adapters import submodules such as
# ``myevoskill.judging`` inside per-task virtual environments that do not need
# or install the Claude harness dependencies. Import harness-facing objects
# lazily so those adapters can use the stable judge API without pulling in
# ``claude_agent_sdk``.
_LAZY_EXPORTS = {
    "HarnessConfig": ("myevoskill.harness", "HarnessConfig"),
    "HarnessOutcome": ("myevoskill.harness", "HarnessOutcome"),
    "JudgeFeedback": ("myevoskill.judge", "JudgeFeedback"),
    "JudgeRunner": ("myevoskill.judge", "JudgeRunner"),
    "RegistrationError": ("myevoskill.registration", "RegistrationError"),
    "RegistrationResult": ("myevoskill.registration", "RegistrationResult"),
    "WorkspaceBuild": ("myevoskill.workspace", "WorkspaceBuild"),
    "WorkspacePolicy": ("myevoskill.workspace", "WorkspacePolicy"),
    "build_workspace": ("myevoskill.workspace", "build_workspace"),
    "register_task": ("myevoskill.registration", "register_task"),
    "run_task_once": ("myevoskill.harness", "run_task_once"),
    "run_tasks_parallel": ("myevoskill.concurrency", "run_tasks_parallel"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "HarnessConfig",
    "HarnessOutcome",
    "JudgeFeedback",
    "JudgeRunner",
    "RegistrationError",
    "RegistrationResult",
    "WorkspaceBuild",
    "WorkspacePolicy",
    "build_workspace",
    "register_task",
    "run_task_once",
    "run_tasks_parallel",
]
