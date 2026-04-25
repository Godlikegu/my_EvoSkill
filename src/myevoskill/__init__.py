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

# Re-export the live harness surface so ``from myevoskill import run_task_once``
# keeps working for callers that don't want to touch the submodule layout.
from .registration import RegistrationError, RegistrationResult, register_task
from .harness import HarnessConfig, HarnessOutcome, run_task_once
from .judge import JudgeFeedback, JudgeRunner
from .workspace import WorkspaceBuild, WorkspacePolicy, build_workspace
from .concurrency import run_tasks_parallel

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
