"""Skill distillation pipeline.

Mines reusable skills from past trajectories on the *train* split, then
validates each candidate skill against the *valid* split. The validator
defaults to one skill-injected run per selected valid task; an optional
baseline-vs-skill comparison mode is available for promotion experiments.

The package enforces a hard isolation boundary: nothing under
``distill.*`` is allowed to read source code, trajectories, or workspaces
belonging to a *valid* task. Access goes through
:class:`distill.universe.DistillUniverse` which audits and denies
out-of-split reads.

Phases (see ``doc/skill_distill_pipeline.md``):

    1. ``episode_miner``       - parse plan_round_*.md + trajectory.jsonl
    2. ``ref_differ``          - LLM-assisted alignment vs reference src/
    3. ``skill_synthesizer``   - LLM produces Anthropic-format SKILL.md
    4. ``skill_sanitizer``     - reject hardcoded / leaky / over-long skills
    5. ``transfer_validator``  - run skill-only or compare validation
"""

from .universe import DistillUniverse, ValidationLeakError
from .skill_sanitizer import SkillSanitizer, SanitizerReport
from .episode_miner import MainPyDigest, TaskEpisode, ToolUseEpisode, FailureSignal, mine_run, mine_train_split, scrub_text
from .skill_synthesizer import (
    SanitizationError,
    SkillSpec,
    collect_train_gap_evidence,
    synthesize_skill,
    write_skill_pack,
)

from .transfer_validator import (
    RunnerFn,
    TaskComparison,
    TransferReport,
    VERDICT_PROMOTE,
    VERDICT_REJECT_NO_NEW_PASS,
    VERDICT_REJECT_REGRESSION,
    stamp_promotion,
    validate_skill,
    write_transfer_report,
)

__all__ = [
    "DistillUniverse",
    "ValidationLeakError",
    "SkillSanitizer",
    "SanitizerReport",
    "TaskEpisode",
    "ToolUseEpisode",
    "FailureSignal",
    "MainPyDigest",
    "mine_run",
    "mine_train_split",
    "scrub_text",
    "SanitizationError",
    "SkillSpec",
    "collect_train_gap_evidence",
    "synthesize_skill",
    "write_skill_pack",

    # transfer_validator
    "RunnerFn",
    "TaskComparison",
    "TransferReport",
    "VERDICT_PROMOTE",
    "VERDICT_REJECT_NO_NEW_PASS",
    "VERDICT_REJECT_REGRESSION",
    "stamp_promotion",
    "validate_skill",
    "write_transfer_report",
]
