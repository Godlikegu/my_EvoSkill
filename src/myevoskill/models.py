"""Core dataclasses shared across MyEvoSkill modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class READMEPolicy:
    """Policy controlling README preservation and sanitization."""

    preserve_sections: Sequence[str] = field(default_factory=list)
    remove_sections: Sequence[str] = field(default_factory=list)
    remove_path_patterns: Sequence[str] = field(default_factory=list)
    preserve_user_eval_notes: bool = True


@dataclass(frozen=True)
class PublicExposurePolicy:
    """Policy controlling public bundle exposure."""

    readme_policy: READMEPolicy = field(default_factory=READMEPolicy)
    public_data_allowlist: Sequence[str] = field(default_factory=list)
    public_data_denylist: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskBundle:
    """Compiled task bundle with public and hidden bundle paths."""

    task_id: str
    family: str
    root_dir: Path
    public_bundle_dir: Path
    hidden_bundle_dir: Path
    task_spec_path: Path
    compile_report_path: Path
    readme_public_path: Optional[Path] = None


@dataclass(frozen=True)
class BootstrapResult:
    """Artifacts produced while bootstrapping a task manifest."""

    task_id: str
    manifest_path: Path
    judge_stub_path: Path
    notes_path: Path
    missing_items: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContractDraftResult:
    """Artifacts produced while drafting a registration contract."""

    task_id: str
    draft_path: Path
    notes_path: Path
    missing_items: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    attempt_count: int = 0
    attempt_summaries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TaskRegistrationResult:
    """Artifacts produced while registering a confirmed task contract."""

    task_id: str
    manifest_path: Path
    judge_path: Path
    notes_path: Path
    missing_items: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EnvCacheRecord:
    """Resolved environment cache directories for one env hash."""

    env_hash: str
    env_dir: Path
    base_image_cache_dir: Path
    task_env_cache_dir: Path
    dataset_cache_dir: Path
    artifact_cache_dir: Path
    checkpoint_cache_dir: Path


@dataclass(frozen=True)
class DevEnvironmentInfo:
    """Metadata about the current MyEvoSkill development environment."""

    conda_env_name: str
    conda_prefix: str
    python_executable: str
    allow_bridge_execution: bool
    allow_task_env_build: bool


@dataclass(frozen=True)
class RuntimeEnvironmentPolicy:
    """Policy for development vs task runtime environments."""

    dev_environment_name: str = "myevoskill"
    allow_bridge_execution: bool = True
    allow_task_env_build: bool = True


@dataclass(frozen=True)
class TaskOutcome:
    """Task-level success outcome used for transfer validation."""

    task_id: str
    all_metrics_passed: bool
    metrics_actual: Dict[str, float] = field(default_factory=dict)
    failed_metrics: List[str] = field(default_factory=list)
    failure_tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    """User-supplied or built-in model provider configuration."""

    provider_name: str
    model_name: str
    base_url: str = ""
    api_key_env: str = ""
    timeout: int = 0
    max_tokens: int = 0
    temperature: float = 0.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutorSessionConfig:
    """Executor session configuration for one run."""

    run_id: str
    env_hash: str
    workspace_root: Optional[Path] = None
    budget_seconds: int = 0
    execution_mode: str = "standard"
    compute_profile: str = "mixed"
    command: Sequence[str] = field(default_factory=list)
    env: Mapping[str, str] = field(default_factory=dict)
    tool_policy: Mapping[str, Any] = field(default_factory=dict)
    provider_extras: Mapping[str, Any] = field(default_factory=dict)
    model_config: Optional[ModelConfig] = None


@dataclass(frozen=True)
class RunPaths:
    """Resolved persistent run directories for one task/run pair."""

    repo_root: Path
    task_id: str
    run_id: str
    workspace_root: Path
    log_root: Path


@dataclass(frozen=True)
class EffectiveRuntimePolicy:
    """Resolved timeout policy after session, manifest, and defaults merge."""

    model_timeout_seconds: int
    execution_budget_seconds: int


@dataclass(frozen=True)
class ProxyFeedback:
    """Low-leakage proxy verification result."""

    task_id: str
    output_exists: bool
    output_shape: Sequence[int] = field(default_factory=list)
    output_dtype: str = ""
    has_nan_or_inf: bool = False
    runtime_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    public_baseline_delta: Dict[str, float] = field(default_factory=dict)
    physical_checks: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeResult:
    """Hidden judge result contract."""

    task_id: str
    all_metrics_passed: bool
    metrics_actual: Dict[str, float]
    failed_metrics: List[str]
    failure_tags: List[str]


@dataclass(frozen=True)
class RunRecord:
    """Provider-neutral execution record."""

    run_id: str
    task_id: str
    provider: str
    env_hash: str
    skills_active: Sequence[str]
    workspace_root: Path
    provider_session_id: str = ""
    model_provider: str = ""
    model_name: str = ""
    artifacts_uri: str = ""
    transcript_uri: str = ""
    stdout: str = ""
    stderr: str = ""
    runtime_seconds: float = 0.0
    proxy_feedback: Optional[ProxyFeedback] = None
    judge_result: Optional[JudgeResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillCandidate:
    """Candidate skill metadata before registry promotion."""

    skill_id: str
    version: str
    description: str
    source_run_ids: Sequence[str]
    legal_source: bool
    reusable: bool
    applicability: Sequence[str] = field(default_factory=list)
    known_failure_modes: Sequence[str] = field(default_factory=list)
    known_bad_triggers: Sequence[str] = field(default_factory=list)
    parent_skill_id: Optional[str] = None
    evidence: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillRegistryEntry:
    """Persisted skill registry entry."""

    skill_id: str
    version: str
    status: str
    description: str
    origin_runs: Sequence[str]
    baseline_successes: Sequence[str]
    treatment_successes: Sequence[str]
    new_successes: Sequence[str]
    regressions: Sequence[str]
    decision_reason: str
    validation_result: Dict[str, Any] = field(default_factory=dict)
    known_failure_modes: Sequence[str] = field(default_factory=list)
    known_bad_triggers: Sequence[str] = field(default_factory=list)
    parent_skill_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    registry_path: Optional[Path] = None


# Backward-compatible aliases while callers migrate.
RunConfig = ExecutorSessionConfig
SkillRecord = SkillRegistryEntry
