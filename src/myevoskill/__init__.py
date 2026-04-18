"""MyEvoSkill core package."""

from .compile_audit import (
    CompileAuditAdapter,
    HeuristicCompileAuditAdapter,
    NullCompileAuditAdapter,
)
from .compiler import TaskBundleCompiler, TaskCompiler
from .datasets import DatasetManifest, FamilySplit
from .doc_sync import DocSyncChecker
from .envs import EnvManager, EnvSpec
from .executor import (
    ClaudeAdapter,
    ClaudeWorkspaceAdapter,
    ExecutorAdapter,
    FallbackAdapter,
    InspectBridgeAdapter,
    LocalRunnerAdapter,
    OpenHandsAdapter,
)
from .feedback import SkillDistiller, SurrogateFeedbackBuilder
from .judging import HiddenJudge, MetricRequirement
from .logging_utils import RunLogger
from .manifest_bootstrap import (
    bootstrap_task,
    load_task_bootstrap_notes,
)
from .registration_contract import load_task_registration_notes
from .model_provider import (
    AnthropicCompatibleAdapter,
    ClaudeSDKAdapter,
    CustomHTTPAdapter,
    ModelProviderAdapter,
    OpenAICompatibleAdapter,
)
from .models import (
    BootstrapResult,
    ContractDraftResult,
    EnvCacheRecord,
    DevEnvironmentInfo,
    EffectiveRuntimePolicy,
    ExecutorSessionConfig,
    JudgeResult,
    ModelConfig,
    ProxyFeedback,
    PublicExposurePolicy,
    RunPaths,
    RunConfig,
    READMEPolicy,
    RunRecord,
    RuntimeEnvironmentPolicy,
    SkillCandidate,
    SkillRegistryEntry,
    SkillRecord,
    TaskRegistrationResult,
    TaskBundle,
    TaskOutcome,
)
from .task_registration import draft_task_contract, register_task
from .proxy import ProxyVerifier
from .registry import SkillRegistry
from .runtime_info import detect_dev_environment
from .task_adapters import (
    cars_spectroscopy_proxy_spec,
    cars_spectroscopy_public_policy,
    evaluate_cars_spectroscopy_run,
    manifest_output_contract_path,
    manifest_primary_output_path,
    manifest_proxy_spec,
)
from .task_runtime import (
    DEFAULT_RUNTIME_LAYOUT,
    DEFAULT_RUNTIME_POLICY,
    coerce_runtime_layout,
    coerce_runtime_policy,
    ensure_clean_run_directory,
    load_task_spec,
    primary_output_relative_path,
    resolve_primary_output_path,
    resolve_run_paths,
    resolve_runtime_policy,
    resolve_runtime_paths,
)
from .validation import TransferValidator, TransferValidationResult

__all__ = [
    "AnthropicCompatibleAdapter",
    "BootstrapResult",
    "ClaudeSDKAdapter",
    "ClaudeAdapter",
    "ClaudeWorkspaceAdapter",
    "ContractDraftResult",
    "CompileAuditAdapter",
    "DatasetManifest",
    "DevEnvironmentInfo",
    "DocSyncChecker",
    "EffectiveRuntimePolicy",
    "EnvCacheRecord",
    "EnvManager",
    "EnvSpec",
    "ExecutorAdapter",
    "ExecutorSessionConfig",
    "FallbackAdapter",
    "FamilySplit",
    "HeuristicCompileAuditAdapter",
    "HiddenJudge",
    "InspectBridgeAdapter",
    "JudgeResult",
    "LocalRunnerAdapter",
    "MetricRequirement",
    "ModelConfig",
    "ModelProviderAdapter",
    "OpenAICompatibleAdapter",
    "OpenHandsAdapter",
    "NullCompileAuditAdapter",
    "ProxyFeedback",
    "ProxyVerifier",
    "PublicExposurePolicy",
    "RunConfig",
    "RunLogger",
    "RunPaths",
    "RunRecord",
    "READMEPolicy",
    "TaskRegistrationResult",
    "register_task",
    "RuntimeEnvironmentPolicy",
    "SkillCandidate",
    "SkillDistiller",
    "SkillRegistryEntry",
    "SkillRecord",
    "SkillRegistry",
    "SurrogateFeedbackBuilder",
    "TaskBundle",
    "TaskBundleCompiler",
    "TaskCompiler",
    "TaskOutcome",
    "TransferValidationResult",
    "TransferValidator",
    "DEFAULT_RUNTIME_LAYOUT",
    "DEFAULT_RUNTIME_POLICY",
    "cars_spectroscopy_proxy_spec",
    "cars_spectroscopy_public_policy",
    "bootstrap_task",
    "draft_task_contract",
    "coerce_runtime_layout",
    "coerce_runtime_policy",
    "detect_dev_environment",
    "ensure_clean_run_directory",
    "evaluate_cars_spectroscopy_run",
    "CustomHTTPAdapter",
    "load_task_spec",
    "load_task_bootstrap_notes",
    "load_task_registration_notes",
    "manifest_output_contract_path",
    "manifest_primary_output_path",
    "manifest_proxy_spec",
    "primary_output_relative_path",
    "resolve_primary_output_path",
    "resolve_run_paths",
    "resolve_runtime_policy",
    "resolve_runtime_paths",
]


def load_registered_manifest(*args, **kwargs):
    from .live_runner import load_registered_manifest as _impl

    return _impl(*args, **kwargs)


def resolve_registered_task_root(*args, **kwargs):
    from .live_runner import resolve_registered_task_root as _impl

    return _impl(*args, **kwargs)


def evaluate_manifest_run(*args, **kwargs):
    from .live_runner import evaluate_manifest_run as _impl

    return _impl(*args, **kwargs)


def run_registered_task_live(*args, **kwargs):
    from .live_runner import run_registered_task_live as _impl

    return _impl(*args, **kwargs)


__all__.extend(
    [
        "evaluate_manifest_run",
        "load_registered_manifest",
        "resolve_registered_task_root",
        "run_registered_task_live",
    ]
)
