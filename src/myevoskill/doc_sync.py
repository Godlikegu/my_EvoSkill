"""Doc-sync policy checks for key modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


class DocSyncChecker:
    """Check that required documentation exists for core module groups."""

    DEFAULT_MAPPING = {
        "compiler": ["task_bundle_spec.md", "first_task_cars_spectroscopy.md"],
        "compile_audit": ["task_bundle_spec.md", "architecture.md"],
        "envs": ["runtime_and_env_cache.md"],
        "executor": ["executor_adapter_spec.md"],
        "task_runtime": [
            "architecture.md",
            "task_bundle_spec.md",
            "runtime_and_env_cache.md",
            "executor_adapter_spec.md",
        ],
        "proxy": ["executor_adapter_spec.md", "judge_and_security.md"],
        "judging": ["judge_and_security.md"],
        "validation": ["dataset_split_and_validation.md"],
        "registry": ["skill_registry_spec.md"],
        "feedback": ["distillation_and_feedback.md"],
        "logging_utils": ["testing_and_logging.md"],
        "model_provider": ["model_provider_spec.md"],
        "dev_environment": ["dev_environment.md"],
        "live_runner": ["architecture.md", "executor_adapter_spec.md"],
        "manifest_bootstrap": ["architecture.md", "task_registration.md"],
        "task_registration": ["architecture.md", "task_registration.md"],
    }

    def __init__(self, doc_root: Path, module_to_docs: Mapping[str, Sequence[str]] | None = None):
        self.doc_root = Path(doc_root)
        self.module_to_docs = dict(module_to_docs or self.DEFAULT_MAPPING)

    def check(self, changed_modules: Iterable[str] | None = None) -> List[str]:
        modules = (
            list(changed_modules)
            if changed_modules is not None
            else list(self.module_to_docs.keys())
        )
        issues: List[str] = []
        for module in modules:
            if module not in self.module_to_docs:
                issues.append(f"missing doc mapping for module: {module}")
                continue
            for doc_name in self.module_to_docs[module]:
                if not (self.doc_root / doc_name).exists():
                    issues.append(f"missing doc '{doc_name}' for module '{module}'")
        return issues
