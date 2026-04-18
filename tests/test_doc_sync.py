from pathlib import Path
from myevoskill.doc_sync import DocSyncChecker


def test_doc_sync_checker_passes_for_known_modules():
    checker = DocSyncChecker(Path(__file__).resolve().parents[1] / "doc")
    assert (
        checker.check(
            [
                "compiler",
                "compile_audit",
                "envs",
                "executor",
                "task_runtime",
                "registry",
                "model_provider",
                "dev_environment",
                "manifest_bootstrap",
                "task_registration",
            ]
        )
        == []
    )


def test_doc_sync_checker_reports_missing_mapping(tmp_path):
    checker = DocSyncChecker(tmp_path, {"compiler": ["task_bundle_spec.md"]})
    issues = checker.check(["compiler", "unknown_module"])
    assert "missing doc 'task_bundle_spec.md' for module 'compiler'" in issues
    assert "missing doc mapping for module: unknown_module" in issues


def test_doc_sync_checker_flags_missing_executor_doc(tmp_path):
    checker = DocSyncChecker(tmp_path)
    issues = checker.check(["executor"])
    assert "missing doc 'executor_adapter_spec.md' for module 'executor'" in issues


