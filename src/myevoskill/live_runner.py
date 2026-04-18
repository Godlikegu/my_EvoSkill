"""Manifest-driven live task runner for registered tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compile_audit import HeuristicCompileAuditAdapter
from .compiler import TaskBundleCompiler
from .executor import ClaudeWorkspaceAdapter
from .logging_utils import RunLogger
from .model_provider import ClaudeSDKAdapter
from .models import ExecutorSessionConfig, JudgeResult, ModelConfig, ProxyFeedback, RunRecord
from .proxy import ProxyVerifier
from .registration_contract import ensure_live_ready_manifest
from .task_adapters import manifest_proxy_spec
from .task_runtime import ensure_clean_run_directory, resolve_run_paths


def run_registered_task_live(
    task_id: str,
    *,
    project_root: Path,
    skills: Sequence[str] = (),
) -> dict[str, Any]:
    """Compile and run one registered task through the Claude workspace harness."""

    project_root = Path(project_root).resolve()
    manifest = load_registered_manifest(task_id, project_root=project_root)
    task_root = resolve_registered_task_root(manifest, project_root=project_root)
    ensure_live_ready_manifest(manifest, task_root=task_root)

    compiled_root = project_root / "artifacts" / "compiled"
    bundle = TaskBundleCompiler(compiled_root).compile(
        task_root,
        task_id=manifest["task_id"],
        family=manifest["family"],
        public_policy=manifest["public_policy"],
        manifest=manifest,
        audit_adapter=HeuristicCompileAuditAdapter(),
    )

    if not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("CLAUDE_API_KEY")
        or os.environ.get("MYEVOSKILL_CLAUDE_API_KEY")
    ):
        raise RuntimeError(
            "one of ANTHROPIC_API_KEY / CLAUDE_API_KEY / MYEVOSKILL_CLAUDE_API_KEY is required"
        )

    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="",
        api_key_env="ANTHROPIC_API_KEY",
        temperature=0.0,
    )
    resolved_model_name = ClaudeSDKAdapter(model).resolve_model_name()
    run_id = f"run-live-{int(time.time())}"
    run_paths = resolve_run_paths(project_root, manifest["task_id"], run_id)
    ensure_clean_run_directory(run_paths.workspace_root)
    ensure_clean_run_directory(run_paths.log_root)

    session = ExecutorSessionConfig(
        run_id=run_id,
        env_hash="env-live",
        workspace_root=run_paths.workspace_root,
        model_config=model,
        provider_extras={
            "repo_root": str(project_root),
            "workspace_prompt_mode": "semantic_only",
            "workspace_completion_policy": "main_success_output_contract",
        },
    )

    started_at = time.time()
    record = ClaudeWorkspaceAdapter().run(bundle, session, list(skills))
    elapsed = time.time() - started_at
    proxy = ProxyVerifier().evaluate(record, manifest_proxy_spec(record, manifest))
    judge = evaluate_manifest_run(task_root, record, manifest)
    log_dir = write_live_run_logs(
        run_paths.log_root,
        manifest=manifest,
        record=record,
        proxy=proxy,
        judge=judge,
        elapsed=elapsed,
    )
    return {
        "bundle": bundle,
        "elapsed_seconds": elapsed,
        "judge": judge,
        "log_dir": log_dir,
        "manifest": manifest,
        "model_label": resolved_model_name or "<claude default>",
        "proxy": proxy,
        "record": record,
        "run_id": run_id,
        "run_paths": run_paths,
        "task_root": task_root,
    }


def load_registered_manifest(task_id: str, *, project_root: Path) -> dict[str, Any]:
    """Load one registered manifest from registry/tasks."""

    manifest_path = Path(project_root) / "registry" / "tasks" / f"{task_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_registered_task_root(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
) -> Path:
    """Resolve the raw task directory referenced by a registered manifest."""

    source_dir = str(manifest["source_task_dir"])
    candidate = Path(source_dir)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for path in (
        (Path(project_root) / source_dir).resolve(),
        (Path(project_root) / "registry" / "tasks" / source_dir).resolve(),
        (Path(project_root).parent / "tasks" / str(manifest["task_id"])).resolve(),
    ):
        if path.exists():
            return path
    return (Path(project_root) / source_dir).resolve()


def evaluate_manifest_run(
    task_root: Path,
    run_record: RunRecord,
    manifest: Mapping[str, Any],
) -> JudgeResult:
    """Load and call the task-local judge adapter declared in the manifest."""

    judge_spec = dict(manifest.get("judge_spec") or {})
    adapter_path = str(judge_spec.get("adapter_path") or "").strip()
    callable_name = str(judge_spec.get("callable") or "evaluate_run").strip()
    if not adapter_path:
        raise RuntimeError("manifest judge_spec.adapter_path is required for live task evaluation")
    module_path = Path(task_root) / adapter_path
    if not module_path.exists():
        raise FileNotFoundError(f"judge adapter not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        f"myevoskill_live_judge_{manifest['task_id']}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load judge adapter: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate = getattr(module, callable_name, None)
    if not callable(evaluate):
        raise AttributeError(f"judge callable '{callable_name}' not found in {module_path}")
    result = evaluate(Path(task_root), run_record, manifest)
    if not isinstance(result, JudgeResult):
        raise TypeError(f"judge callable returned unexpected type: {type(result)!r}")
    return result


def write_live_run_logs(
    log_dir: Path,
    *,
    manifest: Mapping[str, Any],
    record: RunRecord,
    proxy: ProxyFeedback,
    judge: JudgeResult,
    elapsed: float,
) -> Path:
    """Persist live-run summary, transcript, and structured artifacts."""

    logger = RunLogger(Path(log_dir).parent)
    run_dir = logger.create_run_dir(Path(log_dir).name)
    logger.write_summary(
        run_dir,
        {
            "task_id": manifest["task_id"],
            "provider": record.provider,
            "protocol_status": record.metadata.get("protocol_status"),
            "sdk_completion_source": record.metadata.get("sdk_completion_source"),
            "num_turns": record.metadata.get("num_turns"),
            "message_count": record.metadata.get("message_count"),
            "elapsed_seconds": round(elapsed, 1),
            "all_metrics_passed": judge.all_metrics_passed,
        },
    )
    logger.append_text_log(run_dir, "stdout.log", record.stdout)
    logger.append_text_log(run_dir, "stderr.log", record.stderr)
    logger.write_json_artifact(run_dir, "proxy_feedback.json", proxy)
    logger.write_json_artifact(run_dir, "judge_result.json", judge)
    transcript = Path(record.transcript_uri)
    if transcript.exists():
        logger.append_text_log(run_dir, "transcript.txt", transcript.read_text(encoding="utf-8"))
    workspace_main = Path(record.workspace_root) / "work" / "main.py"
    if workspace_main.exists():
        logger.append_text_log(run_dir, "work_main.py", workspace_main.read_text(encoding="utf-8"))
    for extra in (
        "trajectory_normalized.json",
        "trajectory_summary.json",
        "vendor_session_ref.json",
        "agent_prompt_round_1.txt",
        "claude_sdk_error_round_1.txt",
        "claude_sdk_diagnostics_round_1.json",
        "files_written_round_1.json",
        "post_run_audit.json",
        "public_self_eval_round_1.json",
    ):
        path = Path(record.workspace_root) / extra
        if path.exists():
            logger.append_text_log(run_dir, extra, path.read_text(encoding="utf-8"))
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the manifest-driven live runner."""

    parser = argparse.ArgumentParser(
        description="Run one registered MyEvoSkill task through the Claude workspace harness."
    )
    parser.add_argument("--task-id", required=True, help="Registered task id in registry/tasks/")
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Optional skill ids passed to the executor; can be specified multiple times.",
    )
    parser.add_argument(
        "--project-root",
        default="",
        help="Project root containing registry/tasks (defaults to cwd).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    result = run_registered_task_live(
        args.task_id,
        project_root=project_root,
        skills=list(args.skill),
    )

    manifest = result["manifest"]
    proxy = result["proxy"]
    judge = result["judge"]
    record = result["record"]
    print(f"[1/5] Task source: {result['task_root']}")
    print(f"[2/5] Bundle compiled: {result['bundle'].root_dir}")
    print(
        f"[3/5] Running Claude SDK agent "
        f"(task={manifest['task_id']}, model={result['model_label']}, run_id={result['run_id']})..."
    )
    print(f"       Done in {result['elapsed_seconds']:.1f}s")
    print(f"       protocol_status={record.metadata.get('protocol_status')}")
    print(f"       sdk_completion_source={record.metadata.get('sdk_completion_source')}")
    print(f"       num_turns={record.metadata.get('num_turns')}")
    print(f"       message_count={record.metadata.get('message_count')}")
    print(f"[4/5] Proxy: output_exists={proxy.output_exists}, warnings={proxy.warnings}")
    print(f"[5/5] Judge: all_passed={judge.all_metrics_passed}")
    if judge.metrics_actual:
        for key, value in judge.metrics_actual.items():
            print(f"       {key} = {value}")
    if judge.failed_metrics:
        print(f"       failed_metrics = {judge.failed_metrics}")
    if judge.failure_tags:
        print(f"       failure_tags = {judge.failure_tags}")
    print(f"\nLogs saved to: {result['log_dir']}")
    print(f"Workspace: {result['run_paths'].workspace_root}")
    return 0


__all__ = [
    "evaluate_manifest_run",
    "load_registered_manifest",
    "main",
    "resolve_registered_task_root",
    "run_registered_task_live",
    "write_live_run_logs",
]
