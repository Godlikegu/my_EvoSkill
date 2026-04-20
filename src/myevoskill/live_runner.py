"""Manifest-driven live task runner for registered tasks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compile_audit import HeuristicCompileAuditAdapter
from .compiler import TaskBundleCompiler
from .executor import ClaudeWorkspaceAdapter
from .judge_runner import invoke_judge_runner
from .logging_utils import RunLogger
from .model_provider import ClaudeSDKAdapter
from .models import ExecutorSessionConfig, JudgeResult, ModelConfig, ProxyFeedback, RunRecord
from .proxy import ProxyVerifier
from .registration_contract import ensure_live_ready_manifest, ensure_manifest_runtime_env
from .task_adapters import manifest_proxy_spec
from .task_runtime import ensure_clean_run_directory, resolve_run_paths

LIVE_RUN_TIMEOUT_SECONDS = 30 * 60
LIVE_RUN_WORKSPACE_COMPLETION_POLICY = "sdk_result_message"
LIVE_RUN_MAX_WORKSPACE_ITERATIONS = 0
LIVE_RUN_CLAUDE_MAX_TURNS = 0


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
    runtime_env = ensure_manifest_runtime_env(
        manifest,
        task_root=task_root,
        output_root=project_root,
    )
    manifest = {**dict(manifest), "runtime_env": dict(runtime_env)}

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
        timeout=LIVE_RUN_TIMEOUT_SECONDS,
    )
    resolved_model_name = ClaudeSDKAdapter(model).resolve_model_name()
    run_id = f"run-live-{int(time.time())}"
    run_paths = resolve_run_paths(project_root, manifest["task_id"], run_id)
    ensure_clean_run_directory(run_paths.workspace_root)
    ensure_clean_run_directory(run_paths.log_root)

    session = ExecutorSessionConfig(
        run_id=run_id,
        env_hash=str(runtime_env.get("env_hash", "") or "env-live"),
        workspace_root=run_paths.workspace_root,
        budget_seconds=LIVE_RUN_TIMEOUT_SECONDS,
        model_config=model,
        provider_extras={
            "repo_root": str(project_root),
            "workspace_prompt_mode": "semantic_only",
            "workspace_completion_policy": LIVE_RUN_WORKSPACE_COMPLETION_POLICY,
            "max_workspace_iterations": LIVE_RUN_MAX_WORKSPACE_ITERATIONS,
            "claude_max_turns": LIVE_RUN_CLAUDE_MAX_TURNS,
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
    runtime_env = dict(manifest.get("runtime_env") or {})
    adapter_path = str(judge_spec.get("adapter_path") or "").strip()
    if not adapter_path:
        raise RuntimeError("manifest judge_spec.adapter_path is required for live task evaluation")
    python_executable = str(runtime_env.get("python_executable", "") or "").strip()
    if not python_executable:
        raise RuntimeError("manifest runtime_env.python_executable is required for live task evaluation")
    judge_path = Path(task_root) / adapter_path
    if not judge_path.exists():
        raise FileNotFoundError(f"judge adapter not found: {judge_path}")
    payload = invoke_judge_runner(
        python_executable,
        mode="evaluate",
        payload={
            "judge_path": str(judge_path.resolve()),
            "task_root": str(Path(task_root).resolve()),
            "manifest": dict(manifest),
            "run_record": {
                "run_id": run_record.run_id,
                "task_id": run_record.task_id,
                "provider": run_record.provider,
                "env_hash": run_record.env_hash,
                "skills_active": list(run_record.skills_active),
                "workspace_root": str(run_record.workspace_root),
                "provider_session_id": run_record.provider_session_id,
                "model_provider": run_record.model_provider,
                "model_name": run_record.model_name,
                "artifacts_uri": run_record.artifacts_uri,
                "transcript_uri": run_record.transcript_uri,
                "stdout": run_record.stdout,
                "stderr": run_record.stderr,
                "runtime_seconds": run_record.runtime_seconds,
                "metadata": dict(run_record.metadata),
            },
        },
    )
    result_payload = dict(payload.get("judge_result") or {})
    return JudgeResult(
        task_id=str(result_payload.get("task_id", run_record.task_id) or run_record.task_id),
        all_metrics_passed=bool(result_payload.get("all_metrics_passed", False)),
        metrics_actual=dict(result_payload.get("metrics_actual") or {}),
        failed_metrics=list(result_payload.get("failed_metrics") or []),
        failure_tags=list(result_payload.get("failure_tags") or []),
    )


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
        "post_run_audit.json",
    ):
        path = Path(record.workspace_root) / extra
        if path.exists():
            logger.append_text_log(run_dir, extra, path.read_text(encoding="utf-8"))
    for round_file in sorted(Path(record.workspace_root).glob("*_round_*")):
        if not round_file.is_file():
            continue
        if round_file.suffix not in {".json", ".txt", ".log"}:
            continue
        logger.append_text_log(run_dir, round_file.name, round_file.read_text(encoding="utf-8"))
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
