"""Provider-neutral executor adapter interfaces and minimal implementations."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import numpy as np
import os
import re
import shlex
import socket
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Dict, Mapping, Optional, Sequence

from .envs import python_executable_path_entries
from .judge_runner import invoke_judge_runner
from .model_provider import (
    ClaudeSDKAdapter,
    CustomHTTPAdapter,
    OpenAICompatibleAdapter,
)
from .models import (
    EffectiveRuntimePolicy,
    ExecutorSessionConfig,
    JudgeResult,
    RunRecord,
    TaskBundle,
)
from .task_contract import (
    load_public_task_contract_from_root,
    output_field_map,
    output_metric_input_checks,
    output_requirements_from_contract,
    resolve_metric_input_value,
    task_contract_execution,
    validate_output_payload_against_contract,
)
from .task_runtime import (
    coerce_runtime_layout,
    ensure_clean_run_directory,
    load_task_spec,
    resolve_run_paths,
    resolve_runtime_policy,
    resolve_runtime_paths,
)


class ModelResponseParseError(RuntimeError):
    """Raised when a model response cannot be converted into executable code."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str = "",
        parse_error: str = "",
        candidate_count: int = 0,
        selected_source: str = "",
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.parse_error = parse_error
        self.candidate_count = candidate_count
        self.selected_source = selected_source


class ClaudeSDKExecutionError(RuntimeError):
    """Raised when the Claude SDK agent exits without a usable structured summary."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "sdk_error",
        sdk_messages: Optional[list[Any]] = None,
        result_text: str = "",
        diagnostics: Optional[dict[str, Any]] = None,
        private_state: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.sdk_messages = list(sdk_messages or [])
        self.result_text = result_text
        self.diagnostics = dict(diagnostics or {})
        self.private_state = dict(private_state or {})


class ExecutorAdapter(ABC):
    """Provider-neutral executor interface."""

    provider_name = "abstract"

    @abstractmethod
    def run(
        self,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        active_skills: Sequence[str],
    ) -> RunRecord:
        """Execute a task using only the public bundle."""


def _make_sdk_hook_matcher(*, hooks: Sequence[Any], matcher: Optional[str] = None) -> Any:
    """Build a Claude SDK HookMatcher when available, else a lightweight test fallback."""

    hook_list = list(hooks)
    try:
        from claude_agent_sdk.types import HookMatcher
    except ModuleNotFoundError:
        return SimpleNamespace(hooks=hook_list, matcher=matcher)
    if matcher is None:
        return HookMatcher(hooks=hook_list)
    return HookMatcher(matcher=matcher, hooks=hook_list)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _prepare_runtime_workspace(
    task_bundle: TaskBundle,
    workspace_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Stage a stable runtime root for task execution."""

    task_spec = load_task_spec(task_bundle)
    runtime_layout = coerce_runtime_layout(task_spec.get("runtime_layout"))
    paths = resolve_runtime_paths(workspace_root, runtime_layout)
    runtime_root = paths["runtime_root"]

    ensure_clean_run_directory(runtime_root)

    for child in task_bundle.public_bundle_dir.iterdir():
        target = runtime_root / child.name
        _copy_path(child, target)

    public_bundle_mirror = paths["public_bundle_dir"]
    if public_bundle_mirror.exists():
        _remove_path(public_bundle_mirror)
    shutil.copytree(task_bundle.public_bundle_dir, public_bundle_mirror)

    for key in ("data_dir", "work_dir", "output_dir", "checkpoints_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    return task_spec, paths


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_workspace_root(
    task_bundle: TaskBundle,
    session_config: ExecutorSessionConfig,
) -> Path:
    if session_config.workspace_root is not None:
        return Path(session_config.workspace_root)
    repo_root = Path(session_config.provider_extras.get("repo_root", _default_repo_root()))
    return resolve_run_paths(repo_root, task_bundle.task_id, session_config.run_id).workspace_root


def _timeout_metadata(
    policy: EffectiveRuntimePolicy,
    *,
    timed_out: bool = False,
    timeout_scope: str = "",
) -> Dict[str, Any]:
    return {
        "timed_out": timed_out,
        "timeout_scope": timeout_scope,
        "effective_model_timeout_seconds": policy.model_timeout_seconds,
        "effective_execution_budget_seconds": policy.execution_budget_seconds,
    }


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (TimeoutError, socket.timeout))
    return False


def _task_runtime_env_spec(task_bundle: TaskBundle) -> dict[str, Any]:
    task_spec = load_task_spec(task_bundle)
    runtime_env = task_spec.get("runtime_env")
    return dict(runtime_env or {}) if isinstance(runtime_env, Mapping) else {}


def _task_python_executable(task_bundle: TaskBundle) -> str:
    runtime_env = _task_runtime_env_spec(task_bundle)
    value = str(runtime_env.get("python_executable", "") or "").strip()
    return value or sys.executable


def _venv_root_from_python_executable(python_executable: Path) -> Path:
    executable = Path(python_executable).resolve()
    parent = executable.parent
    if parent.name.lower() in {"scripts", "bin"}:
        return parent.parent
    return parent


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    timeout_seconds: int,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    normalized_command = list(command)
    if normalized_command and normalized_command[0] in {"python", "python3"}:
        normalized_command[0] = str(env.get("MYEVOSKILL_TASK_PYTHON", "") or sys.executable)
    process = subprocess.Popen(
        normalized_command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return (
            subprocess.CompletedProcess(normalized_command, process.returncode, stdout, stderr),
            False,
        )
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        timeout_message = (
            f"\nMyEvoSkill execution timed out after {timeout_seconds} seconds and was terminated.\n"
        )
        return (
            subprocess.CompletedProcess(
                normalized_command,
                -9,
                stdout or "",
                (stderr or "") + timeout_message,
            ),
            True,
        )


def _runtime_environment(
    session_config: ExecutorSessionConfig,
    task_bundle: TaskBundle,
    runtime_paths: dict[str, Path],
) -> dict[str, str]:
    """Build common environment variables for staged runtime execution."""

    task_spec = load_task_spec(task_bundle)
    runtime_env = dict(task_spec.get("runtime_env") or {})
    task_python = str(runtime_env.get("python_executable", "") or sys.executable).strip() or sys.executable
    env = {
        **os.environ,
        **dict(session_config.env),
        "MYEVOSKILL_RUNTIME_ROOT": str(runtime_paths["runtime_root"]),
        "MYEVOSKILL_PUBLIC_BUNDLE": str(runtime_paths["public_bundle_dir"]),
        "MYEVOSKILL_WORK_DIR": str(runtime_paths["work_dir"]),
        "MYEVOSKILL_OUTPUT_DIR": str(runtime_paths["output_dir"]),
        "MYEVOSKILL_CHECKPOINT_DIR": str(runtime_paths["checkpoints_dir"]),
        "MYEVOSKILL_WORKSPACE": str(runtime_paths["runtime_root"]),
        "MYEVOSKILL_TASK_ID": task_bundle.task_id,
        "MYEVOSKILL_TASK_PYTHON": task_python,
        "MYEVOSKILL_TASK_ENV_HASH": str(runtime_env.get("env_hash", "") or session_config.env_hash),
        "MYEVOSKILL_TASK_ENV_BACKEND": str(runtime_env.get("backend", "") or ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    task_python_path = Path(task_python)
    if task_python_path.exists():
        task_path_entries = python_executable_path_entries(task_python_path)
        existing_path = str(env.get("PATH", "") or "")
        merged_entries: list[str] = []
        seen: set[str] = set()
        for item in [*task_path_entries, *[segment for segment in existing_path.split(os.pathsep) if segment]]:
            normalized = os.path.normcase(os.path.normpath(item))
            if normalized in seen:
                continue
            seen.add(normalized)
            merged_entries.append(item)
        env["PATH"] = os.pathsep.join(merged_entries)
        env["VIRTUAL_ENV"] = str(_venv_root_from_python_executable(task_python_path))
    return env


def _set_path_mode(path: Path, dir_mode: int, file_mode: int) -> None:
    if not path.exists():
        return
    if path.is_file():
        os.chmod(path, file_mode)
        return
    for child in path.rglob("*"):
        if child.is_dir():
            os.chmod(child, dir_mode)
        else:
            os.chmod(child, file_mode)
    os.chmod(path, dir_mode)


def _configure_workspace_permissions(
    runtime_paths: dict[str, Path],
    *,
    lock_runtime_root: bool,
) -> None:
    runtime_root = runtime_paths["runtime_root"]
    os.chmod(runtime_root, 0o555 if lock_runtime_root else 0o755)

    for public_dir_key in ("data_dir", "public_bundle_dir"):
        _set_path_mode(runtime_paths[public_dir_key], 0o555, 0o444)

    evaluation_dir = runtime_root / "evaluation"
    _set_path_mode(evaluation_dir, 0o555, 0o444)

    for public_name in ("README_public.md", "requirements.txt"):
        public_path = runtime_root / public_name
        if public_path.exists():
            os.chmod(public_path, 0o444)

    for writable_key in ("work_dir", "output_dir", "checkpoints_dir"):
        writable_path = runtime_paths[writable_key]
        writable_path.mkdir(parents=True, exist_ok=True)
        _set_path_mode(writable_path, 0o755, 0o644)


class LocalRunnerAdapter(ExecutorAdapter):
    """Real local subprocess runner for the public bundle workspace."""

    provider_name = "local_runner"

    def run(
        self,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        active_skills: Sequence[str],
    ) -> RunRecord:
        task_spec, runtime_paths = _prepare_runtime_workspace(
            task_bundle, _resolve_workspace_root(task_bundle, session_config)
        )
        policy = resolve_runtime_policy(
            task_spec=task_spec,
            session_config=session_config,
            model_config=session_config.model_config,
        )
        workspace = runtime_paths["runtime_root"]
        start = time.time()
        command = list(session_config.command) or ["python3", "-c", "print('local runner placeholder')"]
        completed, timed_out = _run_subprocess(
            command,
            cwd=workspace,
            env=_runtime_environment(session_config, task_bundle, runtime_paths),
            timeout_seconds=policy.execution_budget_seconds,
        )
        runtime = time.time() - start
        transcript_path = workspace / "transcript.txt"
        transcript_path.write_text(
            "\n".join(
                [
                    f"provider={self.provider_name}",
                    f"command={' '.join(command)}",
                    f"skills={','.join(active_skills)}",
                    f"runtime_layout={json.dumps(task_spec.get('runtime_layout') or {}, sort_keys=True)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return RunRecord(
            run_id=session_config.run_id,
            task_id=task_bundle.task_id,
            provider=self.provider_name,
            provider_session_id=session_config.run_id,
            model_provider=(
                session_config.model_config.provider_name
                if session_config.model_config
                else ""
            ),
            model_name=(session_config.model_config.model_name if session_config.model_config else ""),
            env_hash=session_config.env_hash,
            skills_active=list(active_skills),
            workspace_root=workspace,
            artifacts_uri=str(runtime_paths["output_dir"]),
            transcript_uri=str(transcript_path),
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_seconds=runtime,
            metadata={
                "execution_mode": session_config.execution_mode,
                "returncode": completed.returncode,
                "command": command,
                "runtime_layout": task_spec.get("runtime_layout") or {},
                **_timeout_metadata(
                    policy,
                    timed_out=timed_out,
                    timeout_scope="solver_execution" if timed_out else "",
                ),
            },
        )


class FallbackAdapter(ExecutorAdapter):
    """Deterministic adapter for tests and offline use."""

    provider_name = "fallback"

    def run(
        self,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        active_skills: Sequence[str],
    ) -> RunRecord:
        task_spec, runtime_paths = _prepare_runtime_workspace(
            task_bundle, _resolve_workspace_root(task_bundle, session_config)
        )
        policy = resolve_runtime_policy(
            task_spec=task_spec,
            session_config=session_config,
            model_config=session_config.model_config,
        )
        public_readme = task_bundle.readme_public_path
        summary = ""
        if public_readme and public_readme.exists():
            summary = public_readme.read_text(encoding="utf-8")[:160]
        transcript_path = runtime_paths["runtime_root"] / "transcript.txt"
        transcript_path.write_text(
            "\n".join(
                [
                    f"provider={self.provider_name}",
                    f"skills={','.join(active_skills)}",
                    f"runtime_layout={json.dumps(task_spec.get('runtime_layout') or {}, sort_keys=True)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return RunRecord(
            run_id=session_config.run_id,
            task_id=task_bundle.task_id,
            provider=self.provider_name,
            provider_session_id=session_config.run_id,
            model_provider=(
                session_config.model_config.provider_name
                if session_config.model_config
                else ""
            ),
            model_name=(session_config.model_config.model_name if session_config.model_config else ""),
            env_hash=session_config.env_hash,
            skills_active=list(active_skills),
            workspace_root=runtime_paths["runtime_root"],
            artifacts_uri=str(runtime_paths["output_dir"]),
            transcript_uri=str(transcript_path),
            stdout=f"FallbackAdapter read public bundle for task {task_bundle.task_id}\n{summary}",
            stderr="",
            metadata={
                "execution_mode": session_config.execution_mode,
                "runtime_layout": task_spec.get("runtime_layout") or {},
                **_timeout_metadata(policy),
            },
        )


class OpenHandsAdapter(FallbackAdapter):
    """Placeholder OpenHands adapter for environments where OpenHands is available."""

    provider_name = "openhands"


class InspectBridgeAdapter(FallbackAdapter):
    """Inspect AI bridge adapter with explicit dependency detection."""

    provider_name = "inspect_bridge"

    def __init__(self, inspect_available: Optional[bool] = None):
        self._inspect_available_override = inspect_available

    def _inspect_is_available(self) -> bool:
        if self._inspect_available_override is not None:
            return self._inspect_available_override
        return importlib.util.find_spec("inspect_ai") is not None

    def run(
        self,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        active_skills: Sequence[str],
    ) -> RunRecord:
        if not self._inspect_is_available():
            if session_config.provider_extras.get("allow_fallback", True):
                fallback_record = FallbackAdapter().run(
                    task_bundle, session_config, active_skills
                )
                return RunRecord(
                    **{
                        **fallback_record.__dict__,
                        "provider": "inspect_unavailable_fallback",
                        "metadata": {
                            **fallback_record.metadata,
                            "fallback_reason": "inspect_ai_missing",
                            "requested_provider": self.provider_name,
                        },
                    }
                )
            raise RuntimeError("inspect_ai is unavailable for InspectBridgeAdapter")
        if not session_config.model_config:
            record = super().run(task_bundle, session_config, active_skills)
            return RunRecord(
                **{
                    **record.__dict__,
                    "metadata": {
                        **record.metadata,
                        "bridge_mode": "placeholder",
                    },
                }
            )
        record = self._run_with_model(task_bundle, session_config, active_skills)
        return RunRecord(
            **{
                **record.__dict__,
                "metadata": {
                    **record.metadata,
                    "bridge_mode": record.metadata.get("bridge_mode", "single_shot_agent"),
                },
            }
        )

    def _run_with_model(
        self,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        active_skills: Sequence[str],
    ) -> RunRecord:
        provider_adapter = self._provider_adapter(session_config)
        safe_model_config = provider_adapter.safe_log_config()
        resolved_model_name = self._resolved_model_name(session_config, provider_adapter)
        task_spec, runtime_paths = _prepare_runtime_workspace(
            task_bundle, _resolve_workspace_root(task_bundle, session_config)
        )
        policy = resolve_runtime_policy(
            task_spec=task_spec,
            session_config=session_config,
            model_config=session_config.model_config,
        )
        workspace = runtime_paths["runtime_root"]
        work_dir = runtime_paths["work_dir"]
        output_dir = runtime_paths["output_dir"]
        checkpoints_dir = runtime_paths["checkpoints_dir"]
        transcript_path = workspace / "transcript.txt"

        prompt = self._build_agent_prompt(task_bundle, active_skills)
        transcript_path.write_text(
            "\n".join(
                [
                    f"provider={self.provider_name}",
                    f"task_id={task_bundle.task_id}",
                    f"model_provider={session_config.model_config.provider_name}",
                    f"model_name={resolved_model_name}",
                    f"api_key_env={session_config.model_config.api_key_env}",
                    f"skills={','.join(active_skills)}",
                    f"runtime_layout={json.dumps(task_spec.get('runtime_layout') or {}, sort_keys=True)}",
                    f"effective_model_timeout_seconds={policy.model_timeout_seconds}",
                    f"effective_execution_budget_seconds={policy.execution_budget_seconds}",
                    "",
                    "PROMPT:",
                    prompt,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            generated_code, model_metadata = self._generate_agent_code(
                provider_adapter=provider_adapter,
                session_config=session_config,
                prompt=prompt,
                model_timeout_seconds=policy.model_timeout_seconds,
            )
        except ModelResponseParseError as exc:
            raw_response_path = workspace / "raw_response.txt"
            raw_response_path.write_text(exc.raw_response, encoding="utf-8")
            parse_error_path = workspace / "response_parse_error.txt"
            parse_error_lines = [
                f"message={exc}",
                f"candidate_count={exc.candidate_count}",
                f"selected_source={exc.selected_source}",
            ]
            if exc.parse_error:
                parse_error_lines.append(f"parse_error={exc.parse_error}")
            parse_error_path.write_text("\n".join(parse_error_lines) + "\n", encoding="utf-8")
            raise
        except Exception as exc:
            if not _is_timeout_error(exc):
                raise
            runtime = 0.0
            return RunRecord(
                run_id=session_config.run_id,
                task_id=task_bundle.task_id,
                provider=self.provider_name,
                provider_session_id=session_config.run_id,
                model_provider=session_config.model_config.provider_name,
                model_name=resolved_model_name,
                env_hash=session_config.env_hash,
                skills_active=list(active_skills),
                workspace_root=workspace,
                artifacts_uri=str(output_dir),
                transcript_uri=str(transcript_path),
                stdout="",
                stderr=(
                    f"MyEvoSkill model request timed out after "
                    f"{policy.model_timeout_seconds} seconds.\n"
                ),
                runtime_seconds=runtime,
                metadata={
                    "execution_mode": session_config.execution_mode,
                    "returncode": -1,
                    "command": [],
                    "model_provider_kind": safe_model_config.get("kind", ""),
                    "api_key_env": session_config.model_config.api_key_env,
                    "runtime_layout": task_spec.get("runtime_layout") or {},
                    "runtime_root": str(workspace),
                    "work_dir": str(work_dir),
                    "output_dir": str(output_dir),
                    "checkpoints_dir": str(checkpoints_dir),
                    "prompt_contract_version": "v2_structured_json",
                    "command_history_summary": ["generate_code"],
                    "bridge_mode": "single_shot_openai_compatible",
                    **_timeout_metadata(
                        policy,
                        timed_out=True,
                        timeout_scope="model_request",
                    ),
                },
            )

        agent_script_path = work_dir / "agent_solution.py"
        agent_script_path.write_text(generated_code, encoding="utf-8")
        env = _runtime_environment(session_config, task_bundle, runtime_paths)
        start = time.time()
        completed, timed_out = _run_subprocess(
            ["python3", str(agent_script_path)],
            cwd=workspace,
            env=env,
            timeout_seconds=policy.execution_budget_seconds,
        )
        runtime = time.time() - start
        return RunRecord(
            run_id=session_config.run_id,
            task_id=task_bundle.task_id,
            provider=self.provider_name,
            provider_session_id=session_config.run_id,
            model_provider=session_config.model_config.provider_name,
            model_name=resolved_model_name,
            env_hash=session_config.env_hash,
            skills_active=list(active_skills),
            workspace_root=workspace,
            artifacts_uri=str(output_dir),
            transcript_uri=str(transcript_path),
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_seconds=runtime,
            metadata={
                "execution_mode": session_config.execution_mode,
                "returncode": completed.returncode,
                "command": ["python3", str(agent_script_path)],
                "model_provider_kind": safe_model_config.get("kind", ""),
                "api_key_env": session_config.model_config.api_key_env,
                "runtime_layout": task_spec.get("runtime_layout") or {},
                "runtime_root": str(workspace),
                "work_dir": str(work_dir),
                "output_dir": str(output_dir),
                "checkpoints_dir": str(checkpoints_dir),
                **_timeout_metadata(
                    policy,
                    timed_out=timed_out,
                    timeout_scope="solver_execution" if timed_out else "",
                ),
                "prompt_contract_version": model_metadata.get(
                    "prompt_contract_version", "v2_structured_json"
                ),
                "command_history_summary": ["generate_code", "run_generated_code"],
                **model_metadata,
            },
        )

    def _provider_adapter(self, session_config: ExecutorSessionConfig):
        model_config = session_config.model_config
        if model_config is None:
            raise RuntimeError("InspectBridgeAdapter requires model_config for model-backed runs")
        provider_name = model_config.provider_name.lower()
        if provider_name in {"claude-sdk", "claude_sdk"}:
            return ClaudeSDKAdapter(model_config)
        if provider_name in {"openai", "openai-compatible", "openai_compatible"}:
            return OpenAICompatibleAdapter(model_config)
        if provider_name in {"custom_http", "custom-http"}:
            return CustomHTTPAdapter(model_config)
        raise RuntimeError(
            f"InspectBridgeAdapter does not support provider '{model_config.provider_name}'"
        )

    def _resolved_model_name(
        self,
        session_config: ExecutorSessionConfig,
        provider_adapter: Optional[Any] = None,
    ) -> str:
        model_config = session_config.model_config
        if model_config is None:
            return ""
        explicit = str(model_config.model_name or "").strip()
        if explicit:
            return explicit
        provider_adapter = provider_adapter or self._provider_adapter(session_config)
        resolve_model_name = getattr(provider_adapter, "resolve_model_name", None)
        if callable(resolve_model_name):
            return str(resolve_model_name() or "").strip()
        return ""

    def _build_agent_prompt(
        self,
        task_bundle: TaskBundle,
        active_skills: Sequence[str],
    ) -> str:
        task_spec = load_task_spec(task_bundle)
        runtime_layout = coerce_runtime_layout(task_spec.get("runtime_layout"))
        readme = ""
        if task_bundle.readme_public_path and task_bundle.readme_public_path.exists():
            readme = task_bundle.readme_public_path.read_text(encoding="utf-8")
        compile_report = {}
        if task_bundle.compile_report_path.exists():
            compile_report = json.loads(
                task_bundle.compile_report_path.read_text(encoding="utf-8")
            )
        final_contract = compile_report.get("final_public_contract", {})
        listed_files = sorted(
            path.relative_to(task_bundle.public_bundle_dir).as_posix()
            for path in task_bundle.public_bundle_dir.rglob("*")
            if path.is_file()
        )
        return "\n".join(
            [
                "You are solving a scientific coding task.",
                "Use only staged files from the compiled public bundle.",
                "The current working directory is the runtime root for this run.",
                f"Read staged inputs from relative paths under {runtime_layout['data_dir']}/ when possible.",
                f"Write final outputs under {runtime_layout['output_dir']}/.",
                f"Use {runtime_layout['checkpoints_dir']}/ for intermediate checkpoints if needed.",
                "A read-only public bundle mirror is available via MYEVOSKILL_PUBLIC_BUNDLE.",
                "Do not assume access to hidden assets, ground truth, washed code, or notebooks.",
                "Preferred response format is a single JSON object.",
                'The JSON object should contain keys: "python_code", "declared_outputs", "assumptions", "solver_summary".',
                '"python_code" is required and must contain executable Python only.',
                '"declared_outputs" should list output file paths or output descriptions.',
                '"assumptions" should list key assumptions made by the solver.',
                '"solver_summary" should briefly describe the chosen method.',
                f"Active skills: {', '.join(active_skills) if active_skills else '(none)'}",
                f"Public files: {', '.join(listed_files)}",
                f"Public contract: {json.dumps(final_contract, sort_keys=True)}",
                "",
                "README:",
                readme,
                "",
                "If you cannot follow the JSON format, return only executable Python code that writes the required outputs.",
            ]
        )

    def _generate_agent_code(
        self,
        provider_adapter,
        session_config: ExecutorSessionConfig,
        prompt: str,
        model_timeout_seconds: int,
    ) -> tuple[str, dict[str, object]]:
        if "mock_llm_response" in session_config.provider_extras:
            provider_adapter.resolve_api_key()
            raw_response = str(session_config.provider_extras["mock_llm_response"])
            return self._parse_model_response(
                raw_content=raw_response,
                bridge_mode="single_shot_mock_llm",
                model_provider_kind=provider_adapter.safe_log_config().get("kind", ""),
            )
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            },
            {"role": "user", "content": prompt},
        ]
        url, headers, payload = provider_adapter.build_request(messages)
        timeout = session_config.model_config.timeout or 60
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=model_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        raw_content = self._extract_model_content(body)
        return self._parse_model_response(
            raw_content=raw_content,
            bridge_mode="single_shot_openai_compatible",
            model_provider_kind=provider_adapter.safe_log_config().get("kind", ""),
        )

    def _extract_model_content(self, payload: dict[str, object]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("model response did not contain any choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(parts)
        raise RuntimeError("model response content format is unsupported")

    def _sanitize_generated_code(self, raw_content: str) -> str:
        text = raw_content.strip()
        if not text:
            raise RuntimeError("model response did not contain executable Python code")

        candidates: list[str] = []
        fence_pattern = re.compile(
            r"```(?P<lang>[A-Za-z0-9_-]+)?\s*\n?(?P<body>.*?)```",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in fence_pattern.finditer(text):
            body = (match.group("body") or "").strip()
            if body:
                candidates.append(body)
        candidates.append(text)
        for candidate in candidates:
            cleaned = candidate.strip()
            if not cleaned:
                continue
            normalized = self._normalize_python_candidate(cleaned)
            if normalized is not None:
                return normalized + ("\n" if not normalized.endswith("\n") else "")
        raise RuntimeError("model response did not contain executable Python code")

    def _normalize_python_candidate(self, content: str) -> Optional[str]:
        try:
            compile(content, "<agent_generated>", "exec")
            return content
        except SyntaxError:
            pass

        lines = content.splitlines()
        start = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(
                ("import ", "from ", "def ", "class ", "if ", "for ", "while ", "try:", "with ")
            ) or "=" in stripped:
                start = index
                break
        if start is None:
            return None
        candidate = "\n".join(lines[start:]).strip()
        if not candidate:
            return None
        try:
            compile(candidate, "<agent_generated_trimmed>", "exec")
            return candidate
        except SyntaxError:
            return None

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "You are a coding agent for scientific tasks.",
                "Prefer returning a single JSON object with exactly these top-level keys:",
                '- "python_code": required executable Python string',
                '- "declared_outputs": optional list describing outputs written by the code',
                '- "assumptions": optional list of assumptions',
                '- "solver_summary": optional short text summary',
                "Do not include markdown fences around the JSON.",
                "Do not include prose outside the JSON object.",
                "If strict JSON is impossible, return only executable Python code with no markdown fences.",
            ]
        )

    def _parse_model_response(
        self,
        raw_content: str,
        bridge_mode: str,
        model_provider_kind: str,
    ) -> tuple[str, dict[str, object]]:
        structured_payload, parse_error, candidate_count, selected_source = (
            self._extract_structured_payload(raw_content)
        )
        if structured_payload is not None:
            python_code = structured_payload.get("python_code")
            if not isinstance(python_code, str) or not python_code.strip():
                raise ModelResponseParseError(
                    "structured model response missing required field 'python_code'",
                    raw_response=raw_content,
                    parse_error="structured_json_missing_python_code",
                    candidate_count=candidate_count,
                    selected_source=selected_source,
                )
            cleaned_code = self._sanitize_generated_code(python_code)
            declared_outputs = self._coerce_list_field(
                structured_payload.get("declared_outputs"), default=[]
            )
            assumptions = self._coerce_list_field(
                structured_payload.get("assumptions"), default=[]
            )
            solver_summary = structured_payload.get("solver_summary", "")
            if solver_summary is None:
                solver_summary = ""
            return (
                cleaned_code,
                {
                    "bridge_mode": bridge_mode,
                    "model_provider_kind": model_provider_kind,
                    "raw_response_preview": self._preview_text(raw_content),
                    "raw_response_text": raw_content,
                    "response_format": "structured_json",
                    "response_candidate_count": candidate_count,
                    "response_selected_source": selected_source,
                    "parsed_response": {
                        "declared_outputs": declared_outputs,
                        "assumptions": assumptions,
                        "solver_summary": str(solver_summary),
                    },
                    "prompt_contract_version": "v2_structured_json",
                },
            )

        try:
            cleaned_code = self._sanitize_generated_code(raw_content)
        except RuntimeError as exc:
            raise ModelResponseParseError(
                str(exc),
                raw_response=raw_content,
                parse_error=parse_error or "plain_text_code_parse_failed",
                candidate_count=candidate_count,
                selected_source=selected_source or "plain_text_fallback",
            ) from exc
        metadata: Dict[str, object] = {
            "bridge_mode": bridge_mode,
            "model_provider_kind": model_provider_kind,
            "sdk_error_type": "",
            "raw_response_preview": self._preview_text(raw_content),
            "raw_response_text": raw_content,
            "response_format": "plain_text_code",
            "response_candidate_count": candidate_count,
            "response_selected_source": "plain_text_fallback",
            "parsed_response": {
                "declared_outputs": [],
                "assumptions": [],
                "solver_summary": "",
            },
            "prompt_contract_version": "v2_structured_json",
        }
        if parse_error:
            metadata["response_parse_error"] = parse_error
        return cleaned_code, metadata

    def _extract_structured_payload(
        self, raw_content: str
    ) -> tuple[Optional[dict[str, Any]], Optional[str], int, str]:
        text = raw_content.strip()
        if not text:
            return None, "empty_response", 0, ""

        candidates = self._extract_json_candidates(text)
        last_error: Optional[str] = None
        for source, candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = str(exc)
                continue
            if isinstance(parsed, dict):
                return parsed, None, len(candidates), source
            last_error = "structured response was not a JSON object"
        return None, last_error or "response_not_structured_json", len(candidates), ""

    def _extract_json_candidates(self, raw_content: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add_candidate(source: str, candidate: str) -> None:
            cleaned = candidate.strip()
            key = (source, cleaned)
            if not cleaned or key in seen:
                return
            seen.add(key)
            candidates.append((source, cleaned))

        add_candidate("full_text", raw_content)

        fence_pattern = re.compile(r"```(?P<lang>[A-Za-z0-9_-]+)?\s*\n?(?P<body>.*?)```", re.DOTALL)
        for match in fence_pattern.finditer(raw_content):
            lang = (match.group("lang") or "").strip().lower()
            body = match.group("body") or ""
            if lang == "json":
                add_candidate("fenced_json_block", body)
            elif not lang:
                add_candidate("fenced_untyped_block", body)

        balanced_slice = self._extract_balanced_json_slice(raw_content)
        if balanced_slice is not None:
            add_candidate("balanced_json_slice", balanced_slice)
        return candidates

    def _extract_balanced_json_slice(self, raw_content: str) -> Optional[str]:
        start = raw_content.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(raw_content)):
            char = raw_content[index]
            if in_string:
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    return raw_content[start : index + 1]
        return None

    def _coerce_list_field(self, value: Any, default: list[Any]) -> list[Any]:
        if value is None:
            return list(default)
        if isinstance(value, list):
            return value
        return [value]

    def _preview_text(self, content: str, limit: int = 240) -> str:
        collapsed = re.sub(r"\s+", " ", content).strip()
        return collapsed[:limit]


class ClaudeWorkspaceAdapter(InspectBridgeAdapter):
    """Claude SDK constrained workspace agent and formal model-backed executor."""

    provider_name = "claude_workspace"
    WORKSPACE_PROMPT_VERSION = "v7_claude_sdk_public_self_eval"
    LEGACY_WORKSPACE_PROMPT_VERSION = "v3_workspace_json"
    DEFAULT_MAX_ITERATIONS = 3
    DEFAULT_STOP_ORACLE = "public_self_eval"
    DEFAULT_WORKSPACE_PROMPT_MODE = "inline_public_content"
    DEFAULT_COMPLETION_POLICY = "sdk_result_message"
    DEFAULT_ALLOWED_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep"]
    NETWORK_ENABLED_ALLOWED_TOOLS = ["WebFetch", "WebSearch"]
    DEFAULT_DISALLOWED_TOOLS = ["WebFetch", "WebSearch", "TodoWrite"]
    SUBMISSION_TOOL_NAMES = ["check_ready", "submit_result"]
    HIDDEN_JUDGE_TOOL_NAMES = ["submit_result"]
    SUBMISSION_SERVER_NAME = "myevoskill_harness"
    HIDDEN_JUDGE_SERVER_NAME = "myevoskill_hidden_judge"
    DEFAULT_BASH_ALLOWED_PREFIXES = [
        "python",
        "python3",
        "py",
        "ls",
        "find",
        "cat",
        "head",
        "sed",
        "grep",
        "pytest",
        "mkdir",
        "pwd",
        "dir",
        "echo",
        "cd",
    ]
    DEFAULT_BASH_DENIED_TOKENS = [
        "curl",
        "wget",
        "ssh",
        "scp",
        "git ",
        "pip install",
        "conda ",
    ]
    WORKSPACE_FEEDBACK_SCOPE = "public_self_check_and_policy_denials"
    WORKSPACE_HARNESS_FEEDBACK_MODE = "bash_policy_denial_repair"
    HIDDEN_JUDGE_FEEDBACK_SCOPE = "hidden_judge_pass_fail_and_policy_denials"
    HIDDEN_JUDGE_HARNESS_FEEDBACK_MODE = "hidden_judge_submit_pass_fail_only"
    HIDDEN_JUDGE_AGENT_FEEDBACK_MODE = "pass_fail_only"
    WORKSPACE_PLAN_PATH = "output/plan.md"
    WORKSPACE_POLICY_REPAIR_INSTRUCTION = "仅在 workspace_root 和允许写入目录内使用 Bash。"

    def run(
        self,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        active_skills: Sequence[str],
    ) -> RunRecord:
        if not session_config.model_config:
            record = FallbackAdapter().run(task_bundle, session_config, active_skills)
            return RunRecord(
                **{
                    **record.__dict__,
                    "provider": self.provider_name,
                    "metadata": {
                        **record.metadata,
                        "bridge_mode": "workspace_placeholder",
                        "agent_mode": "workspace_edit",
                        "prompt_contract_version": self.WORKSPACE_PROMPT_VERSION,
                    },
                }
            )

        provider_name = session_config.model_config.provider_name.lower()
        if provider_name not in {"claude-sdk", "claude_sdk"}:
            raise RuntimeError(
                "ClaudeWorkspaceAdapter only supports provider 'claude-sdk' in the formal "
                "workspace execution flow"
            )
        return self._run_claude_sdk_workspace(task_bundle, session_config, active_skills)

    def _run_claude_sdk_workspace(
        self,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        active_skills: Sequence[str],
    ) -> RunRecord:
        provider_adapter = self._provider_adapter(session_config)
        safe_model_config = provider_adapter.safe_log_config()
        resolved_model_name = self._resolved_model_name(session_config, provider_adapter)
        provider_adapter.resolve_api_key()
        task_spec, runtime_paths = _prepare_runtime_workspace(
            task_bundle, _resolve_workspace_root(task_bundle, session_config)
        )
        policy = resolve_runtime_policy(
            task_spec=task_spec,
            session_config=session_config,
            model_config=session_config.model_config,
        )
        workspace = runtime_paths["runtime_root"]
        work_dir = runtime_paths["work_dir"]
        output_dir = runtime_paths["output_dir"]
        checkpoints_dir = runtime_paths["checkpoints_dir"]
        transcript_path = workspace / "transcript.txt"
        tool_policy = self._coerce_tool_policy(session_config)
        stop_oracle = self._resolve_workspace_stop_oracle(session_config)
        prompt_mode = self._resolve_workspace_prompt_mode(session_config)
        completion_policy = self._resolve_workspace_completion_policy(session_config)
        self._seed_workspace_scaffold(runtime_paths)
        if stop_oracle != "hidden_judge_submit":
            self._install_public_self_eval_runtime(
                task_spec=task_spec,
                runtime_paths=runtime_paths,
                tool_policy=tool_policy,
            )
        base_prompt = self._build_workspace_agent_prompt(
            task_bundle,
            active_skills,
            workspace_root=workspace,
            tool_policy=tool_policy,
            stop_oracle=stop_oracle,
            prompt_mode=prompt_mode,
            completion_policy=completion_policy,
        )
        configured_max_iterations = self._resolve_workspace_max_iterations(
            session_config,
            default_limit=1,
        )
        env = _runtime_environment(session_config, task_bundle, runtime_paths)
        sdk_provider_env = (
            provider_adapter.build_sdk_env(env)
            if hasattr(provider_adapter, "build_sdk_env")
            else {}
        )
        sdk_env = self._build_workspace_sdk_env(env, sdk_provider_env)
        configured_iterations_label = (
            "unbounded"
            if configured_max_iterations is None
            else str(configured_max_iterations)
        )
        transcript_lines = [
            f"provider={self.provider_name}",
            "sdk_backend=claude_sdk",
            f"feedback_scope={self._workspace_feedback_scope(stop_oracle)}",
            f"harness_feedback_mode={self._workspace_harness_feedback_mode(stop_oracle)}",
            f"agent_stop_policy={self._agent_stop_policy(stop_oracle)}",
            f"stop_oracle={stop_oracle}",
            f"workspace_prompt_mode={prompt_mode}",
            f"workspace_completion_policy={completion_policy}",
            f"task_id={task_bundle.task_id}",
            f"model_provider={session_config.model_config.provider_name}",
            f"model_name={resolved_model_name}",
            f"api_key_env={session_config.model_config.api_key_env or 'ANTHROPIC_API_KEY'}",
            f"skills={','.join(active_skills)}",
            f"allowed_tools={','.join(self._all_allowed_tools(stop_oracle, tool_policy=tool_policy))}",
            f"runtime_layout={json.dumps(task_spec.get('runtime_layout') or {}, sort_keys=True)}",
            f"effective_model_timeout_seconds={policy.model_timeout_seconds}",
            f"effective_execution_budget_seconds={policy.execution_budget_seconds}",
            f"configured_max_workspace_iterations={configured_iterations_label}",
            f"workspace_query_limit={configured_iterations_label}",
            f"task_python_executable={env.get('MYEVOSKILL_TASK_PYTHON', sys.executable)}",
            f"task_env_hash={env.get('MYEVOSKILL_TASK_ENV_HASH', session_config.env_hash)}",
            f"workspace_python_executable={sdk_env.get('MYEVOSKILL_PYTHON_EXE', sys.executable)}",
            "",
        ]

        final_completed = subprocess.CompletedProcess(["python", "work/main.py"], 1, "", "")
        final_timed_out = False
        final_runtime = 0.0
        final_metadata: Dict[str, Any] = {}
        final_self_check: Dict[str, Any] = {
            "self_check_passed": False,
            "schema_warnings": ["workspace agent did not run"],
        }
        final_files_written: list[str] = []
        commands_run: list[str] = []
        workspace_write_violations: list[dict[str, Any]] = []
        workspace_policy_denials: list[dict[str, Any]] = []
        iteration_count = 0
        trajectory_rounds: list[dict[str, Any]] = []
        final_error_info: dict[str, Any] = {}
        final_submission_state = self._empty_submission_state()
        private_submission_state = self._empty_private_submission_state()
        final_hidden_judge_result: Optional[JudgeResult] = None
        final_plan_feedback: Optional[dict[str, Any]] = None
        run_failure_reason = ""
        output_contract_satisfied_post_run = False
        entrypoint_run_seen_in_trace = False
        public_self_eval_seen_in_trace = False
        final_provider_session_id = session_config.run_id
        execution_deadline = time.monotonic() + max(1, int(policy.execution_budget_seconds))
        round_index = 1
        previous_feedback: Optional[dict[str, Any]] = None
        previous_summary: Optional[dict[str, Any]] = None

        while configured_max_iterations is None or round_index <= configured_max_iterations:
            remaining_budget_seconds = max(
                0,
                int(execution_deadline - time.monotonic()),
            )
            if remaining_budget_seconds <= 0:
                final_timed_out = True
                final_runtime = float(policy.execution_budget_seconds)
                run_failure_reason = "execution_budget_exhausted"
                final_error_info = {
                    "error_type": "execution_budget_exhausted",
                    "message": (
                        "workspace execution exceeded the total execution budget before "
                        "producing a successful run"
                    ),
                }
                break
            round_model_timeout_seconds = max(
                1,
                min(int(policy.model_timeout_seconds), remaining_budget_seconds),
            )
            iteration_count = round_index
            round_prompt = self._compose_workspace_sdk_round_prompt(
                base_prompt=base_prompt,
                round_index=round_index,
                previous_feedback=previous_feedback,
                previous_summary=previous_summary,
            )
            (workspace / f"agent_prompt_round_{round_index}.txt").write_text(
                round_prompt,
                encoding="utf-8",
            )
            transcript_lines.extend(
                [
                    f"QUERY {round_index}",
                    f"prompt_file=agent_prompt_round_{round_index}.txt",
                ]
            )
            round_trace: dict[str, Any] = {
                "round_index": round_index,
                "prompt_path": f"agent_prompt_round_{round_index}.txt",
            }
            trajectory_rounds.append(round_trace)

            read_snapshot = self._snapshot_roots(workspace, tool_policy["read_roots"])
            work_dir.mkdir(parents=True, exist_ok=True)
            writable_before = self._snapshot_roots(workspace, tool_policy["write_roots"])

            response_payload: Optional[dict[str, Any]] = None
            response_metadata: dict[str, Any] = {}
            response_error: Optional[ClaudeSDKExecutionError] = None
            try:
                response_payload, response_metadata = self._generate_workspace_sdk_response(
                    task_bundle=task_bundle,
                    session_config=session_config,
                    task_spec=task_spec,
                    runtime_paths=runtime_paths,
                    workspace=workspace,
                    prompt=round_prompt,
                    model_timeout_seconds=round_model_timeout_seconds,
                    round_index=round_index,
                    tool_policy=tool_policy,
                    readonly_before=read_snapshot,
                    stop_oracle=stop_oracle,
                    completion_policy=completion_policy,
                    sdk_env=sdk_env,
                )
            except ClaudeSDKExecutionError as exc:
                response_error = exc
            except Exception as exc:
                (workspace / f"claude_sdk_error_round_{round_index}.txt").write_text(
                    str(exc) + "\n",
                    encoding="utf-8",
                )
                transcript_lines.extend(
                    [
                        f"QUERY {round_index}",
                        f"prompt_file=agent_prompt_round_{round_index}.txt",
                        f"round_error={exc}",
                        "",
                    ]
                )
                transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
                raise

            writable_after = self._snapshot_roots(workspace, tool_policy["write_roots"])
            readonly_after = self._snapshot_roots(workspace, tool_policy["read_roots"])
            readonly_mutations = self._detect_snapshot_mutations(read_snapshot, readonly_after)

            sdk_messages = list(response_metadata.get("sdk_messages", []))
            sdk_diagnostics = dict(response_metadata.get("sdk_diagnostics", {}))
            raw_result_text = str(response_metadata.get("raw_response_text", ""))
            vendor_session_ref = dict(response_metadata.get("vendor_session_ref", {}))
            round_private_submission_state = self._merge_private_submission_state(
                self._empty_private_submission_state(),
                response_metadata.pop("private_submission_state", {}),
            )
            if response_error is not None:
                sdk_messages = list(response_error.sdk_messages)
                sdk_diagnostics = dict(response_error.diagnostics)
                raw_result_text = str(response_error.result_text or "")
                vendor_session_ref = dict(sdk_diagnostics.get("vendor_session_ref", {}))
                round_private_submission_state = self._merge_private_submission_state(
                    round_private_submission_state,
                    getattr(response_error, "private_state", {}),
                )
            private_submission_state = self._merge_private_submission_state(
                private_submission_state,
                round_private_submission_state,
            )
            latest_private_judge_result = self._latest_private_judge_result(round_private_submission_state)
            if latest_private_judge_result is not None:
                final_hidden_judge_result = latest_private_judge_result
            sanitized_sdk_messages = self._sanitize_workspace_visible_payload(
                sdk_messages,
                workspace_root=workspace,
            )
            sanitized_sdk_diagnostics = self._sanitize_workspace_visible_payload(
                sdk_diagnostics,
                workspace_root=workspace,
            )
            sanitized_vendor_session_ref = self._sanitize_workspace_visible_payload(
                vendor_session_ref,
                workspace_root=workspace,
            )
            sanitized_raw_result_text = self._sanitize_workspace_visible_payload(
                raw_result_text,
                workspace_root=workspace,
            )

            round_trace["sdk_messages"] = list(sanitized_sdk_messages)
            round_trace["vendor_session_ref"] = dict(sanitized_vendor_session_ref)
            round_trace["raw_result_text"] = sanitized_raw_result_text
            if sdk_diagnostics:
                round_trace["claude_diagnostics"] = dict(sanitized_sdk_diagnostics)
            (workspace / f"sdk_messages_round_{round_index}.json").write_text(
                json.dumps(sanitized_sdk_messages, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if sdk_diagnostics:
                (workspace / f"claude_sdk_diagnostics_round_{round_index}.json").write_text(
                    json.dumps(sanitized_sdk_diagnostics, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            round_submission_state = self._resolve_submission_state(
                response_metadata=response_metadata,
                sdk_diagnostics=sdk_diagnostics,
            )
            final_submission_state = self._merge_submission_state(
                final_submission_state,
                round_submission_state,
            )
            round_trace["submission_state"] = dict(round_submission_state)
            round_policy_denials = self._extract_denied_bash_attempts(
                sdk_diagnostics.get("hook_events", []),
                workspace_root=workspace,
                tool_policy=tool_policy,
            )
            if round_policy_denials:
                workspace_policy_denials.extend(round_policy_denials)
                round_trace["workspace_policy_denials"] = list(round_policy_denials)
                (workspace / f"workspace_policy_denials_round_{round_index}.json").write_text(
                    json.dumps({"policy_denials": round_policy_denials}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            if stop_oracle in {"submit_tool", "hidden_judge_submit"}:
                self._write_submission_artifacts(
                    workspace=workspace,
                    round_index=round_index,
                    submission_state=round_submission_state,
                    stop_oracle=stop_oracle,
                )

            if readonly_mutations:
                readonly_violations = [
                    self._make_workspace_violation(
                        "<post-run-audit>",
                        "readonly_mutation",
                        path=item,
                        detail="read-only path modified",
                    )
                    for item in readonly_mutations
                ]
                workspace_write_violations.extend(readonly_violations)
                (workspace / f"workspace_write_violations_round_{round_index}.json").write_text(
                    json.dumps({"violations": readonly_violations}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                round_trace["error_type"] = "permission_violation"
                round_trace["error_message"] = "workspace agent modified read-only paths"
                round_trace["readonly_mutations"] = list(readonly_violations)
                final_error_info = {
                    "error_type": "permission_violation",
                    "message": "workspace agent modified read-only paths: "
                    + self._format_workspace_violations(readonly_violations),
                }
                self._write_workspace_trajectory_artifacts(
                    workspace=workspace,
                    task_id=task_bundle.task_id,
                    model_provider=session_config.model_config.provider_name,
                    model_name=resolved_model_name,
                    trajectory_rounds=trajectory_rounds,
                    final_status="error",
                    final_error_info=final_error_info,
                )
                raise RuntimeError(
                    "workspace agent modified read-only paths: "
                    + self._format_workspace_violations(readonly_violations)
                )

            round_commands = self._extract_sdk_commands(sdk_messages)
            if response_payload is not None:
                round_commands = list(dict.fromkeys(round_commands + list(response_payload["commands_run"])))
            invalid_commands = self._validate_bash_commands(round_commands, workspace, tool_policy)
            handled_denials, invalid_commands = self._split_handled_bash_policy_violations(
                invalid_commands,
                round_policy_denials,
            )
            if handled_denials and "workspace_policy_denials" not in round_trace:
                round_trace["workspace_policy_denials"] = list(round_policy_denials)
            round_commands = self._filter_denied_bash_commands(round_commands, round_policy_denials)
            if invalid_commands:
                workspace_write_violations.extend(invalid_commands)
                (workspace / f"workspace_write_violations_round_{round_index}.json").write_text(
                    json.dumps({"violations": invalid_commands}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                round_trace["error_type"] = "bash_policy_violation"
                round_trace["error_message"] = "workspace agent attempted disallowed bash commands"
                round_trace["invalid_commands"] = list(invalid_commands)
                final_error_info = {
                    "error_type": "bash_policy_violation",
                    "message": "workspace agent attempted disallowed bash commands: "
                    + self._format_workspace_violations(invalid_commands),
                }
                self._write_workspace_trajectory_artifacts(
                    workspace=workspace,
                    task_id=task_bundle.task_id,
                    model_provider=session_config.model_config.provider_name,
                    model_name=resolved_model_name,
                    trajectory_rounds=trajectory_rounds,
                    final_status="error",
                    final_error_info=final_error_info,
                )
                raise RuntimeError(
                    "workspace agent attempted disallowed bash commands: "
                    + self._format_workspace_violations(invalid_commands)
                )

            final_files_written = self._collect_changed_paths(writable_before, writable_after)
            commands_run.extend(round_commands)
            round_trace["files_written"] = list(final_files_written)
            round_trace["commands_run"] = list(round_commands)
            round_plan_feedback = self._evaluate_workspace_plan_requirement(
                sdk_messages=sdk_messages,
                workspace_root=workspace,
                files_written=final_files_written,
                submission_state=final_submission_state,
            )
            final_plan_feedback = dict(round_plan_feedback) if round_plan_feedback is not None else None
            if round_plan_feedback is not None:
                round_trace["workspace_plan_feedback"] = dict(round_plan_feedback)
                (workspace / f"workspace_plan_feedback_round_{round_index}.json").write_text(
                    json.dumps(round_plan_feedback, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            (workspace / f"files_written_round_{round_index}.json").write_text(
                json.dumps(final_files_written, indent=2),
                encoding="utf-8",
            )
            (workspace / f"commands_run_round_{round_index}.json").write_text(
                json.dumps(round_commands, indent=2),
                encoding="utf-8",
            )
            entrypoint_run_seen_in_trace = any(
                "work/main.py" in str(command).replace("\\", "/").lower()
                and "python" in str(command).lower()
                for command in round_commands
            )
            public_self_eval_seen_in_trace = any(
                "evaluation/self_eval.py" in str(command).replace("\\", "/").lower()
                and "python" in str(command).lower()
                for command in round_commands
            )
            entrypoint_tool_result = self._find_latest_bash_command_result(
                sdk_messages,
                self._workspace_entrypoint_command(),
            )
            if entrypoint_tool_result is not None:
                round_trace["entrypoint_tool_result"] = dict(entrypoint_tool_result)
            can_retry_after_round = (
                configured_max_iterations is None or round_index < configured_max_iterations
            )

            if round_plan_feedback is not None and can_retry_after_round:
                transcript_lines.extend(
                    [
                        "protocol_status=repair_pending",
                        f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                        f"workspace_plan_feedback={json.dumps(round_plan_feedback, sort_keys=True)}",
                        "",
                    ]
                )
                round_trace["protocol_status"] = "repair_pending"
                round_trace["error_type"] = "workspace_plan_required"
                round_trace["error_message"] = (
                    "workspace agent must update output/plan.md before code edits or solver execution"
                )
                previous_feedback = dict(round_plan_feedback)
                previous_summary = dict(response_payload or {})
                round_index += 1
                continue

            if (
                response_error is not None
                and stop_oracle != "hidden_judge_submit"
                and completion_policy == "main_success_output_contract"
                and entrypoint_tool_result is not None
                and entrypoint_tool_result.get("succeeded", False)
            ):
                trace_completed = subprocess.CompletedProcess(
                    ["python", "work/main.py"],
                    int(entrypoint_tool_result.get("returncode", 0) or 0),
                    str(entrypoint_tool_result.get("stdout", "") or ""),
                    str(entrypoint_tool_result.get("stderr", "") or ""),
                )
                trace_completion_check = self._evaluate_workspace_completion(
                    task_spec=task_spec,
                    runtime_paths=runtime_paths,
                    completed=trace_completed,
                    timed_out=False,
                    entrypoint="work/main.py",
                    env=env,
                    timeout_seconds=remaining_budget_seconds,
                    completion_policy=completion_policy,
                )
                if trace_completion_check.get("self_check_passed", False):
                    sdk_diagnostics = dict(sdk_diagnostics)
                    sdk_diagnostics["external_completion"] = {
                        "entrypoint_result": dict(entrypoint_tool_result),
                        "completion_check": dict(trace_completion_check),
                        "fallback_from_error_type": str(response_error.error_type or ""),
                    }
                    response_payload = self._default_workspace_summary(
                        task_spec,
                        solver_summary=(
                            "python work/main.py succeeded and produced the required output artifacts."
                        ),
                    )
                    response_metadata = self._build_sdk_metadata(
                        sdk_messages=sdk_messages,
                        parsed_summary=response_payload,
                        bridge_mode="workspace_claude_sdk",
                        model_provider_kind="claude_sdk",
                        raw_response_text=raw_result_text,
                        sdk_diagnostics=sdk_diagnostics,
                        submission_state=final_submission_state,
                        completion_source="external_output_contract",
                    )
                    response_metadata["vendor_session_ref"] = dict(vendor_session_ref)
                    response_error = None

            if response_error is not None:
                protocol_failure_reason = self._normalize_protocol_failure_reason(
                    response_error.error_type
                )
                if (
                    stop_oracle == "hidden_judge_submit"
                    and not bool(final_submission_state.get("submission_attempted", False))
                    and can_retry_after_round
                    and protocol_failure_reason not in {"sdk_error", "request_timeout"}
                ):
                    submission_feedback = self._build_hidden_judge_submission_required_feedback()
                    transcript_lines.extend(
                        [
                            "protocol_status=repair_pending",
                            f"protocol_failure_reason={protocol_failure_reason}",
                            f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                            "submission_status=not_submitted",
                            "",
                        ]
                    )
                    round_trace["protocol_status"] = "repair_pending"
                    round_trace["protocol_failure_reason"] = protocol_failure_reason
                    round_trace["sdk_result"] = dict(sdk_diagnostics.get("sdk_result", {}))
                    round_trace["error_type"] = "hidden_judge_not_submitted"
                    round_trace["error_message"] = str(response_error)
                    round_trace["repair_feedback"] = dict(submission_feedback)
                    previous_feedback = submission_feedback
                    previous_summary = dict(response_payload or {})
                    round_index += 1
                    continue
                if (
                    stop_oracle == "hidden_judge_submit"
                    and self._hidden_judge_submission_failed(final_submission_state)
                    and can_retry_after_round
                    and protocol_failure_reason not in {"sdk_error", "request_timeout"}
                ):
                    hidden_judge_feedback = self._build_hidden_judge_retry_feedback()
                    transcript_lines.extend(
                        [
                            "protocol_status=repair_pending",
                            f"protocol_failure_reason={protocol_failure_reason}",
                            f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                            "submission_status=fail",
                            "",
                        ]
                    )
                    round_trace["protocol_status"] = "repair_pending"
                    round_trace["protocol_failure_reason"] = protocol_failure_reason
                    round_trace["sdk_result"] = dict(sdk_diagnostics.get("sdk_result", {}))
                    round_trace["error_type"] = "hidden_judge_fail"
                    round_trace["error_message"] = str(response_error)
                    round_trace["repair_feedback"] = dict(hidden_judge_feedback)
                    previous_feedback = hidden_judge_feedback
                    previous_summary = self._latest_submission_request_summary(final_submission_state)
                    round_index += 1
                    continue
                if (
                    round_policy_denials
                    and can_retry_after_round
                    and protocol_failure_reason not in {"sdk_error", "request_timeout"}
                ):
                    denial_feedback = self._build_workspace_policy_feedback(
                        round_policy_denials
                    )
                    transcript_lines.extend(
                        [
                            "protocol_status=repair_pending",
                            f"protocol_failure_reason={protocol_failure_reason}",
                            f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                            f"workspace_policy_denials={json.dumps(round_policy_denials, sort_keys=True)}",
                            "",
                        ]
                    )
                    round_trace["protocol_status"] = "repair_pending"
                    round_trace["protocol_failure_reason"] = protocol_failure_reason
                    round_trace["sdk_result"] = dict(sdk_diagnostics.get("sdk_result", {}))
                    round_trace["error_type"] = "bash_policy_denial"
                    round_trace["error_message"] = str(response_error)
                    round_trace["repair_feedback"] = dict(denial_feedback)
                    previous_feedback = denial_feedback
                    previous_summary = dict(response_payload or {})
                    round_index += 1
                    continue
                response_metadata = self._build_sdk_failure_metadata(
                    sdk_messages=sdk_messages,
                    bridge_mode="workspace_claude_sdk",
                    model_provider_kind="claude_sdk",
                    raw_response_text=raw_result_text,
                    sdk_diagnostics=sdk_diagnostics,
                    vendor_session_ref=vendor_session_ref,
                    protocol_failure_reason=protocol_failure_reason,
                    submission_state=final_submission_state,
                )
                protocol_metadata = self._build_protocol_metadata(
                    sdk_diagnostics=sdk_diagnostics,
                    protocol_status="failed",
                    protocol_failure_reason=protocol_failure_reason,
                )
                round_trace["protocol_status"] = "failed"
                round_trace["protocol_failure_reason"] = protocol_failure_reason
                round_trace["sdk_result"] = dict(sdk_diagnostics.get("sdk_result", {}))
                round_trace["error_type"] = protocol_failure_reason
                round_trace["error_message"] = str(response_error)
                (workspace / f"claude_sdk_error_round_{round_index}.txt").write_text(
                    str(response_error) + "\n",
                    encoding="utf-8",
                )
                transcript_lines.extend(
                    [
                        "protocol_status=failed",
                        f"protocol_failure_reason={protocol_failure_reason}",
                        f"files_written={','.join(final_files_written)}",
                        f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                        f"entrypoint_run_seen_in_trace={entrypoint_run_seen_in_trace}",
                        f"public_self_eval_seen_in_trace={public_self_eval_seen_in_trace}",
                        f"round_error={response_error}",
                        "",
                    ]
                )
                final_error_info = {
                    "error_type": protocol_failure_reason,
                    "message": str(response_error),
                }
                final_provider_session_id = (
                    protocol_metadata.get("sdk_result_session_id", "")
                    or str(vendor_session_ref.get("session_id", "") or "")
                    or session_config.run_id
                )
                self._write_workspace_trajectory_artifacts(
                    workspace=workspace,
                    task_id=task_bundle.task_id,
                    model_provider=session_config.model_config.provider_name,
                    model_name=resolved_model_name,
                    trajectory_rounds=trajectory_rounds,
                    final_status="protocol_failed",
                    final_error_info=final_error_info,
                )
                transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
                return RunRecord(
                    run_id=session_config.run_id,
                    task_id=task_bundle.task_id,
                    provider=self.provider_name,
                    provider_session_id=final_provider_session_id,
                    model_provider=session_config.model_config.provider_name,
                    model_name=resolved_model_name,
                    env_hash=session_config.env_hash,
                    skills_active=list(active_skills),
                    workspace_root=workspace,
                    artifacts_uri=str(output_dir),
                    transcript_uri=str(transcript_path),
                    stdout=self._completed_process_text(final_completed.stdout),
                    stderr=self._append_protocol_failure_stderr(
                        self._completed_process_text(final_completed.stderr),
                        str(response_error),
                    ),
                    runtime_seconds=final_runtime,
                    judge_result=final_hidden_judge_result,
                    metadata={
                        "execution_mode": session_config.execution_mode,
                        "returncode": final_completed.returncode,
                        "command": ["python", "work/main.py"],
                        "model_provider_kind": safe_model_config.get("kind", ""),
                        "api_key_env": session_config.model_config.api_key_env or "ANTHROPIC_API_KEY",
                        "runtime_layout": task_spec.get("runtime_layout") or {},
                        "runtime_root": str(workspace),
                        "work_dir": str(work_dir),
                        "output_dir": str(output_dir),
                        "checkpoints_dir": str(checkpoints_dir),
                        "agent_mode": "workspace_edit",
                        "sdk_backend": "claude_sdk",
                        "allowed_tools": list(
                            self._all_allowed_tools(stop_oracle, tool_policy=tool_policy)
                        ),
                        "tool_policy_summary": self._summarize_tool_policy(tool_policy),
                        "feedback_scope": self._workspace_feedback_scope(stop_oracle),
                        "harness_feedback_mode": self._workspace_harness_feedback_mode(stop_oracle),
                        "agent_stop_policy": self._agent_stop_policy(stop_oracle),
                        "stop_oracle": stop_oracle,
                        "workspace_prompt_mode": prompt_mode,
                        "workspace_completion_policy": completion_policy,
                        "configured_max_workspace_iterations": (
                            0 if configured_max_iterations is None else configured_max_iterations
                        ),
                        "iteration_count": iteration_count,
                        "files_written": final_files_written,
                        "commands_run": list(dict.fromkeys(commands_run)),
                        "entrypoint_run_seen_in_trace": entrypoint_run_seen_in_trace,
                        "public_self_eval_seen_in_trace": public_self_eval_seen_in_trace,
                        "public_self_eval_passed_post_run": False,
                        "output_contract_satisfied_post_run": False,
                        "run_status": "failed",
                        "run_failure_reason": f"protocol_{protocol_failure_reason}",
                        "public_self_check_status": final_self_check,
                        "post_run_audit": final_self_check,
                        "workspace_write_violations": workspace_write_violations,
                        "workspace_policy_denials": list(workspace_policy_denials),
                        "hidden_judge_feedback_mode": (
                            self.HIDDEN_JUDGE_AGENT_FEEDBACK_MODE
                            if stop_oracle == "hidden_judge_submit"
                            else ""
                        ),
                        "final_hidden_judge_passed_post_run": self._hidden_judge_submission_passed(
                            final_submission_state
                        ),
                        **(
                            self._submission_metadata(final_submission_state, stop_oracle=stop_oracle)
                            if stop_oracle in {"submit_tool", "hidden_judge_submit"}
                            else {}
                        ),
                        **protocol_metadata,
                        **_timeout_metadata(
                            policy,
                            timed_out=bool(sdk_diagnostics.get("timeout_occurred", False)),
                            timeout_scope=(
                                "model_request"
                                if bool(sdk_diagnostics.get("timeout_occurred", False))
                                else ""
                            ),
                        ),
                        "prompt_contract_version": self.WORKSPACE_PROMPT_VERSION,
                        "command_history_summary": ["workspace_sdk_agent"],
                        **response_metadata,
                    },
                )

            response_payload = self._coerce_sdk_summary(
                {
                    **(response_payload or self._default_workspace_summary(task_spec, solver_summary="")),
                    "files_written": list(final_files_written),
                    "commands_run": list(dict.fromkeys(round_commands)),
                }
            )
            response_metadata = dict(response_metadata)
            response_metadata["parsed_response"] = dict(response_payload)
            round_trace["summary"] = dict(response_payload)
            round_trace["protocol_status"] = "completed"
            round_trace["protocol_failure_reason"] = ""
            round_trace["sdk_result"] = dict(sdk_diagnostics.get("sdk_result", {}))
            (workspace / f"agent_summary_round_{round_index}.json").write_text(
                json.dumps(response_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            external_completion_source = (
                str(response_metadata.get("sdk_completion_source", "") or "")
                == "external_output_contract"
            )
            hidden_judge_passed = self._hidden_judge_submission_passed(final_submission_state)
            hidden_judge_failed = self._hidden_judge_submission_failed(final_submission_state)
            if stop_oracle == "hidden_judge_submit":
                if entrypoint_tool_result is not None:
                    final_completed = subprocess.CompletedProcess(
                        ["python", "work/main.py"],
                        int(entrypoint_tool_result.get("returncode", 0) or 0),
                        str(entrypoint_tool_result.get("stdout", "") or ""),
                        str(entrypoint_tool_result.get("stderr", "") or ""),
                    )
                else:
                    final_completed = subprocess.CompletedProcess(["python", "work/main.py"], 1, "", "")
                final_timed_out = False
                final_runtime = 0.0
                final_self_check = self._evaluate_workspace_completion(
                    task_spec=task_spec,
                    runtime_paths=runtime_paths,
                    completed=final_completed,
                    timed_out=final_timed_out,
                    entrypoint="work/main.py",
                    env=env,
                    timeout_seconds=max(
                        1,
                        max(0, int(execution_deadline - time.monotonic())),
                    ),
                    completion_policy="main_success_output_contract",
                )
            elif external_completion_source:
                if entrypoint_tool_result is None or not entrypoint_tool_result.get("succeeded", False):
                    raise RuntimeError(
                        "external_output_contract completion requires a successful `python work/main.py` trace"
                    )
                final_completed = subprocess.CompletedProcess(
                    ["python", "work/main.py"],
                    int(entrypoint_tool_result.get("returncode", 0) or 0),
                    str(entrypoint_tool_result.get("stdout", "") or ""),
                    str(entrypoint_tool_result.get("stderr", "") or ""),
                )
                final_timed_out = False
                final_runtime = 0.0
                final_self_check = self._evaluate_workspace_completion(
                    task_spec=task_spec,
                    runtime_paths=runtime_paths,
                    completed=final_completed,
                    timed_out=final_timed_out,
                    entrypoint="work/main.py",
                    env=env,
                    timeout_seconds=remaining_budget_seconds,
                    completion_policy=completion_policy,
                )
            else:
                _configure_workspace_permissions(runtime_paths, lock_runtime_root=False)
                _configure_workspace_permissions(runtime_paths, lock_runtime_root=True)
                start = time.time()
                final_completed, final_timed_out = _run_subprocess(
                    ["python", "work/main.py"],
                    cwd=workspace,
                    env=env,
                    timeout_seconds=remaining_budget_seconds,
                )
                final_runtime = time.time() - start
                _configure_workspace_permissions(runtime_paths, lock_runtime_root=False)
                final_self_check = self._evaluate_workspace_completion(
                    task_spec=task_spec,
                    runtime_paths=runtime_paths,
                    completed=final_completed,
                    timed_out=final_timed_out,
                    entrypoint="work/main.py",
                    env=env,
                    timeout_seconds=max(
                        1,
                        max(0, int(execution_deadline - time.monotonic())),
                    ),
                    completion_policy=completion_policy,
                )

            final_completed_stdout = self._completed_process_text(final_completed.stdout)
            final_completed_stderr = self._completed_process_text(final_completed.stderr)
            (workspace / f"stdout_round_{round_index}.log").write_text(
                final_completed_stdout,
                encoding="utf-8",
            )
            (workspace / f"stderr_round_{round_index}.log").write_text(
                final_completed_stderr,
                encoding="utf-8",
            )
            if stop_oracle != "hidden_judge_submit":
                (workspace / "post_run_audit.json").write_text(
                    json.dumps(final_self_check, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                (workspace / f"public_self_eval_round_{round_index}.json").write_text(
                    json.dumps(final_self_check, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                (workspace / f"public_self_eval_stdout_round_{round_index}.log").write_text(
                    str(final_self_check.get("public_self_eval_stdout", "") or ""),
                    encoding="utf-8",
                )
                (workspace / f"public_self_eval_stderr_round_{round_index}.log").write_text(
                    str(final_self_check.get("public_self_eval_stderr", "") or ""),
                    encoding="utf-8",
                )
            round_trace["post_run_audit"] = dict(final_self_check)
            round_trace["public_self_check"] = dict(final_self_check)
            round_trace["returncode"] = final_completed.returncode
            output_contract_satisfied_post_run = bool(final_self_check.get("self_check_passed"))
            if stop_oracle == "hidden_judge_submit":
                if hidden_judge_passed:
                    run_failure_reason = ""
                elif hidden_judge_failed:
                    run_failure_reason = "hidden_judge_fail"
                else:
                    run_failure_reason = "hidden_judge_not_submitted"
            elif final_timed_out:
                run_failure_reason = "entrypoint_timeout"
            elif final_completed.returncode != 0:
                run_failure_reason = "entrypoint_failed"
            elif not final_self_check.get("output_exists", False):
                run_failure_reason = "missing_output_artifact"
            elif final_self_check.get("missing_output_fields") or final_self_check.get("schema_warnings"):
                run_failure_reason = "output_contract_failed"
            else:
                run_failure_reason = ""
            if round_plan_feedback is not None:
                run_failure_reason = "workspace_plan_required"
            run_status = "succeeded" if not run_failure_reason else "failed"
            transcript_lines.extend(
                [
                    f"response_format={response_metadata.get('response_format', '')}",
                    "protocol_status=completed",
                    f"files_written={','.join(final_files_written)}",
                    f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                    f"entrypoint_run_seen_in_trace={entrypoint_run_seen_in_trace}",
                    f"public_self_eval_seen_in_trace={public_self_eval_seen_in_trace}",
                    f"returncode={final_completed.returncode}",
                    f"public_self_eval_passed_post_run={bool(final_self_check.get('public_self_eval_passed', False))}",
                    f"output_contract_satisfied_post_run={output_contract_satisfied_post_run}",
                    f"final_hidden_judge_passed_post_run={hidden_judge_passed}",
                    f"run_status={run_status}",
                    f"run_failure_reason={run_failure_reason}",
                    "",
                ]
            )

            final_metadata = {
                **response_metadata,
                "bridge_mode": response_metadata.get("bridge_mode", "workspace_claude_sdk"),
            }
            final_provider_session_id = (
                str(final_metadata.get("sdk_result_session_id", "") or "")
                or str(vendor_session_ref.get("session_id", "") or "")
                or session_config.run_id
            )
            sdk_signaled_completion = (
                str(response_metadata.get("sdk_completion_source", "") or "") == "result_message"
            )
            if stop_oracle == "hidden_judge_submit":
                if not bool(final_submission_state.get("submission_attempted", False)) and can_retry_after_round:
                    submission_feedback = self._build_hidden_judge_submission_required_feedback()
                    transcript_lines.extend(
                        [
                            "protocol_status=repair_pending",
                            f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                            "submission_status=not_submitted",
                            "",
                        ]
                    )
                    round_trace["error_type"] = "hidden_judge_not_submitted"
                    round_trace["error_message"] = (
                        "workspace agent returned a summary without calling submit_result(...)"
                    )
                    round_trace["repair_feedback"] = dict(submission_feedback)
                    previous_feedback = submission_feedback
                    previous_summary = dict(response_payload)
                    round_index += 1
                    continue
                if hidden_judge_failed and can_retry_after_round:
                    hidden_judge_feedback = self._build_hidden_judge_retry_feedback()
                    transcript_lines.extend(
                        [
                            "protocol_status=repair_pending",
                            f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                            "submission_status=fail",
                            "",
                        ]
                    )
                    round_trace["error_type"] = "hidden_judge_fail"
                    round_trace["error_message"] = (
                        "workspace agent submitted to the hidden judge and received fail"
                    )
                    round_trace["repair_feedback"] = dict(hidden_judge_feedback)
                    previous_feedback = hidden_judge_feedback
                    previous_summary = (
                        self._latest_submission_request_summary(final_submission_state)
                        or dict(response_payload)
                    )
                    round_index += 1
                    continue
                if hidden_judge_passed:
                    break
                if sdk_signaled_completion:
                    break
                round_index += 1
                continue
            if round_policy_denials and not output_contract_satisfied_post_run and can_retry_after_round:
                denial_feedback = self._build_workspace_policy_feedback(round_policy_denials)
                transcript_lines.extend(
                    [
                        "protocol_status=repair_pending",
                        f"commands_run={json.dumps(round_commands, sort_keys=True)}",
                        f"workspace_policy_denials={json.dumps(round_policy_denials, sort_keys=True)}",
                        "",
                    ]
                )
                round_trace["error_type"] = "bash_policy_denial"
                round_trace["error_message"] = (
                    "workspace agent hit recoverable Bash policy denials and will retry"
                )
                round_trace["repair_feedback"] = dict(denial_feedback)
                previous_feedback = denial_feedback
                previous_summary = dict(response_payload)
                round_index += 1
                continue
            if sdk_signaled_completion or output_contract_satisfied_post_run:
                break
            round_index += 1

        final_hidden_judge_passed_post_run = self._hidden_judge_submission_passed(
            final_submission_state
        )
        final_run_succeeded = (
            final_hidden_judge_passed_post_run
            if stop_oracle == "hidden_judge_submit"
            else bool(final_self_check.get("self_check_passed"))
        )
        final_run_succeeded = final_run_succeeded and final_plan_feedback is None
        if not final_run_succeeded and not final_error_info:
            final_error_info = {
                "error_type": run_failure_reason or "post_run_output_audit_failed",
                "message": (
                    "workspace agent did not update output/plan.md before making changes"
                    if final_plan_feedback is not None
                    else "workspace agent returned a summary but the post-run output audit failed"
                ),
            }
        self._write_workspace_trajectory_artifacts(
            workspace=workspace,
            task_id=task_bundle.task_id,
            model_provider=session_config.model_config.provider_name,
            model_name=resolved_model_name,
            trajectory_rounds=trajectory_rounds,
            final_status=("success" if final_run_succeeded else "failed"),
            final_error_info=final_error_info,
        )
        transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        sdk_diagnostics = dict(final_metadata.get("sdk_diagnostics", {}))
        protocol_metadata = self._build_protocol_metadata(
            sdk_diagnostics=sdk_diagnostics,
            protocol_status="completed",
            protocol_failure_reason="",
        )
        return RunRecord(
            run_id=session_config.run_id,
            task_id=task_bundle.task_id,
            provider=self.provider_name,
            provider_session_id=final_provider_session_id,
            model_provider=session_config.model_config.provider_name,
            model_name=resolved_model_name,
            env_hash=session_config.env_hash,
            skills_active=list(active_skills),
            workspace_root=workspace,
            artifacts_uri=str(output_dir),
            transcript_uri=str(transcript_path),
            stdout=final_completed_stdout,
            stderr=final_completed_stderr,
            runtime_seconds=final_runtime,
            judge_result=final_hidden_judge_result,
            metadata={
                "execution_mode": session_config.execution_mode,
                "returncode": final_completed.returncode,
                "command": ["python", "work/main.py"],
                "model_provider_kind": safe_model_config.get("kind", ""),
                "api_key_env": session_config.model_config.api_key_env or "ANTHROPIC_API_KEY",
                "runtime_layout": task_spec.get("runtime_layout") or {},
                "runtime_root": str(workspace),
                "work_dir": str(work_dir),
                "output_dir": str(output_dir),
                "checkpoints_dir": str(checkpoints_dir),
                "task_python_executable": str(
                    env.get("MYEVOSKILL_TASK_PYTHON")
                    or sdk_env.get("MYEVOSKILL_PYTHON_EXE")
                    or sys.executable
                ),
                "task_env_hash": str(
                    env.get("MYEVOSKILL_TASK_ENV_HASH") or session_config.env_hash
                ),
                "task_env_backend": str(env.get("MYEVOSKILL_TASK_ENV_BACKEND") or ""),
                "task_env_ready": bool(
                    ((task_spec.get("runtime_env") or {}) if isinstance(task_spec, Mapping) else {}).get(
                        "ready",
                        False,
                    )
                ),
                "agent_mode": "workspace_edit",
                "sdk_backend": "claude_sdk",
                "allowed_tools": list(
                    self._all_allowed_tools(stop_oracle, tool_policy=tool_policy)
                ),
                "tool_policy_summary": self._summarize_tool_policy(tool_policy),
                "feedback_scope": self._workspace_feedback_scope(stop_oracle),
                "harness_feedback_mode": self._workspace_harness_feedback_mode(stop_oracle),
                "agent_stop_policy": self._agent_stop_policy(stop_oracle),
                "stop_oracle": stop_oracle,
                "workspace_prompt_mode": prompt_mode,
                "workspace_completion_policy": completion_policy,
                "configured_max_workspace_iterations": (
                    0 if configured_max_iterations is None else configured_max_iterations
                ),
                "iteration_count": iteration_count,
                "files_written": final_files_written,
                "commands_run": list(dict.fromkeys(commands_run)),
                "entrypoint_run_seen_in_trace": entrypoint_run_seen_in_trace,
                "public_self_eval_seen_in_trace": public_self_eval_seen_in_trace,
                "public_self_eval_passed_post_run": bool(
                    final_self_check.get("public_self_eval_passed", False)
                ),
                "final_hidden_judge_passed_post_run": final_hidden_judge_passed_post_run,
                "hidden_judge_feedback_mode": (
                    self.HIDDEN_JUDGE_AGENT_FEEDBACK_MODE
                    if stop_oracle == "hidden_judge_submit"
                    else ""
                ),
                "output_contract_satisfied_post_run": output_contract_satisfied_post_run,
                "run_status": run_status,
                "run_failure_reason": run_failure_reason,
                "public_self_check_status": final_self_check,
                "post_run_audit": final_self_check,
                "workspace_write_violations": workspace_write_violations,
                "workspace_policy_denials": list(workspace_policy_denials),
                **(
                    self._submission_metadata(final_submission_state, stop_oracle=stop_oracle)
                    if stop_oracle in {"submit_tool", "hidden_judge_submit"}
                    else {}
                ),
                **protocol_metadata,
                **_timeout_metadata(
                    policy,
                    timed_out=final_timed_out,
                    timeout_scope="solver_execution" if final_timed_out else "",
                ),
                "prompt_contract_version": self.WORKSPACE_PROMPT_VERSION,
                "command_history_summary": (
                    ["workspace_sdk_agent", "workspace_run"]
                    if stop_oracle == "hidden_judge_submit"
                    else [
                        "workspace_sdk_agent",
                        "workspace_run",
                        "workspace_public_self_eval",
                    ]
                ),
                **final_metadata,
            },
        )

    def _run_legacy_workspace(
        self,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        active_skills: Sequence[str],
    ) -> RunRecord:
        provider_adapter = self._provider_adapter(session_config)
        safe_model_config = provider_adapter.safe_log_config()
        resolved_model_name = self._resolved_model_name(session_config, provider_adapter)
        task_spec, runtime_paths = _prepare_runtime_workspace(
            task_bundle, _resolve_workspace_root(task_bundle, session_config)
        )
        policy = resolve_runtime_policy(
            task_spec=task_spec,
            session_config=session_config,
            model_config=session_config.model_config,
        )
        workspace = runtime_paths["runtime_root"]
        work_dir = runtime_paths["work_dir"]
        output_dir = runtime_paths["output_dir"]
        checkpoints_dir = runtime_paths["checkpoints_dir"]
        transcript_path = workspace / "transcript.txt"
        base_prompt = self._build_legacy_workspace_agent_prompt(task_bundle, active_skills)
        tool_policy = self._coerce_tool_policy(session_config)
        self._install_public_self_eval_runtime(
            task_spec=task_spec,
            runtime_paths=runtime_paths,
            tool_policy=tool_policy,
        )
        max_iterations = int(
            session_config.provider_extras.get(
                "max_workspace_iterations", self.DEFAULT_MAX_ITERATIONS
            )
            or self.DEFAULT_MAX_ITERATIONS
        )
        env = _runtime_environment(session_config, task_bundle, runtime_paths)
        transcript_lines = [
            f"provider={self.provider_name}",
            "sdk_backend=legacy_http_json",
            "feedback_scope=public_self_check_only",
            f"task_id={task_bundle.task_id}",
            f"model_provider={session_config.model_config.provider_name}",
            f"model_name={resolved_model_name}",
            f"api_key_env={session_config.model_config.api_key_env}",
            f"skills={','.join(active_skills)}",
            f"runtime_layout={json.dumps(task_spec.get('runtime_layout') or {}, sort_keys=True)}",
            f"effective_model_timeout_seconds={policy.model_timeout_seconds}",
            f"effective_execution_budget_seconds={policy.execution_budget_seconds}",
            f"max_workspace_iterations={max_iterations}",
            "",
        ]

        final_completed = subprocess.CompletedProcess(["python3", "work/main.py"], 1, "", "")
        final_timed_out = False
        final_runtime = 0.0
        final_metadata: Dict[str, Any] = {}
        final_self_check: Dict[str, Any] = {
            "self_check_passed": False,
            "schema_warnings": ["workspace agent did not run"],
        }
        final_files_written: list[str] = []
        commands_run: list[str] = []
        workspace_write_violations: list[str] = []
        previous_feedback: Optional[dict[str, Any]] = None
        previous_summary: Optional[dict[str, Any]] = None

        for round_index in range(1, max_iterations + 1):
            round_prompt = self._compose_legacy_workspace_round_prompt(
                base_prompt=base_prompt,
                round_index=round_index,
                previous_feedback=previous_feedback,
                previous_summary=previous_summary,
            )
            (workspace / f"agent_prompt_round_{round_index}.txt").write_text(
                round_prompt,
                encoding="utf-8",
            )
            transcript_lines.extend(
                [
                    f"ROUND {round_index}",
                    f"prompt_file=agent_prompt_round_{round_index}.txt",
                ]
            )

            try:
                response_payload, response_metadata = self._generate_legacy_workspace_response(
                    provider_adapter=provider_adapter,
                    session_config=session_config,
                    prompt=round_prompt,
                    model_timeout_seconds=policy.model_timeout_seconds,
                    round_index=round_index,
                )
            except ModelResponseParseError as exc:
                (workspace / f"raw_response_round_{round_index}.txt").write_text(
                    exc.raw_response,
                    encoding="utf-8",
                )
                parse_error_lines = [
                    f"message={exc}",
                    f"candidate_count={exc.candidate_count}",
                    f"selected_source={exc.selected_source}",
                ]
                if exc.parse_error:
                    parse_error_lines.append(f"parse_error={exc.parse_error}")
                (workspace / f"response_parse_error_round_{round_index}.txt").write_text(
                    "\n".join(parse_error_lines) + "\n",
                    encoding="utf-8",
                )
                raise

            (workspace / f"raw_response_round_{round_index}.txt").write_text(
                str(response_metadata.get("raw_response_text", "")),
                encoding="utf-8",
            )
            (workspace / f"parsed_response_round_{round_index}.json").write_text(
                json.dumps(response_metadata.get("parsed_response", {}), indent=2, sort_keys=True),
                encoding="utf-8",
            )

            files = dict(response_payload["files"])
            entrypoint = str(response_payload["entrypoint"])
            violations = self._validate_workspace_paths(files, entrypoint)
            if violations:
                workspace_write_violations.extend(violations)
                (workspace / f"workspace_write_violations_round_{round_index}.json").write_text(
                    json.dumps({"violations": violations}, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "workspace agent attempted to write outside allowed paths: "
                    + ", ".join(violations)
                )

            ensure_clean_run_directory(work_dir)
            final_files_written = self._write_workspace_files(workspace, files)
            (workspace / f"files_written_round_{round_index}.json").write_text(
                json.dumps(final_files_written, indent=2),
                encoding="utf-8",
            )
            declared_commands = [f"python3 {entrypoint}"]
            commands_run.extend(declared_commands)
            (workspace / f"commands_run_round_{round_index}.json").write_text(
                json.dumps(declared_commands, indent=2),
                encoding="utf-8",
            )
            (workspace / f"agent_summary_round_{round_index}.json").write_text(
                json.dumps(
                    {
                        "solver_summary": response_payload["solver_summary"],
                        "declared_outputs": response_payload["declared_outputs"],
                        "assumptions": response_payload["assumptions"],
                        "files_written": final_files_written,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            _configure_workspace_permissions(runtime_paths, lock_runtime_root=False)
            _configure_workspace_permissions(runtime_paths, lock_runtime_root=True)
            start = time.time()
            final_completed, final_timed_out = _run_subprocess(
                ["python3", entrypoint],
                cwd=workspace,
                env=env,
                timeout_seconds=policy.execution_budget_seconds,
            )
            final_runtime = time.time() - start
            _configure_workspace_permissions(runtime_paths, lock_runtime_root=False)

            final_completed_stdout = self._completed_process_text(final_completed.stdout)
            final_completed_stderr = self._completed_process_text(final_completed.stderr)
            (workspace / f"stdout_round_{round_index}.log").write_text(
                final_completed_stdout,
                encoding="utf-8",
            )
            (workspace / f"stderr_round_{round_index}.log").write_text(
                final_completed_stderr,
                encoding="utf-8",
            )
            final_self_check = self._public_self_check(
                task_spec=task_spec,
                runtime_paths=runtime_paths,
                completed=final_completed,
                timed_out=final_timed_out,
                entrypoint=entrypoint,
                env=env,
                timeout_seconds=policy.execution_budget_seconds,
            )
            (workspace / f"public_self_check_round_{round_index}.json").write_text(
                json.dumps(final_self_check, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            transcript_lines.extend(
                [
                    f"response_format={response_metadata.get('response_format', '')}",
                    f"files_written={','.join(final_files_written)}",
                    f"returncode={final_completed.returncode}",
                    f"self_check_passed={final_self_check.get('self_check_passed', False)}",
                    "",
                ]
            )

            final_metadata = {
                **response_metadata,
                "entrypoint": entrypoint,
                "bridge_mode": response_metadata.get(
                    "bridge_mode", "workspace_openai_compatible"
                ),
                "sdk_backend": "legacy_http_json",
                "feedback_scope": "public_self_check_only",
            }
            previous_feedback = final_self_check
            previous_summary = response_metadata.get("parsed_response", {})
            if final_self_check.get("self_check_passed"):
                break

        if not final_self_check.get("self_check_passed") and not final_error_info:
            final_error_info = {
                "error_type": "public_self_check_failed",
                "message": "workspace agent exhausted repair rounds without passing public self-check",
            }
        self._write_workspace_trajectory_artifacts(
            workspace=workspace,
            task_id=task_bundle.task_id,
            model_provider=session_config.model_config.provider_name,
            model_name=resolved_model_name,
            trajectory_rounds=trajectory_rounds,
            final_status=("success" if final_self_check.get("self_check_passed") else "failed"),
            final_error_info=final_error_info,
        )
        transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        return RunRecord(
            run_id=session_config.run_id,
            task_id=task_bundle.task_id,
            provider=self.provider_name,
            provider_session_id=session_config.run_id,
            model_provider=session_config.model_config.provider_name,
            model_name=resolved_model_name,
            env_hash=session_config.env_hash,
            skills_active=list(active_skills),
            workspace_root=workspace,
            artifacts_uri=str(output_dir),
            transcript_uri=str(transcript_path),
            stdout=final_completed_stdout,
            stderr=final_completed_stderr,
            runtime_seconds=final_runtime,
            metadata={
                "execution_mode": session_config.execution_mode,
                "returncode": final_completed.returncode,
                "command": ["python3", final_metadata.get("entrypoint", "work/main.py")],
                "model_provider_kind": safe_model_config.get("kind", ""),
                "api_key_env": session_config.model_config.api_key_env,
                "runtime_layout": task_spec.get("runtime_layout") or {},
                "runtime_root": str(workspace),
                "work_dir": str(work_dir),
                "output_dir": str(output_dir),
                "checkpoints_dir": str(checkpoints_dir),
                "task_python_executable": str(env.get("MYEVOSKILL_TASK_PYTHON") or sys.executable),
                "task_env_hash": str(
                    env.get("MYEVOSKILL_TASK_ENV_HASH") or session_config.env_hash
                ),
                "task_env_backend": str(env.get("MYEVOSKILL_TASK_ENV_BACKEND") or ""),
                "task_env_ready": bool(
                    ((task_spec.get("runtime_env") or {}) if isinstance(task_spec, Mapping) else {}).get(
                        "ready",
                        False,
                    )
                ),
                "agent_mode": "workspace_edit",
                "iteration_count": len(commands_run),
                "files_written": final_files_written,
                "commands_run": commands_run,
                "public_self_check_status": final_self_check,
                "workspace_write_violations": workspace_write_violations,
                **_timeout_metadata(
                    policy,
                    timed_out=final_timed_out,
                    timeout_scope="solver_execution" if final_timed_out else "",
                ),
                "prompt_contract_version": self.LEGACY_WORKSPACE_PROMPT_VERSION,
                "command_history_summary": ["workspace_generate", "workspace_run"],
                **final_metadata,
            },
        )

    def _load_manifest_output_contract(self, task_bundle: TaskBundle) -> dict[str, Any]:
        task_spec = load_task_spec(task_bundle)
        task_contract = self._load_public_task_contract(
            task_spec=task_spec,
            root=task_bundle.public_bundle_dir,
        )
        if task_contract:
            required_outputs = output_requirements_from_contract(task_contract)
            if not required_outputs:
                raise RuntimeError(
                    f"task '{task_bundle.task_id}' is missing task_contract.public.json output"
                )
            return {"required_outputs": required_outputs}
        output_contract = dict(task_spec.get("output_contract") or {})
        required_outputs = list(output_contract.get("required_outputs") or [])
        if not required_outputs:
            raise RuntimeError(
                f"task '{task_bundle.task_id}' is missing manifest output_contract.required_outputs"
            )
        return output_contract

    def _render_output_requirements(self, output_contract: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        required_outputs = list(output_contract.get("required_outputs") or [])
        for index, item in enumerate(required_outputs, start=1):
            path_value = str(item.get("path", "") or "").strip()
            format_value = str(item.get("format", "") or "").strip()
            field_specs = [
                dict(field or {})
                for field in item.get("fields", []) or []
                if isinstance(field, Mapping)
            ]
            required_fields = [str(field) for field in item.get("required_fields", []) or []]
            if field_specs and not required_fields:
                required_fields = [
                    str(field.get("name", "") or "")
                    for field in field_specs
                    if str(field.get("name", "") or "")
                ]
            description = f"Output requirement {index}: produce `{path_value}`"
            if format_value:
                description += f" ({format_value})"
            lines.append(description)
            if required_fields:
                lines.append("   Required fields: " + ", ".join(required_fields))
            for field in field_specs:
                field_name = str(field.get("name", "") or "").strip()
                if not field_name:
                    continue
                field_shape = list(field.get("shape", []) or [])
                field_dtype = str(field.get("dtype", "") or "").strip()
                details = f"   Field `{field_name}`"
                if field_dtype:
                    details += f" dtype={field_dtype}"
                if field_shape:
                    details += f" shape={field_shape}"
                lines.append(details)
        return lines

    def _workspace_entrypoint_command(self) -> str:
        return "python work/main.py"

    def _workspace_self_eval_command(self) -> str:
        return "python evaluation/self_eval.py"

    def _resolve_workspace_prompt_mode(self, session_config: ExecutorSessionConfig) -> str:
        raw_value = session_config.provider_extras.get(
            "workspace_prompt_mode",
            self.DEFAULT_WORKSPACE_PROMPT_MODE,
        )
        normalized = str(raw_value or self.DEFAULT_WORKSPACE_PROMPT_MODE).strip().lower()
        aliases = {
            "inline": "inline_public_content",
            "inline_public_content": "inline_public_content",
            "semantic": "semantic_only",
            "semantic_only": "semantic_only",
        }
        if normalized not in aliases:
            raise RuntimeError(
                "workspace_prompt_mode must be one of: inline_public_content, semantic_only"
            )
        return aliases[normalized]

    def _resolve_workspace_completion_policy(self, session_config: ExecutorSessionConfig) -> str:
        raw_value = session_config.provider_extras.get(
            "workspace_completion_policy",
            self.DEFAULT_COMPLETION_POLICY,
        )
        normalized = str(raw_value or self.DEFAULT_COMPLETION_POLICY).strip().lower()
        aliases = {
            "sdk_result_message": "sdk_result_message",
            "result_message": "sdk_result_message",
            "main_success_output_contract": "main_success_output_contract",
        }
        if normalized not in aliases:
            raise RuntimeError(
                "workspace_completion_policy must be one of: "
                "sdk_result_message, main_success_output_contract"
            )
        return aliases[normalized]

    def _resolve_workspace_max_iterations(
        self,
        session_config: ExecutorSessionConfig,
        *,
        default_limit: int,
    ) -> Optional[int]:
        raw_value = session_config.provider_extras.get(
            "max_workspace_iterations",
            default_limit,
        )
        if raw_value is None:
            return None
        try:
            numeric = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("max_workspace_iterations must be an integer") from exc
        if numeric <= 0:
            return None
        return numeric

    def _workspace_visible_contract_path(self, relative_path: str) -> str:
        normalized = str(relative_path or "").replace("\\", "/").strip()
        if normalized == "README.md":
            return "README_public.md"
        return normalized

    def _load_public_task_contract(
        self,
        *,
        task_spec: Mapping[str, Any],
        root: Path,
    ) -> dict[str, Any]:
        try:
            return load_public_task_contract_from_root(root, task_spec)
        except FileNotFoundError:
            return {}

    def _load_workspace_registration_contract(self, task_bundle: TaskBundle) -> dict[str, Any]:
        if not task_bundle.task_spec_path.exists():
            return {}
        try:
            task_spec = json.loads(task_bundle.task_spec_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        public_contract = self._load_public_task_contract(
            task_spec=task_spec,
            root=task_bundle.public_bundle_dir,
        )
        if public_contract:
            return public_contract
        contract = task_spec.get("registration_contract")
        if isinstance(contract, dict):
            return contract
        return {}

    def _workspace_execution_conventions(
        self,
        registration_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        if "execution" in registration_contract:
            execution = task_contract_execution(registration_contract)
            files_by_id = {
                str(item.get("id", "") or ""): dict(item or {})
                for item in registration_contract.get("files", []) or []
                if isinstance(item, Mapping) and str(item.get("id", "") or "")
            }
            return {
                "read_first": [
                    self._workspace_visible_contract_path(
                        str(files_by_id[item].get("path", "") or "")
                    )
                    for item in execution.get("read_first", []) or []
                    if item in files_by_id
                ],
                "readable_paths": [
                    self._workspace_visible_contract_path(
                        str(files_by_id[item].get("path", "") or "")
                    )
                    for item in execution.get("readable_files", []) or []
                    if item in files_by_id
                ],
                "writable_paths": list(execution.get("writable_paths", []) or []),
                "entrypoint": self._workspace_visible_contract_path(
                    str(execution.get("entrypoint", "") or "").strip()
                ),
            }
        execution = dict(registration_contract.get("execution_conventions") or {})
        return {
            "read_first": [
                self._workspace_visible_contract_path(item)
                for item in execution.get("read_first", []) or []
                if str(item or "").strip()
            ],
            "readable_paths": [
                self._workspace_visible_contract_path(item)
                for item in execution.get("readable_paths", []) or []
                if str(item or "").strip()
            ],
            "writable_paths": [
                str(item).replace("\\", "/").strip()
                for item in execution.get("writable_paths", []) or []
                if str(item or "").strip()
            ],
            "entrypoint": self._workspace_visible_contract_path(
                str(execution.get("entrypoint", "") or "").strip()
            ),
        }

    def _build_workspace_file_semantics(
        self,
        listed_files: Sequence[str],
        registration_contract: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        available = {str(item) for item in listed_files}
        lines: list[str] = []
        seen_paths: set[str] = set()
        if isinstance(registration_contract, Mapping):
            if "files" in registration_contract:
                for raw_file in registration_contract.get("files", []) or []:
                    if not isinstance(raw_file, Mapping):
                        continue
                    if str(raw_file.get("visibility", "") or "") != "public":
                        continue
                    workspace_path = self._workspace_visible_contract_path(
                        str(raw_file.get("path", "") or "")
                    )
                    if not workspace_path or workspace_path not in available:
                        continue
                    semantics = str(raw_file.get("semantics", "") or "").strip()
                    if not semantics:
                        continue
                    lines.append(f"- `{workspace_path}`: {semantics}")
                    seen_paths.add(workspace_path)
            for raw_resource in registration_contract.get("resources", []) or []:
                if not isinstance(raw_resource, Mapping):
                    continue
                if str(raw_resource.get("visibility", "") or "") != "public":
                    continue
                workspace_path = self._workspace_visible_contract_path(
                    str(raw_resource.get("path", "") or "")
                )
                if not workspace_path or workspace_path not in available:
                    continue
                semantics = str(raw_resource.get("semantics", "") or "").strip()
                authority = str(raw_resource.get("authority", "") or "").strip()
                if not semantics:
                    continue
                if authority == "authoritative":
                    semantics = f"authoritative source: {semantics}"
                lines.append(f"- `{workspace_path}`: {semantics}")
                seen_paths.add(workspace_path)
        fallback_semantics = [
            ("README_public.md", "authoritative task description and public constraints."),
            ("task_contract.public.json", "authoritative interface contract for files, outputs, and metrics."),
            ("data/raw_data.npz", "public observation data, including measured spectra and axes."),
            ("data/meta_data.json", "physical parameters, constants, and experiment configuration."),
            ("requirements.txt", "available dependency set for the workspace."),
            ("evaluation/self_eval.py", "public check script available inside the workspace."),
        ]
        for relative_path, description in fallback_semantics:
            if relative_path in available and relative_path not in seen_paths:
                lines.append(f"- `{relative_path}`: {description}")
        return lines

    def _public_self_eval_read_roots(self, tool_policy: dict[str, Any]) -> list[str]:
        return [str(item) for item in tool_policy.get("read_roots", []) if str(item) != "evaluation"]

    def _resolve_public_eval_alignments(self, task_spec: dict[str, Any]) -> list[dict[str, Any]]:
        public_eval_spec = dict(task_spec.get("public_eval_spec") or {})
        alignments: list[dict[str, Any]] = []
        for item in public_eval_spec.get("alignments", []) or []:
            if not isinstance(item, dict):
                continue
            output_path = str(item.get("output_path", "") or item.get("path", "") or "").strip()
            field_name = str(item.get("field", "") or "").strip()
            source_path = str(item.get("source_path", "") or "").strip()
            source_field = str(item.get("source_field", "") or "").strip()
            mode = str(item.get("mode", "shape") or "shape").strip().lower()
            if not output_path or not field_name or not source_path or not source_field:
                continue
            if mode not in {"shape", "allclose"}:
                mode = "shape"
            alignments.append(
                {
                    "output_path": output_path,
                    "field": field_name,
                    "source_path": source_path,
                    "source_field": source_field,
                    "mode": mode,
                    "rtol": float(item.get("rtol", 1e-6) or 1e-6),
                    "atol": float(item.get("atol", 1e-6) or 1e-6),
                }
            )
        return alignments

    def _build_public_self_eval_spec(
        self,
        *,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        tool_policy: dict[str, Any],
    ) -> dict[str, Any]:
        task_contract = self._load_public_task_contract(
            task_spec=task_spec,
            root=runtime_paths["runtime_root"],
        )
        required_outputs = self._required_output_specs(task_spec, task_contract=task_contract)

        readonly_roots = self._public_self_eval_read_roots(tool_policy)
        readonly_snapshot = self._snapshot_roots(runtime_paths["runtime_root"], readonly_roots)
        return {
            "version": 1,
            "entrypoint": self._workspace_entrypoint_command(),
            "self_eval_command": self._workspace_self_eval_command(),
            "required_outputs": required_outputs,
            "metric_input_checks": output_metric_input_checks(task_contract) if task_contract else [],
            "readonly_roots": readonly_roots,
            "readonly_snapshot": readonly_snapshot,
            "alignments": self._resolve_public_eval_alignments(task_spec),
        }

    def _render_public_self_eval_script(self) -> str:
        return (
            '"""Harness-generated public self evaluation."""\n'
            "from __future__ import annotations\n\n"
            "import hashlib\n"
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import numpy as np\n\n"
            "def _hash_path(path: Path) -> str:\n"
            "    digest = hashlib.sha256()\n"
            "    digest.update(path.read_bytes())\n"
            "    return digest.hexdigest()\n\n"
            "def _snapshot_roots(workspace_root: Path, roots: list[str]) -> dict[str, str]:\n"
            "    snapshot: dict[str, str] = {}\n"
            "    for root_name in roots:\n"
            "        target = workspace_root / root_name\n"
            "        if target.is_file():\n"
            "            snapshot[root_name] = _hash_path(target)\n"
            "            continue\n"
            "        if not target.exists():\n"
            "            continue\n"
            "        for child in sorted(path for path in target.rglob('*') if path.is_file()):\n"
            "            rel = child.relative_to(workspace_root).as_posix()\n"
            "            snapshot[rel] = _hash_path(child)\n"
            "    return snapshot\n\n"
            "def _load_npz(path: Path) -> dict[str, np.ndarray]:\n"
            "    with np.load(path, allow_pickle=False) as payload:\n"
            "        return {name: np.asarray(payload[name]) for name in payload.files}\n\n"
            "def _apply_slice(value: np.ndarray, slice_selector):\n"
            "    if slice_selector is None:\n"
            "        return value\n"
            "    start, stop, step = list(slice_selector)[:3]\n"
            "    return np.asarray(value)[slice(start, stop, step)]\n\n"
            "def _apply_selectors(value: np.ndarray, selectors: dict[str, object]) -> np.ndarray:\n"
            "    result = np.asarray(value)\n"
            "    slice_selector = selectors.get('slice')\n"
            "    if slice_selector is not None:\n"
            "        result = _apply_slice(result, slice_selector)\n"
            "    if selectors.get('index') is not None:\n"
            "        result = np.asarray(result)[int(selectors['index'])]\n"
            "    if bool(selectors.get('squeeze', False)):\n"
            "        result = np.squeeze(result)\n"
            "    return np.asarray(result)\n\n"
            "def _apply_preprocess(value: np.ndarray, preprocess: str) -> np.ndarray:\n"
            "    mode = str(preprocess or 'identity').strip().lower() or 'identity'\n"
            "    if mode == 'identity':\n"
            "        return np.asarray(value)\n"
            "    if mode == 'abs':\n"
            "        return np.abs(value)\n"
            "    if mode == 'angle':\n"
            "        return np.angle(value)\n"
            "    if mode == 'real':\n"
            "        return np.real(value)\n"
            "    if mode == 'imag':\n"
            "        return np.imag(value)\n"
            "    raise ValueError(f'unsupported preprocess: {preprocess!r}')\n\n"
            "def main() -> int:\n"
            "    workspace_root = Path(__file__).resolve().parents[1]\n"
            "    spec_path = Path(__file__).with_name('self_eval_spec.json')\n"
            "    spec = json.loads(spec_path.read_text(encoding='utf-8'))\n"
            "    checks: list[dict[str, object]] = []\n"
            "    errors: list[str] = []\n"
            "    warnings: list[str] = []\n"
            "    output_cache: dict[str, dict[str, np.ndarray]] = {}\n"
            "    source_cache: dict[str, dict[str, np.ndarray]] = {}\n\n"
            "    readonly_before = dict(spec.get('readonly_snapshot') or {})\n"
            "    readonly_after = _snapshot_roots(workspace_root, list(spec.get('readonly_roots') or []))\n"
            "    readonly_violations = [\n"
            "        key for key in sorted(set(readonly_before) | set(readonly_after))\n"
            "        if readonly_before.get(key) != readonly_after.get(key)\n"
            "    ]\n"
            "    for item in readonly_violations:\n"
            "        errors.append(f'read-only path modified: {item}')\n\n"
            "    for item in list(spec.get('required_outputs') or []):\n"
            "        relative_path = str(item.get('path', '') or '').strip()\n"
            "        if not relative_path:\n"
            "            continue\n"
            "        output_path = workspace_root / relative_path\n"
            "        if not output_path.exists():\n"
            "            errors.append(f'missing output artifact: {relative_path}')\n"
            "            continue\n"
            "        checks.append({'name': f'output_exists:{relative_path}', 'ok': True})\n"
            "        if str(item.get('format', '') or '').strip().lower() != 'npz':\n"
            "            continue\n"
            "        try:\n"
            "            payload = _load_npz(output_path)\n"
            "        except Exception:\n"
            "            errors.append(f'unreadable artifact: {relative_path}')\n"
            "            continue\n"
            "        output_cache[relative_path] = payload\n"
            "        field_specs = [dict(field) for field in item.get('fields', []) or []]\n"
            "        required_fields = [str(field) for field in item.get('required_fields', []) or [] if str(field or '').strip()]\n"
            "        if field_specs:\n"
            "            required_fields = list(dict.fromkeys([\n"
            "                *required_fields,\n"
            "                *[str(field.get('name', '') or '') for field in field_specs if str(field.get('name', '') or '')],\n"
            "            ]))\n"
            "        for field in required_fields:\n"
            "            if field not in payload:\n"
            "                errors.append(f'missing required field: {field}')\n"
            "            else:\n"
            "                checks.append({'name': f'field_present:{relative_path}:{field}', 'ok': True})\n"
            "        numeric_fields = [str(field) for field in item.get('numeric_fields', []) or [] if str(field or '').strip()]\n"
            "        same_shape_fields = [str(field) for field in item.get('same_shape_fields', []) or [] if str(field or '').strip()]\n"
            "        for field_spec in field_specs:\n"
            "            field = str(field_spec.get('name', '') or '')\n"
            "            if not field or field not in payload:\n"
            "                continue\n"
            "            value = np.asarray(payload[field])\n"
            "            if not np.issubdtype(value.dtype, np.number):\n"
            "                errors.append(f'non-numeric field: {field}')\n"
            "                continue\n"
            "            checks.append({'name': f'numeric_field:{relative_path}:{field}', 'ok': True})\n"
            "            if np.any(~np.isfinite(np.asarray(value, dtype=np.complex128).view(np.float64))):\n"
            "                errors.append(f'nan_or_inf field: {field}')\n"
            "            expected_shape = list(field_spec.get('shape', []) or [])\n"
            "            if expected_shape and list(value.shape) != expected_shape:\n"
            "                errors.append(\n"
            "                    f'invalid shape for field {field}: expected {expected_shape}, observed {list(value.shape)}'\n"
            "                )\n"
            "            expected_dtype = str(field_spec.get('dtype', '') or '')\n"
            "            if expected_dtype and str(value.dtype) != expected_dtype:\n"
            "                errors.append(\n"
            "                    f'invalid dtype for field {field}: expected {expected_dtype}, observed {value.dtype}'\n"
            "                )\n\n"
            "        detected_shapes = {}\n"
            "        for field in numeric_fields:\n"
            "            if field not in payload:\n"
            "                continue\n"
            "            value = np.asarray(payload[field])\n"
            "            if not np.issubdtype(value.dtype, np.number):\n"
            "                errors.append(f'non-numeric field: {field}')\n"
            "                continue\n"
            "            checks.append({'name': f'numeric_field:{relative_path}:{field}', 'ok': True})\n"
            "            if np.any(~np.isfinite(np.asarray(value, dtype=np.complex128).view(np.float64))):\n"
            "                errors.append(f'nan_or_inf field: {field}')\n"
            "            if field in same_shape_fields:\n"
            "                detected_shapes[field] = tuple(value.shape)\n"
            "        if detected_shapes and len(set(detected_shapes.values())) > 1:\n"
            "            errors.append('required fields have inconsistent shapes')\n\n"
            "    for rule in list(spec.get('metric_input_checks') or []):\n"
            "        output_path = str(rule.get('output_path', '') or '').strip()\n"
            "        field = str(rule.get('field', '') or '').strip()\n"
            "        metric_name = str(rule.get('metric_name', '') or '').strip()\n"
            "        input_name = str(rule.get('input_name', '') or '').strip()\n"
            "        expected_shape = list(rule.get('expected_shape', []) or [])\n"
            "        payload = output_cache.get(output_path)\n"
            "        if payload is None or field not in payload:\n"
            "            continue\n"
            "        try:\n"
            "            value = _apply_selectors(np.asarray(payload[field]), dict(rule.get('selectors') or {}))\n"
            "            value = _apply_preprocess(value, str(rule.get('preprocess', 'identity') or 'identity'))\n"
            "        except Exception as exc:\n"
            "            errors.append(f'metric input resolution failed: {metric_name}:{input_name} ({exc})')\n"
            "            continue\n"
            "        observed_shape = list(np.asarray(value).shape)\n"
            "        if observed_shape != expected_shape:\n"
            "            errors.append(\n"
            "                f'metric input shape mismatch: {metric_name}:{input_name} expected {expected_shape}, observed {observed_shape}'\n"
            "            )\n"
            "        else:\n"
            "            checks.append({'name': f'metric_input_shape:{metric_name}:{input_name}', 'ok': True})\n\n"
            "    for rule in list(spec.get('alignments') or []):\n"
            "        output_path = str(rule.get('output_path', '') or rule.get('path', '') or '').strip()\n"
            "        field = str(rule.get('field', '') or '').strip()\n"
            "        source_path = str(rule.get('source_path', '') or '').strip()\n"
            "        source_field = str(rule.get('source_field', '') or '').strip()\n"
            "        mode = str(rule.get('mode', 'shape') or 'shape').strip().lower()\n"
            "        if not output_path or not field or not source_path or not source_field:\n"
            "            continue\n"
            "        payload = output_cache.get(output_path)\n"
            "        if payload is None or field not in payload:\n"
            "            continue\n"
            "        source_full_path = workspace_root / source_path\n"
            "        try:\n"
            "            source_payload = source_cache.setdefault(source_path, _load_npz(source_full_path))\n"
            "        except Exception:\n"
            "            errors.append(f'unreadable public source: {source_path}')\n"
            "            continue\n"
            "        if source_field not in source_payload:\n"
            "            errors.append(f'missing public source field: {source_path}:{source_field}')\n"
            "            continue\n"
            "        output_value = np.asarray(payload[field])\n"
            "        source_value = np.asarray(source_payload[source_field])\n"
            "        if mode == 'allclose':\n"
            "            if output_value.shape != source_value.shape or not np.allclose(\n"
            "                output_value,\n"
            "                source_value,\n"
            "                rtol=float(rule.get('rtol', 1e-6) or 1e-6),\n"
            "                atol=float(rule.get('atol', 1e-6) or 1e-6),\n"
            "            ):\n"
            "                errors.append(\n"
            "                    f'public alignment failed: {output_path}:{field} != {source_path}:{source_field}'\n"
            "                )\n"
            "            else:\n"
            "                checks.append({'name': f'allclose:{output_path}:{field}', 'ok': True})\n"
            "        else:\n"
            "            if output_value.shape != source_value.shape:\n"
            "                errors.append(\n"
            "                    f'public shape alignment failed: {output_path}:{field} vs {source_path}:{source_field}'\n"
            "                )\n"
            "            else:\n"
            "                checks.append({'name': f'shape:{output_path}:{field}', 'ok': True})\n\n"
            "    result = {\n"
            "        'passed': not errors,\n"
            "        'checks': checks,\n"
            "        'errors': list(dict.fromkeys(errors)),\n"
            "        'warnings': list(dict.fromkeys(warnings)),\n"
            "    }\n"
            "    print(json.dumps(result, indent=2, sort_keys=True))\n"
            "    return 0 if result['passed'] else 1\n\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n"
        )

    def _install_public_self_eval_runtime(
        self,
        *,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        tool_policy: dict[str, Any],
    ) -> dict[str, Any]:
        evaluation_dir = runtime_paths["runtime_root"] / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        spec = self._build_public_self_eval_spec(
            task_spec=task_spec,
            runtime_paths=runtime_paths,
            tool_policy=tool_policy,
        )
        (evaluation_dir / "self_eval_spec.json").write_text(
            json.dumps(spec, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (evaluation_dir / "self_eval.py").write_text(
            self._render_public_self_eval_script(),
            encoding="utf-8",
        )
        return spec

    def _required_output_specs(
        self,
        task_spec: dict[str, Any],
        *,
        task_contract: Optional[Mapping[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        task_contract = dict(task_contract or {})
        if task_contract:
            return output_requirements_from_contract(task_contract)
        primary_output = str(task_spec.get("primary_output_path", "") or "").strip()
        if primary_output:
            return [{"path": primary_output, "format": "npz"}]
        output_contract = dict(task_spec.get("output_contract") or {})
        required_outputs = list(output_contract.get("required_outputs") or [])
        if required_outputs:
            proxy_spec = dict(task_spec.get("proxy_spec") or {})
            merged_outputs: list[dict[str, Any]] = []
            for item in required_outputs:
                normalized = dict(item or {})
                normalized.setdefault("format", str(proxy_spec.get("output_dtype", "") or ""))
                if not normalized.get("required_fields"):
                    normalized["required_fields"] = list(proxy_spec.get("required_fields") or [])
                if not normalized.get("numeric_fields"):
                    normalized["numeric_fields"] = list(proxy_spec.get("numeric_fields") or [])
                if not normalized.get("same_shape_fields"):
                    normalized["same_shape_fields"] = list(proxy_spec.get("same_shape_fields") or [])
                merged_outputs.append(normalized)
            return merged_outputs
        proxy_spec = dict(task_spec.get("proxy_spec") or {})
        primary_output = str(proxy_spec.get("primary_output", "") or "").strip()
        if not primary_output:
            return []
        return [
            {
                "path": primary_output,
                "format": str(proxy_spec.get("output_dtype", "") or ""),
                "required_fields": list(proxy_spec.get("required_fields") or []),
                "numeric_fields": list(proxy_spec.get("numeric_fields") or []),
                "same_shape_fields": list(proxy_spec.get("same_shape_fields") or []),
            }
        ]

    def _inspect_output_contract_state(
        self,
        *,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
    ) -> dict[str, Any]:
        task_contract = self._load_public_task_contract(
            task_spec=task_spec,
            root=runtime_paths["runtime_root"],
        )
        required_outputs = self._required_output_specs(task_spec, task_contract=task_contract)
        primary_output = str(required_outputs[0].get("path", "") or "").strip() if required_outputs else ""

        existing_outputs: list[str] = []
        missing_outputs: list[str] = []
        missing_output_fields: list[str] = []
        schema_warnings: list[str] = []
        primary_output_exists = False

        for item in required_outputs:
            relative_path = str(item.get("path", "") or "").strip()
            if not relative_path:
                continue
            output_path = runtime_paths["runtime_root"] / relative_path
            if output_path.exists():
                existing_outputs.append(relative_path)
            else:
                missing_outputs.append(relative_path)
                if relative_path == primary_output:
                    primary_output_exists = False
                continue
            if relative_path == primary_output:
                primary_output_exists = True

            if str(item.get("format", "") or "").strip().lower() != "npz":
                continue
            try:
                with np.load(output_path, allow_pickle=False) as payload:
                    if task_contract:
                        output_validation = validate_output_payload_against_contract(payload, task_contract)
                        item_missing_fields = list(output_validation.get("missing_fields", []))
                        schema_warnings.extend(list(output_validation.get("warnings", [])))
                        for check in output_metric_input_checks(task_contract):
                            if str(check.get("output_path", "") or "").strip() != relative_path:
                                continue
                            try:
                                resolve_metric_input_value(
                                    runtime_paths["runtime_root"],
                                    task_contract,
                                    {
                                        "source": "output",
                                        "field": str(check.get("field", "") or ""),
                                        "selectors": dict(check.get("selectors") or {}),
                                        "preprocess": str(check.get("preprocess", "identity") or "identity"),
                                        "expected_shape": list(check.get("expected_shape", []) or []),
                                    },
                                    output_payload=payload,
                                )
                            except Exception as exc:
                                schema_warnings.append(
                                    f"metric input resolution failed: {check.get('metric_name', '')}:{check.get('input_name', '')} ({exc})"
                                )
                    else:
                        required_fields = [str(field) for field in item.get("required_fields") or []]
                        numeric_fields = [str(field) for field in item.get("numeric_fields") or []]
                        same_shape_fields = [str(field) for field in item.get("same_shape_fields") or []]
                        item_missing_fields = []
                        for field in required_fields:
                            if field not in payload.files:
                                item_missing_fields.append(field)
                        shapes: list[tuple[int, ...]] = []
                        for field in numeric_fields:
                            if field not in payload.files:
                                continue
                            value = np.asarray(payload[field])
                            if not np.issubdtype(value.dtype, np.number):
                                schema_warnings.append(f"non-numeric field: {field}")
                                continue
                            if np.any(~np.isfinite(value)):
                                schema_warnings.append(f"nan_or_inf field: {field}")
                            if field in same_shape_fields:
                                shapes.append(tuple(value.shape))
                        if shapes and any(shape != shapes[0] for shape in shapes[1:]):
                            schema_warnings.append("required fields have inconsistent shapes")
            except Exception:
                schema_warnings.append(f"unreadable artifact: {relative_path}")
                continue

            missing_output_fields.extend(item_missing_fields)

        schema_warnings = sorted(dict.fromkeys(schema_warnings))
        missing_outputs = sorted(dict.fromkeys(missing_outputs))
        missing_output_fields = sorted(dict.fromkeys(missing_output_fields))
        existing_outputs = sorted(dict.fromkeys(existing_outputs))
        if missing_output_fields:
            schema_warnings.append(
                "missing required fields: " + ", ".join(missing_output_fields)
            )
        output_schema_valid = not missing_outputs and not missing_output_fields and not schema_warnings
        return {
            "primary_output": primary_output,
            "primary_output_exists": primary_output_exists,
            "required_outputs": [str(item.get("path", "") or "").strip() for item in required_outputs if str(item.get("path", "") or "").strip()],
            "existing_outputs": existing_outputs,
            "missing_outputs": missing_outputs,
            "missing_output_fields": missing_output_fields,
            "schema_warnings": schema_warnings,
            "output_schema_valid": output_schema_valid,
            "self_check_output_ready": output_schema_valid,
        }

    def _evaluate_workspace_completion(
        self,
        *,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        completed: subprocess.CompletedProcess[str],
        timed_out: bool,
        entrypoint: str,
        env: dict[str, str],
        timeout_seconds: int,
        completion_policy: str,
    ) -> dict[str, Any]:
        if completion_policy != "main_success_output_contract":
            result = self._public_self_check(
                task_spec=task_spec,
                runtime_paths=runtime_paths,
                completed=completed,
                timed_out=timed_out,
                entrypoint=entrypoint,
                env=env,
                timeout_seconds=timeout_seconds,
            )
            result.setdefault(
                "output_contract_satisfied",
                bool(
                    completed.returncode == 0
                    and not timed_out
                    and not result.get("missing_outputs")
                    and not result.get("missing_output_fields")
                    and not result.get("schema_warnings")
                ),
            )
            return result

        output_state = self._inspect_output_contract_state(
            task_spec=task_spec,
            runtime_paths=runtime_paths,
        )
        output_exists = output_state["primary_output_exists"]
        schema_warnings = list(output_state["schema_warnings"])
        if output_state["missing_outputs"] and not output_exists:
            schema_warnings = ["missing output artifact", *schema_warnings]
        output_contract_satisfied = bool(output_state.get("output_schema_valid", False))
        completed_stdout = self._completed_process_text(completed.stdout)
        completed_stderr = self._completed_process_text(completed.stderr)
        return {
            "run_succeeded": completed.returncode == 0 and not timed_out,
            "returncode": completed.returncode,
            "timed_out": timed_out,
            "entrypoint": entrypoint,
            "output_exists": output_exists,
            "missing_outputs": list(output_state["missing_outputs"]),
            "missing_output_fields": list(output_state["missing_output_fields"]),
            "schema_warnings": schema_warnings,
            "stdout_tail": completed_stdout[-2000:],
            "stderr_tail": completed_stderr[-2000:],
            "public_self_eval_command": self._workspace_self_eval_command(),
            "public_self_eval_returncode": None,
            "public_self_eval_timed_out": False,
            "public_self_eval_parse_error": "",
            "public_self_eval_passed": False,
            "public_self_eval_checks": [],
            "public_self_eval_errors": [],
            "public_self_eval_warnings": [],
            "public_self_eval_stdout": "",
            "public_self_eval_stderr": "",
            "output_contract_satisfied": output_contract_satisfied,
            "self_check_passed": (
                completed.returncode == 0 and not timed_out and output_contract_satisfied
            ),
        }

    def _build_external_completion_callback(
        self,
        *,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        workspace: Path,
        tool_policy: dict[str, Any],
        readonly_before: dict[str, str],
        completion_policy: str,
        timeout_seconds: int,
    ):
        if completion_policy != "main_success_output_contract":
            return None

        entrypoint_command = self._workspace_entrypoint_command()

        def _callback(serialized_messages: list[Any]) -> Optional[dict[str, Any]]:
            entrypoint_result = self._find_latest_bash_command_result(
                serialized_messages,
                entrypoint_command,
            )
            if not entrypoint_result or not entrypoint_result.get("succeeded", False):
                return None
            readonly_after = self._snapshot_roots(workspace, tool_policy["read_roots"])
            readonly_violations = self._detect_snapshot_mutations(readonly_before, readonly_after)
            if readonly_violations:
                return None
            completed = subprocess.CompletedProcess(
                ["python", "work/main.py"],
                int(entrypoint_result.get("returncode", 0) or 0),
                str(entrypoint_result.get("stdout", "") or ""),
                str(entrypoint_result.get("stderr", "") or ""),
            )
            completion_check = self._evaluate_workspace_completion(
                task_spec=task_spec,
                runtime_paths=runtime_paths,
                completed=completed,
                timed_out=False,
                entrypoint="work/main.py",
                env={},
                timeout_seconds=timeout_seconds,
                completion_policy=completion_policy,
            )
            if not completion_check.get("self_check_passed", False):
                return None
            return {
                "summary": self._default_workspace_summary(
                    task_spec,
                    solver_summary=(
                        "python work/main.py succeeded and produced the required output artifacts."
                    ),
                ),
                "completion_source": "external_output_contract",
                "result_text": str(entrypoint_result.get("stdout", "") or ""),
                "diagnostics": {
                    "external_completion": {
                        "entrypoint_result": dict(entrypoint_result),
                        "completion_check": dict(completion_check),
                        "readonly_violations": list(readonly_violations),
                    }
                },
            }

        return _callback

    def _sanitize_workspace_visible_payload(
        self,
        value: Any,
        *,
        workspace_root: Path,
    ) -> Any:
        workspace_root = Path(workspace_root).resolve()
        workspace_root_win = str(workspace_root)
        workspace_root_posix = workspace_root.as_posix()
        replacements = [
            (workspace_root_win + "\\", ".\\"),
            (workspace_root_win + "/", "./"),
            (workspace_root_win, "."),
            (workspace_root_posix + "/", "./"),
            (workspace_root_posix, "."),
        ]
        if isinstance(value, Mapping):
            return {
                key: self._sanitize_workspace_visible_payload(item, workspace_root=workspace_root)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._sanitize_workspace_visible_payload(item, workspace_root=workspace_root)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                self._sanitize_workspace_visible_payload(item, workspace_root=workspace_root)
                for item in value
            )
        if isinstance(value, str):
            sanitized = value
            for source, target in replacements:
                sanitized = sanitized.replace(source, target)
            return sanitized
        return value

    def _workspace_prompt_root_label(self) -> str:
        return "."

    def _workspace_prompt_plan_rule_items(self) -> list[str]:
        return [
            f"Before any code edits or solver runs in this round, create or update `{self.WORKSPACE_PLAN_PATH}` in the workspace output directory.",
            f"On later rounds, revise `{self.WORKSPACE_PLAN_PATH}` first so it reflects the new repair plan before making more changes.",
            f"Keep `{self.WORKSPACE_PLAN_PATH}` short and concrete: planned steps, expected outputs, and the next validation you will run.",
        ]

    def _workspace_prompt_plan_workflow_items(self) -> list[str]:
        return [
            f"Create or update `{self.WORKSPACE_PLAN_PATH}` first with the plan for this round.",
            "Then make the planned code or execution changes.",
        ]

    def _build_workspace_agent_prompt(
        self,
        task_bundle: TaskBundle,
        active_skills: Sequence[str],
        *,
        workspace_root: Path,
        tool_policy: dict[str, Any],
        stop_oracle: str,
        prompt_mode: str = DEFAULT_WORKSPACE_PROMPT_MODE,
        completion_policy: str = DEFAULT_COMPLETION_POLICY,
    ) -> str:
        output_contract = self._load_manifest_output_contract(task_bundle)
        registration_contract = self._load_workspace_registration_contract(task_bundle)
        execution_conventions = self._workspace_execution_conventions(registration_contract)
        listed_files = sorted(
            path.relative_to(task_bundle.public_bundle_dir).as_posix()
            for path in task_bundle.public_bundle_dir.rglob("*")
            if path.is_file()
        )
        if (workspace_root / "evaluation" / "self_eval.py").exists():
            listed_files = sorted(dict.fromkeys([*listed_files, "evaluation/self_eval.py"]))

        inline_public_content = prompt_mode == "inline_public_content"
        external_output_contract = (
            completion_policy == "main_success_output_contract"
            and stop_oracle not in {"submit_tool", "hidden_judge_submit"}
        )
        readme = (
            task_bundle.readme_public_path.read_text(encoding="utf-8")
            if inline_public_content
            and task_bundle.readme_public_path
            and task_bundle.readme_public_path.exists()
            else ""
        )
        requirements = (
            self._safe_read_text(task_bundle.public_bundle_dir / "requirements.txt")
            if inline_public_content
            else ""
        )
        network_access_enabled = bool(tool_policy.get("network_access", False))
        write_tool_rule = (
            "Prefer the `Write` tool to create or modify `work/` source files such as "
            "`work/src/*.py` and `work/main.py`; do not rely on Bash for routine source edits."
        )
        bash_usage_rule = (
            "Use Bash mainly to run programs, inspect files, debug, and perform workspace-local "
            "file operations that stay inside writable roots."
        )
        bash_boundary_rule = self._workspace_bash_boundary_rule(network_access_enabled)
        path_boundary_rule = (
            "Absolute or relative paths are both acceptable when they resolve inside the "
            "workspace; write targets must stay under `work/`, `output/`, or `checkpoints/`."
        )
        workspace_root_label = self._workspace_prompt_root_label()
        write_tool_workflow = (
            "Create and update the solver under `work/src/*.py` plus `work/main.py`, "
            "preferably with the `Write` tool."
        )
        meta_data = (
            self._safe_read_text(task_bundle.public_bundle_dir / "data" / "meta_data.json")
            if inline_public_content
            else ""
        )
        network_rule_items = self._workspace_network_research_rule_items(network_access_enabled)
        network_workflow_items = self._workspace_network_workflow_items(network_access_enabled)
        plan_rule_items = self._workspace_prompt_plan_rule_items()
        plan_workflow_items = self._workspace_prompt_plan_workflow_items()
        if stop_oracle == "hidden_judge_submit":
            important_rules = [
                "STOP CONDITION: After `python work/main.py` succeeds and you have a candidate output, call `submit_result(...)`. If it returns `{ \"status\": \"fail\" }`, keep debugging locally and try again. If it returns `{ \"status\": \"pass\" }`, immediately return the final structured summary and stop all further tool use.",
                *self._number_prompt_items(
                    [
                        "Read `README_public.md` first and treat it as the authoritative task specification.",
                        "Use relative paths from the current workspace root; do not guess hardcoded absolute paths.",
                        *self._render_output_requirements(output_contract),
                        "Use only packages already available in `requirements.txt`.",
                        "Explore public inputs with tools instead of assuming file contents.",
                        *network_rule_items,
                        *plan_rule_items,
                        "You may write only under `work/`, `output/`, and `checkpoints/`.",
                        write_tool_rule,
                        bash_usage_rule,
                        bash_boundary_rule,
                        path_boundary_rule,
                        "Do not modify public inputs such as `README_public.md`, `requirements.txt`, `data/`, `evaluation/`, or `public_bundle/`.",
                        "Run the solver from the workspace root with `python work/main.py`.",
                        "If `python work/main.py` fails, debug locally within the same session and rerun it as needed.",
                        "Use `submit_result(...)` as the authoritative hidden-judge stop oracle. It returns only `{ \"status\": \"pass\" }` or `{ \"status\": \"fail\" }`.",
                        "A hidden-judge `fail` does not include metric details, tags, or reasons. Diagnose locally from the workspace and retry.",
                        "Do not stop after a hidden-judge `fail`; keep iterating until `submit_result(...)` returns `{ \"status\": \"pass\" }` or the harness budget ends.",
                        "Use `submit_result(...)` with keys: `solver_summary`, `declared_outputs`, `assumptions`, `files_written`, `commands_run`.",
                        "After `submit_result(...)` returns `{ \"status\": \"pass\" }`, return the same structured summary and stop immediately.",
                        "Organize the solver as a multi-file workspace project under `work/src/*.py` plus `work/main.py`.",
                    ]
                ),
            ]
            workflow_lines = self._number_prompt_items(
                [
                    "Read `README_public.md`, then inspect the public data and metadata you need.",
                    *network_workflow_items,
                    *plan_workflow_items,
                    write_tool_workflow,
                    "Run `python work/main.py` from the workspace root.",
                    "If the run fails, debug locally in this same session and rerun `python work/main.py`.",
                    "When you have a candidate output, call `submit_result(...)` immediately.",
                    "If `submit_result(...)` returns `{ \"status\": \"fail\" }`, continue debugging locally and resubmit after making changes.",
                    "If `submit_result(...)` returns `{ \"status\": \"pass\" }`, immediately return the structured summary and stop.",
                ]
            )
            intro_lines = [
                "You are operating inside a constrained Claude workspace harness.",
                "Your job is to complete the workspace contract and stop cleanly, not to keep improving the science indefinitely.",
                "The harness exposes a hidden-judge submission tool that returns only pass or fail.",
                "If the hidden judge returns fail, continue iterating locally in this same workspace session.",
                f"workspace_root={workspace_root_label}",
                f"cwd={workspace_root_label}",
                "README_public.md is the authoritative task specification narrative, and task_contract.public.json is the authoritative interface contract.",
                "Use relative paths first from the current workspace root.",
                "Explore the workspace with the provided tools instead of assuming file contents.",
                f"Active skills: {', '.join(active_skills) if active_skills else '(none)'}",
                f"Writable roots: {', '.join(tool_policy['write_roots'])}",
                f"Public files: {', '.join(listed_files)}",
                "Available harness tools: `submit_result(...)`",
            ]
        elif stop_oracle == "submit_tool":
            important_rules = [
                "STOP CONDITION: After `python work/main.py` succeeds and the required output artifacts exist, your next step must be `check_ready()`, not more analysis or optimization. When `check_ready()` returns `ready=true`, call `submit_result(...)`. If `submit_result(...)` is accepted, immediately return the final structured summary and stop all further tool use.",
                *self._number_prompt_items(
                    [
                        "Read `README_public.md` first and treat it as the authoritative task specification.",
                        "Use relative paths from the current workspace root; do not guess hardcoded absolute paths.",
                        *self._render_output_requirements(output_contract),
                        "Use only packages already available in `requirements.txt`.",
                        "Explore public inputs with tools instead of assuming file contents.",
                        *network_rule_items,
                        *plan_rule_items,
                        "You may write only under `work/`, `output/`, and `checkpoints/`.",
                        write_tool_rule,
                        bash_usage_rule,
                        bash_boundary_rule,
                        path_boundary_rule,
                        "Do not modify public inputs such as `README_public.md`, `requirements.txt`, `data/`, `evaluation/`, or `public_bundle/`.",
                        "Run the solver from the workspace root with `python work/main.py`.",
                        "If `python work/main.py` fails, debug locally within the same session and rerun it as needed.",
                        "Use `check_ready()` as the authoritative public completion oracle. Completion is defined by the harness contract, not by your subjective confidence or by hidden metrics.",
                        "If the run succeeded and required outputs exist, call `check_ready()` immediately even if you think the model could still be improved.",
                        "Do not spend extra turns on scientific interpretation, parameter sweeps, scaling analysis, plotting, or physics refinement after the public contract is already satisfied.",
                        "Use `submit_result(...)` to submit the final summary with keys: `solver_summary`, `declared_outputs`, `assumptions`, `files_written`, `commands_run`.",
                        "After `submit_result(...)` is accepted, return the same structured summary and stop immediately.",
                        "Organize the solver as a multi-file workspace project under `work/src/*.py` plus `work/main.py`.",
                    ]
                ),
            ]
            workflow_lines = self._number_prompt_items(
                [
                    "Read `README_public.md`, then inspect the public data and metadata you need.",
                    *network_workflow_items,
                    *plan_workflow_items,
                    write_tool_workflow,
                    "Run `python work/main.py` from the workspace root.",
                    "If the run fails, debug locally in this same session and rerun `python work/main.py`.",
                    "Once the run succeeds and required outputs exist, call `check_ready()` immediately.",
                    "Do not do more analysis or improvement before `check_ready()` once the contract artifacts already exist.",
                    "When `check_ready()` returns `ready=true`, call `submit_result(...)` with the final summary.",
                    "If `submit_result(...)` is accepted, immediately return the same structured summary and stop.",
                ]
            )
            intro_lines = [
                "You are operating inside a constrained Claude workspace harness.",
                "Your job is to complete the workspace contract and stop cleanly, not to keep improving the science indefinitely.",
                "The harness, via `check_ready()` and `submit_result(...)`, decides when the task is complete.",
                "If `python work/main.py` succeeds and the required outputs exist, do not keep researching or refining before calling `check_ready()`.",
                f"workspace_root={workspace_root_label}",
                f"cwd={workspace_root_label}",
                "README_public.md is the authoritative task specification narrative, and task_contract.public.json is the authoritative interface contract.",
                "Use relative paths first from the current workspace root.",
                "Explore the workspace with the provided tools instead of assuming file contents.",
                f"Active skills: {', '.join(active_skills) if active_skills else '(none)'}",
                f"Writable roots: {', '.join(tool_policy['write_roots'])}",
                f"Public files: {', '.join(listed_files)}",
                "Available harness tools: `check_ready()`, `submit_result(...)`",
            ]
        elif external_output_contract:
            important_rules = [
                "STOP CONDITION: After `python work/main.py` succeeds and the required outputs exist, immediately return the final structured summary and stop all further tool use.",
                *self._number_prompt_items(
                    [
                        "Read `README_public.md` first and treat it as the authoritative task specification.",
                        "Use relative paths from the current workspace root; do not guess hardcoded absolute paths.",
                        *self._render_output_requirements(output_contract),
                        "Use only packages already available in `requirements.txt`.",
                        "Explore public inputs with tools instead of assuming file contents.",
                        *network_rule_items,
                        *plan_rule_items,
                        "You may write only under `work/`, `output/`, and `checkpoints/`.",
                        write_tool_rule,
                        bash_usage_rule,
                        bash_boundary_rule,
                        path_boundary_rule,
                        "Do not modify public inputs such as `README_public.md`, `requirements.txt`, `data/`, `evaluation/`, or `public_bundle/`.",
                        "Run the solver from the workspace root with `python work/main.py`.",
                        "If `python work/main.py` fails, debug locally within the same session and rerun it as needed.",
                        "If `python work/main.py` succeeds and the required outputs exist, immediately return the structured summary with keys: `solver_summary`, `declared_outputs`, `assumptions`, `files_written`, `commands_run`.",
                        "Do not spend extra turns on scientific interpretation, parameter sweeps, plotting, or physics refinement after the output contract is already satisfied.",
                        "Organize the solver as a multi-file workspace project under `work/src/*.py` plus `work/main.py`.",
                    ]
                ),
            ]
            workflow_lines = self._number_prompt_items(
                [
                    "Read `README_public.md`, then inspect the public data and metadata you need.",
                    *network_workflow_items,
                    *plan_workflow_items,
                    write_tool_workflow,
                    "Run `python work/main.py` from the workspace root.",
                    "If the run fails, debug locally in this same session and rerun `python work/main.py`.",
                    "When `python work/main.py` succeeds and the required outputs exist, immediately return the structured summary and stop.",
                ]
            )
            intro_lines = [
                "You are operating inside a constrained Claude workspace harness.",
                "Your job is to complete the workspace contract and stop cleanly, not to keep improving the science indefinitely.",
                "The harness decides completion from `python work/main.py` plus the required output artifacts.",
                "If `python work/main.py` succeeds and the required outputs exist, your next action must be to return the final structured summary, not more analysis or optimization.",
                f"workspace_root={workspace_root_label}",
                f"cwd={workspace_root_label}",
                "README_public.md is the authoritative task specification narrative, and task_contract.public.json is the authoritative interface contract.",
                "Use relative paths first from the current workspace root.",
                "Explore the workspace with the provided tools instead of assuming file contents.",
                f"Active skills: {', '.join(active_skills) if active_skills else '(none)'}",
                f"Writable roots: {', '.join(tool_policy['write_roots'])}",
                f"Public files: {', '.join(listed_files)}",
            ]
        else:
            important_rules = [
                "STOP CONDITION: After `python work/main.py` succeeds, run `python evaluation/self_eval.py`. If it exits successfully, immediately return the final structured summary and stop all further tool use.",
                *self._number_prompt_items(
                    [
                        "Read `README_public.md` first and treat it as the authoritative task specification.",
                        "Use relative paths from the current workspace root; do not guess hardcoded absolute paths.",
                        *self._render_output_requirements(output_contract),
                        "Use only packages already available in `requirements.txt`.",
                        "Explore public inputs with tools instead of assuming file contents.",
                        *network_rule_items,
                        *plan_rule_items,
                        "You may write only under `work/`, `output/`, and `checkpoints/`.",
                        write_tool_rule,
                        bash_usage_rule,
                        bash_boundary_rule,
                        path_boundary_rule,
                        "Do not modify public inputs such as `README_public.md`, `requirements.txt`, `data/`, `evaluation/`, or `public_bundle/`.",
                        "Run the solver from the workspace root with `python work/main.py`.",
                        "After the solver run, use `python evaluation/self_eval.py` as the public completion oracle.",
                        "If `python evaluation/self_eval.py` fails, debug locally within the same session and rerun the solver and self-eval as needed.",
                        "If `python evaluation/self_eval.py` passes, immediately return the structured summary with keys: `solver_summary`, `declared_outputs`, `assumptions`, `files_written`, `commands_run`.",
                        "Do not spend extra turns on scientific interpretation, parameter sweeps, plotting, or physics refinement after `python evaluation/self_eval.py` passes.",
                        "Organize the solver as a multi-file workspace project under `work/src/*.py` plus `work/main.py`.",
                    ]
                ),
            ]
            workflow_lines = self._number_prompt_items(
                [
                    "Read `README_public.md`, then inspect the public data and metadata you need.",
                    *network_workflow_items,
                    *plan_workflow_items,
                    write_tool_workflow,
                    "Run `python work/main.py` from the workspace root.",
                    "If the run fails, debug locally in this same session and rerun `python work/main.py`.",
                    "When `python work/main.py` succeeds, run `python evaluation/self_eval.py` immediately.",
                    "If `python evaluation/self_eval.py` fails, repair locally and rerun the solver and self-eval.",
                    "If `python evaluation/self_eval.py` succeeds, immediately return the structured summary and stop.",
                ]
            )
            intro_lines = [
                "You are operating inside a constrained Claude workspace harness.",
                "Your job is to complete the workspace contract and stop cleanly, not to keep improving the science indefinitely.",
                "The harness-generated file `evaluation/self_eval.py` is the public completion oracle.",
                "If `python work/main.py` succeeds, your next action must be `python evaluation/self_eval.py`, not more analysis or optimization.",
                f"workspace_root={workspace_root_label}",
                f"cwd={workspace_root_label}",
                "README_public.md is the authoritative task specification narrative, and task_contract.public.json is the authoritative interface contract.",
                "Use relative paths first from the current workspace root.",
                "Explore the workspace with the provided tools instead of assuming file contents.",
                f"Active skills: {', '.join(active_skills) if active_skills else '(none)'}",
                f"Writable roots: {', '.join(tool_policy['write_roots'])}",
                f"Public files: {', '.join(listed_files)}",
                "Use `python evaluation/self_eval.py` after the solver run. When it passes, immediately return the structured summary and stop.",
            ]
        prompt_lines = [*intro_lines]
        if inline_public_content:
            prompt_lines.extend(
                [
                    "",
                    "## Problem Description",
                    "The task description below comes from README_public.md.",
                    "",
                    readme,
                    "",
                    "## Data Specification",
                    "The public metadata below comes from data/meta_data.json.",
                    "",
                    meta_data or "(data/meta_data.json not provided)",
                    "",
                    "Available packages from requirements.txt:",
                    requirements or "(requirements.txt not provided)",
                ]
            )
        else:
            prompt_lines.extend(
                [
                    "",
                    "## Contract",
                    (
                        "Read first: "
                        + ", ".join(execution_conventions["read_first"])
                        if execution_conventions.get("read_first")
                        else "Read first: README_public.md, task_contract.public.json"
                    ),
                    (
                        "Readable paths: "
                        + ", ".join(execution_conventions["readable_paths"])
                        if execution_conventions.get("readable_paths")
                        else "Readable paths: " + ", ".join(listed_files)
                    ),
                    (
                        "Writable paths: "
                        + ", ".join(execution_conventions["writable_paths"])
                        if execution_conventions.get("writable_paths")
                        else "Writable paths: " + ", ".join(tool_policy["write_roots"])
                    ),
                    (
                        "Entrypoint: python "
                        + execution_conventions["entrypoint"]
                        if execution_conventions.get("entrypoint")
                        else "Entrypoint: python work/main.py"
                    ),
                    "",
                    "## Public Files",
                    *(
                        self._build_workspace_file_semantics(
                            listed_files,
                            registration_contract=registration_contract,
                        )
                        or ["- Public files are available in the workspace."]
                    ),
                ]
            )
        prompt_lines.extend(
            [
                "",
                "## IMPORTANT_RULES",
                *important_rules,
                "",
                "## Recommended Workflow",
                *workflow_lines,
            ]
        )
        return "\n".join(prompt_lines)

    def _number_prompt_items(self, items: Sequence[str]) -> list[str]:
        return [f"{index}. {item}" for index, item in enumerate(items, start=1)]

    def _workspace_bash_boundary_rule(self, network_access_enabled: bool) -> str:
        prohibited_side_effects = [
            "package installation",
            "version control",
            "privilege escalation",
            "system control",
            "killing processes",
        ]
        if not network_access_enabled:
            prohibited_side_effects.insert(0, "network access")
        return (
            "Any Bash command is acceptable if it stays inside the workspace, avoids read-only "
            "roots, and does not trigger prohibited external side effects such as "
            + ", ".join(prohibited_side_effects)
            + "."
        )

    def _workspace_network_research_rule_items(
        self, network_access_enabled: bool
    ) -> list[str]:
        if not network_access_enabled:
            return []
        return [
            "After reading `README_public.md` and `task_contract.public.json`, do at least one brief external search that is directly relevant to the task method before writing code.",
            "Use a paper-first search order: prioritize papers, project pages, and paper abstract pages; consult official or author implementations only when the papers do not provide enough implementation detail.",
            "Prefer Claude SDK web tools such as `WebSearch` and `WebFetch` for external lookup instead of Bash-based web access.",
            "Keep external research bounded: use it to form a solver strategy, not to conduct an open-ended literature review. If a short search does not surface high-value sources, proceed with the best justified local assumptions.",
        ]

    def _workspace_network_workflow_items(
        self, network_access_enabled: bool
    ) -> list[str]:
        if not network_access_enabled:
            return []
        return [
            "Before writing code, use `WebSearch` and `WebFetch` to do a brief paper-first search for the task method, then extract the solver strategy you will implement.",
        ]

    def _build_legacy_workspace_agent_prompt(

        self,
        task_bundle: TaskBundle,
        active_skills: Sequence[str],
    ) -> str:
        compile_report: Dict[str, Any] = {}
        if task_bundle.compile_report_path.exists():
            compile_report = json.loads(task_bundle.compile_report_path.read_text(encoding="utf-8"))
        final_contract = dict(compile_report.get("final_public_contract", {}))
        readme = (
            task_bundle.readme_public_path.read_text(encoding="utf-8")
            if task_bundle.readme_public_path and task_bundle.readme_public_path.exists()
            else ""
        )
        requirements = self._safe_read_text(task_bundle.public_bundle_dir / "requirements.txt")
        metrics = self._safe_read_text(task_bundle.public_bundle_dir / "evaluation" / "metrics.json")
        meta_data = self._safe_read_text(task_bundle.public_bundle_dir / "data" / "meta_data.json")
        listed_files = sorted(
            path.relative_to(task_bundle.public_bundle_dir).as_posix()
            for path in task_bundle.public_bundle_dir.rglob("*")
            if path.is_file()
        )
        return "\n".join(
            [
                "You are a Claude-style workspace coding agent for scientific tasks.",
                "README_public.md is the authoritative task specification narrative, and task_contract.public.json is the authoritative interface contract. Read both before writing code.",
                "Return a single JSON object with top-level keys files, entrypoint, declared_outputs, assumptions, solver_summary.",
                "All file writes must stay under work/.",
                f"Active skills: {', '.join(active_skills) if active_skills else '(none)'}",
                f"Public files: {', '.join(listed_files)}",
                f"Public contract: {json.dumps(final_contract, sort_keys=True)}",
                "",
                "README_public.md:",
                readme,
                "",
                "requirements.txt:",
                requirements,
                "",
                "evaluation/metrics.json:",
                metrics,
                "",
                "data/meta_data.json:",
                meta_data,
            ]
        )

    def _compose_legacy_workspace_round_prompt(
        self,
        *,
        base_prompt: str,
        round_index: int,
        previous_feedback: Optional[dict[str, Any]],
        previous_summary: Optional[dict[str, Any]],
    ) -> str:
        if round_index == 1 or previous_feedback is None:
            return base_prompt
        feedback_text = json.dumps(previous_feedback, indent=2, sort_keys=True)
        summary_text = json.dumps(previous_summary or {}, indent=2, sort_keys=True)
        return "\n".join(
            [
                base_prompt,
                "",
                f"Round {round_index} repair instructions:",
                "The previous attempt did not satisfy public self-check.",
                "Use the feedback below to repair the workspace while preserving valid behavior.",
                "Return a full replacement file set for work/.",
                "Do not use or speculate about hidden judge results.",
                "",
                "Previous summary:",
                summary_text,
                "",
                "Public self-check feedback:",
                feedback_text,
            ]
        )

    def _compose_workspace_sdk_round_prompt(
        self,
        *,
        base_prompt: str,
        round_index: int,
        previous_feedback: Optional[dict[str, Any]],
        previous_summary: Optional[dict[str, Any]],
    ) -> str:
        if round_index == 1 or previous_feedback is None:
            return base_prompt
        feedback_text = json.dumps(previous_feedback, indent=2, sort_keys=True, ensure_ascii=False)
        summary_text = json.dumps(previous_summary or {}, indent=2, sort_keys=True, ensure_ascii=False)
        if str(previous_feedback.get("failure_mode", "") or "") == "workspace_plan_required":
            return "\n".join(
                [
                    base_prompt,
                    "",
                    f"Round {round_index} repair instructions:",
                    "The previous attempt did not update the required workspace execution plan before making changes.",
                    f"Update `{self.WORKSPACE_PLAN_PATH}` first in this round, then continue with code or execution changes.",
                    "Keep the plan concise and actionable, then follow it.",
                    "",
                    "Previous summary:",
                    summary_text,
                    "",
                    "Plan feedback:",
                    feedback_text,
                ]
            )
        if str(previous_feedback.get("submission_status", "") or "") == "not_submitted":
            return "\n".join(
                [
                    base_prompt,
                    "",
                    f"Round {round_index} repair instructions:",
                    "The previous attempt ended without calling `submit_result(...)`.",
                    "After you have a candidate output, you must call `submit_result(...)` before stopping.",
                    "",
                    "Previous summary:",
                    summary_text,
                    "",
                    "Submission feedback:",
                    feedback_text,
                ]
            )
        if str(previous_feedback.get("submission_status", "") or "") == "fail":
            return "\n".join(
                [
                    base_prompt,
                    "",
                    f"Round {round_index} repair instructions:",
                    "The previous candidate was submitted to the hidden judge and returned fail.",
                    "No hidden metric details or failure reasons are available.",
                    "Use the prior summary plus local workspace evidence to diagnose, repair, and resubmit.",
                    "Call `submit_result(...)` again after producing a revised candidate output.",
                    "Do not use or speculate about hidden judge results.",
                    "",
                    "Previous summary:",
                    summary_text,
                    "",
                    "Submission feedback:",
                    feedback_text,
                ]
            )
        return "\n".join(
            [
                base_prompt,
                "",
                f"Round {round_index} repair instructions:",
                "The previous attempt hit harness Bash policy denials before execution completed.",
                "Use the feedback below to repair the workspace while preserving valid behavior.",
                "Keep Bash activity inside the workspace root and writable roots only.",
                "Return a full replacement file set for work/.",
                "Do not use or speculate about hidden judge results.",
                "",
                "Previous summary:",
                summary_text,
                "",
                "Policy denial feedback:",
                feedback_text,
            ]
        )

    def _build_workspace_policy_feedback(
        self,
        policy_denials: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "self_check_passed": False,
            "failure_mode": "bash_policy_denial",
            "policy_denials": [dict(item) for item in policy_denials],
            "repair_instruction": self.WORKSPACE_POLICY_REPAIR_INSTRUCTION,
        }

    def _build_hidden_judge_retry_feedback(self) -> dict[str, Any]:
        return {"submission_status": "fail"}

    def _build_hidden_judge_submission_required_feedback(self) -> dict[str, Any]:
        return {
            "submission_status": "not_submitted",
            "repair_instruction": "After producing a candidate output, call submit_result(...) before stopping.",
        }

    def _build_workspace_plan_feedback(self) -> dict[str, Any]:
        return {
            "failure_mode": "workspace_plan_required",
            "required_plan_path": self.WORKSPACE_PLAN_PATH,
            "repair_instruction": (
                f"Create or update `{self.WORKSPACE_PLAN_PATH}` first in each round before code edits or solver execution."
            ),
        }

    def _generate_workspace_sdk_response(
        self,
        *,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        workspace: Path,
        prompt: str,
        model_timeout_seconds: int,
        round_index: int,
        tool_policy: dict[str, Any],
        readonly_before: dict[str, str],
        stop_oracle: str,
        completion_policy: str,
        sdk_env: Optional[dict[str, str]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if "mock_claude_sdk_response" in session_config.provider_extras:
            payload = self._resolve_mock_sdk_response(
                session_config.provider_extras["mock_claude_sdk_response"],
                round_index,
            )
            self._apply_mock_workspace_files(workspace, payload.get("files") or {})
            submission_state = self._merge_submission_state(
                self._empty_submission_state(),
                payload.get("submission_state"),
            )
            private_submission_state = self._merge_private_submission_state(
                self._empty_private_submission_state(),
                payload.get("private_submission_state"),
            )
            if payload.get("mock_execution_error"):
                diagnostics = dict(payload.get("sdk_diagnostics") or {})
                if "sdk_result" not in diagnostics and payload.get("sdk_result_metadata") is not None:
                    diagnostics["sdk_result"] = dict(payload.get("sdk_result_metadata") or {})
                diagnostics["submission_state"] = dict(submission_state)
                raise ClaudeSDKExecutionError(
                    str(payload.get("error_message") or "mock Claude SDK execution error"),
                    error_type=str(payload.get("error_type") or "missing_result_message"),
                    sdk_messages=list(payload.get("sdk_messages") or []),
                    result_text=str(payload.get("result_text") or ""),
                    diagnostics=diagnostics,
                    private_state=private_submission_state,
                )
            summary = self._coerce_sdk_summary(payload)
            diagnostics = dict(payload.get("sdk_diagnostics") or {})
            diagnostics.setdefault(
                "sdk_result",
                {
                    "subtype": str(payload.get("subtype", "success") or "success"),
                    "stop_reason": str(payload.get("stop_reason", "") or ""),
                    "is_error": False,
                    "num_turns": int(payload.get("num_turns", 0) or 0),
                    "session_id": str(payload.get("session_id", "") or ""),
                },
            )
            metadata = self._build_sdk_metadata(
                sdk_messages=payload.get("sdk_messages") or [],
                parsed_summary=summary,
                bridge_mode="workspace_mock_claude_sdk",
                model_provider_kind="claude_sdk",
                raw_response_text=json.dumps(payload, indent=2, sort_keys=True),
                sdk_diagnostics=diagnostics,
                submission_state=submission_state,
            )
            metadata["vendor_session_ref"] = dict(payload.get("vendor_session_ref") or {})
            metadata["private_submission_state"] = dict(private_submission_state)
            return summary, metadata

        if importlib.util.find_spec("claude_agent_sdk") is None:
            raise RuntimeError(
                "claude_agent_sdk is required for provider 'claude-sdk'. "
                "Install the Claude Agent SDK in the active environment."
            )

        system_prompt = self._build_workspace_system_prompt(tool_policy, stop_oracle=stop_oracle)
        result = asyncio.run(
            self._execute_claude_sdk_query(
                task_bundle=task_bundle,
                session_config=session_config,
                workspace=workspace,
                prompt=prompt,
                system_prompt=system_prompt,
                model_timeout_seconds=model_timeout_seconds,
                task_spec=task_spec,
                runtime_paths=runtime_paths,
                tool_policy=tool_policy,
                readonly_before=readonly_before,
                stop_oracle=stop_oracle,
                completion_policy=completion_policy,
                sdk_env=sdk_env or {},
            )
        )
        summary = self._coerce_sdk_summary(result["summary"])
        metadata = self._build_sdk_metadata(
            sdk_messages=result["messages"],
            parsed_summary=summary,
            bridge_mode="workspace_claude_sdk",
            model_provider_kind="claude_sdk",
            raw_response_text=result["result_text"],
            sdk_diagnostics=result.get("sdk_diagnostics", {}),
            submission_state=result.get("submission_state"),
            completion_source=str(result.get("completion_source", "result_message") or "result_message"),
        )
        metadata["vendor_session_ref"] = result.get("vendor_session_ref", {})
        metadata["private_submission_state"] = dict(result.get("private_submission_state") or {})
        return summary, metadata

    def _is_claude_result_message(self, message: Any, result_message_type: Any = None) -> bool:
        if result_message_type is not None:
            return isinstance(message, result_message_type)
        return message.__class__.__name__ == "ResultMessage"

    def _normalize_protocol_failure_reason(self, error_type: str) -> str:
        normalized = str(error_type or "").strip().lower()
        if normalized in {
            "accepted_submission_missing_result_message",
            "missing_result_message",
            "request_timeout",
            "max_turns",
            "result_error",
            "invalid_structured_summary",
            "sdk_error",
            "protocol_incomplete_after_stop",
        }:
            return normalized
        if normalized == "error_max_turns":
            return "max_turns"
        if normalized in {"", "empty_result"}:
            return "missing_result_message"
        return "sdk_error"

    def _saw_protocol_stop_signal(self, diagnostics: dict[str, Any]) -> bool:
        if bool(diagnostics.get("stop_hook_seen")):
            return True
        last_message_type = str(diagnostics.get("last_message_type", "") or "")
        last_task_notification_status = str(
            diagnostics.get("last_task_notification_status", "") or ""
        ).lower()
        return (
            last_message_type == "TaskNotificationMessage"
            and last_task_notification_status in {"completed", "failed", "stopped"}
        )

    def _build_protocol_diagnostics(
        self,
        messages: Sequence[Any],
        *,
        hook_events: Sequence[dict[str, Any]],
        result_message: Any = None,
        timeout_occurred: bool,
    ) -> dict[str, Any]:
        task_lifecycle_events: list[Any] = []
        task_notification_statuses: list[str] = []
        last_message_type = ""
        last_task_notification_status = ""
        for message in messages:
            message_type = message.__class__.__name__
            last_message_type = message_type
            if message_type in {"TaskStartedMessage", "TaskProgressMessage", "TaskNotificationMessage"}:
                payload = self._json_safe(message)
                if isinstance(payload, dict):
                    payload.setdefault("message_type", message_type)
                task_lifecycle_events.append(payload)
                if message_type == "TaskNotificationMessage":
                    status = str(getattr(message, "status", "") or "")
                    if status:
                        task_notification_statuses.append(status)
                        last_task_notification_status = status
        serialized_hook_events = [self._json_safe(item) for item in hook_events]
        stop_hook_seen = any(
            isinstance(item, dict) and str(item.get("hook_event_name", "") or "") == "Stop"
            for item in serialized_hook_events
        )
        notification_hook_seen = any(
            isinstance(item, dict)
            and str(item.get("hook_event_name", "") or "") == "Notification"
            for item in serialized_hook_events
        )
        return {
            "sdk_result_seen": result_message is not None,
            "sdk_result": self._extract_sdk_result_metadata(result_message),
            "stop_hook_seen": stop_hook_seen,
            "notification_hook_seen": notification_hook_seen,
            "task_notification_statuses": task_notification_statuses,
            "task_lifecycle_events": task_lifecycle_events,
            "hook_events": serialized_hook_events,
            "message_count": len(messages),
            "last_message_type": last_message_type,
            "last_task_notification_status": last_task_notification_status,
            "timeout_occurred": bool(timeout_occurred),
        }

    async def _consume_claude_response(
        self,
        message_stream: AsyncIterator[Any],
        *,
        total_timeout_seconds: int,
        hook_events: Optional[Sequence[dict[str, Any]]] = None,
        result_message_type: Any = None,
        submission_state: Optional[dict[str, Any]] = None,
        external_completion_callback: Optional[Any] = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        iterator = message_stream.__aiter__()
        messages: list[Any] = []
        serialized_messages: list[Any] = []
        result_text = ""
        hook_events = list(hook_events or [])

        while True:
            remaining_total = total_timeout_seconds - (time.monotonic() - start)
            if remaining_total <= 0:
                diagnostics = self._build_protocol_diagnostics(
                    messages,
                    hook_events=hook_events,
                    timeout_occurred=True,
                )
                diagnostics["submission_state"] = dict(self._merge_submission_state({}, submission_state))
                failure_reason = self._resolve_missing_result_failure_reason(
                    diagnostics,
                    submission_state=submission_state,
                    timeout_occurred=True,
                )
                raise ClaudeSDKExecutionError(
                    f"Claude SDK query timed out after {total_timeout_seconds} seconds",
                    error_type=failure_reason,
                    sdk_messages=serialized_messages,
                    result_text=result_text,
                    diagnostics=diagnostics,
                )
            try:
                message = await asyncio.wait_for(
                    iterator.__anext__(), timeout=max(remaining_total, 0.01)
                )
            except StopAsyncIteration:
                diagnostics = self._build_protocol_diagnostics(
                    messages,
                    hook_events=hook_events,
                    timeout_occurred=False,
                )
                diagnostics["submission_state"] = dict(self._merge_submission_state({}, submission_state))
                failure_reason = self._resolve_missing_result_failure_reason(
                    diagnostics,
                    submission_state=submission_state,
                    timeout_occurred=False,
                )
                error_message = (
                    "Claude SDK accepted submit_result(...) but did not return a ResultMessage"
                    if failure_reason == "accepted_submission_missing_result_message"
                    else (
                        "Claude SDK response stream ended after stop/task completion signals but without a ResultMessage"
                        if failure_reason == "protocol_incomplete_after_stop"
                        else "Claude SDK response stream ended without a ResultMessage"
                    )
                )
                raise ClaudeSDKExecutionError(
                    error_message,
                    error_type=failure_reason,
                    sdk_messages=serialized_messages,
                    result_text=result_text,
                    diagnostics=diagnostics,
                )
            except asyncio.TimeoutError as exc:
                diagnostics = self._build_protocol_diagnostics(
                    messages,
                    hook_events=hook_events,
                    timeout_occurred=True,
                )
                diagnostics["submission_state"] = dict(self._merge_submission_state({}, submission_state))
                failure_reason = self._resolve_missing_result_failure_reason(
                    diagnostics,
                    submission_state=submission_state,
                    timeout_occurred=True,
                )
                raise ClaudeSDKExecutionError(
                    f"Claude SDK query timed out after {total_timeout_seconds} seconds",
                    error_type=failure_reason,
                    sdk_messages=serialized_messages,
                    result_text=result_text,
                    diagnostics=diagnostics,
                ) from exc

            messages.append(message)
            serialized_messages = self._serialize_sdk_messages(messages)
            if self._is_claude_result_message(message, result_message_type=result_message_type):
                result_text = self._extract_sdk_result_text(message)
                diagnostics = self._build_protocol_diagnostics(
                    messages,
                    hook_events=hook_events,
                    result_message=message,
                    timeout_occurred=False,
                )
                diagnostics["submission_state"] = dict(self._merge_submission_state({}, submission_state))
                try:
                    summary_payload = self._coerce_sdk_summary(
                        self._extract_sdk_summary(message, serialized_messages)
                    )
                except ClaudeSDKExecutionError as exc:
                    raise ClaudeSDKExecutionError(
                        str(exc),
                        error_type=self._normalize_protocol_failure_reason(exc.error_type),
                        sdk_messages=serialized_messages,
                        result_text=result_text,
                        diagnostics=diagnostics,
                    ) from exc
                except RuntimeError as exc:
                    raise ClaudeSDKExecutionError(
                        str(exc),
                        error_type="invalid_structured_summary",
                        sdk_messages=serialized_messages,
                        result_text=result_text,
                        diagnostics=diagnostics,
                    ) from exc
                return {
                    "messages": serialized_messages,
                    "summary": summary_payload,
                    "result_text": result_text,
                    "sdk_diagnostics": diagnostics,
                    "completion_source": "result_message",
                }

            current_result_text = self._extract_sdk_result_text(message)
            if current_result_text.strip():
                result_text = current_result_text
            if external_completion_callback is not None:
                external_completion = external_completion_callback(serialized_messages)
                if external_completion is not None:
                    diagnostics = self._build_protocol_diagnostics(
                        messages,
                        hook_events=hook_events,
                        timeout_occurred=False,
                    )
                    diagnostics["submission_state"] = dict(
                        self._merge_submission_state({}, submission_state)
                    )
                    for key, value in dict(external_completion.get("diagnostics") or {}).items():
                        diagnostics[key] = self._json_safe(value)
                    return {
                        "messages": serialized_messages,
                        "summary": self._coerce_sdk_summary(external_completion.get("summary") or {}),
                        "result_text": str(external_completion.get("result_text", "") or result_text),
                        "sdk_diagnostics": diagnostics,
                        "completion_source": str(
                            external_completion.get("completion_source", "external_output_contract")
                            or "external_output_contract"
                        ),
                    }

    def _build_claude_sdk_hooks(self, hook_events: list[dict[str, Any]]) -> dict[str, Any]:
        def _make_hook(event_name: str) -> Any:
            async def _callback(input_data, tool_use_id, context):
                hook_events.append(
                    {
                        "hook_event_name": event_name,
                        "tool_use_id": str(tool_use_id or ""),
                        "input": self._json_safe(input_data),
                    }
                )
                return {"continue_": True}

            return _make_sdk_hook_matcher(hooks=[_callback])

        return {
            "Stop": [_make_hook("Stop")],
            "Notification": [_make_hook("Notification")],
        }

    def build_workspace_sdk_hooks(
        self,
        hook_events: list[dict[str, Any]],
        *,
        workspace_root: Path,
        tool_policy: dict[str, Any],
    ) -> dict[str, Any]:
        hooks = dict(self._build_claude_sdk_hooks(hook_events))
        workspace_root = Path(workspace_root).resolve()

        async def _pretool_bash_callback(input_data, tool_use_id, _context):
            event_payload = self._json_safe(input_data)
            command = str((event_payload or {}).get("tool_input", {}).get("command", "") or "").strip()
            violations = self._validate_bash_commands([command], workspace_root, tool_policy)
            event_record = {
                "hook_event_name": "PreToolUse",
                "tool_use_id": str(tool_use_id or ""),
                "tool_name": "Bash",
                "command": command,
                "input": event_payload,
                "violations": [dict(item) for item in violations],
            }
            if not violations:
                event_record["permission_decision"] = "allow"
                hook_events.append(event_record)
                return {
                    "continue_": True,
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                    },
                }
            detail = (
                "workspace Bash command blocked by harness policy; keep commands inside "
                f"workspace_root and writable roots only. violations: {self._format_workspace_violations(violations)}"
            )
            event_record["permission_decision"] = "deny"
            event_record["permission_decision_reason"] = detail
            hook_events.append(event_record)
            return {
                "continue_": True,
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": detail,
                },
                "systemMessage": detail,
            }

        pretool_hooks = list(hooks.get("PreToolUse", []))
        pretool_hooks.append(_make_sdk_hook_matcher(matcher="Bash", hooks=[_pretool_bash_callback]))
        hooks["PreToolUse"] = pretool_hooks
        return hooks

    async def _execute_claude_sdk_query(
        self,
        *,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        workspace: Path,
        prompt: str,
        system_prompt: dict[str, Any],
        model_timeout_seconds: int,
        tool_policy: dict[str, Any],
        readonly_before: dict[str, str],
        stop_oracle: str,
        completion_policy: str,
        sdk_env: dict[str, str],
    ) -> dict[str, Any]:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        from claude_agent_sdk.types import ResultMessage

        hook_events: list[dict[str, Any]] = []
        submission_state = self._empty_submission_state()
        private_submission_state = self._empty_private_submission_state()
        mcp_servers: dict[str, Any] = {}
        if stop_oracle == "submit_tool":
            mcp_servers = self._build_submission_mcp_servers(
                task_spec=task_spec,
                runtime_paths=runtime_paths,
                workspace=workspace,
                tool_policy=tool_policy,
                readonly_before=readonly_before,
                submission_state=submission_state,
            )
        elif stop_oracle == "hidden_judge_submit":
            mcp_servers = self._build_hidden_judge_submission_mcp_servers(
                task_bundle=task_bundle,
                session_config=session_config,
                task_spec=task_spec,
                workspace=workspace,
                submission_state=submission_state,
                private_submission_state=private_submission_state,
            )
        options_kwargs = self._build_claude_sdk_options_kwargs(
            session_config=session_config,
            workspace=workspace,
            system_prompt=system_prompt,
            stop_oracle=stop_oracle,
            tool_policy=tool_policy,
            mcp_servers=mcp_servers,
            sdk_env=sdk_env,
        )
        claude_session_id = self._new_claude_session_id()
        options_kwargs["session_id"] = claude_session_id
        options_kwargs["continue_conversation"] = False
        options_kwargs["hooks"] = self.build_workspace_sdk_hooks(
            hook_events,
            workspace_root=workspace,
            tool_policy=tool_policy,
        )
        options = ClaudeAgentOptions(**options_kwargs)
        resolved_model_name = self._resolved_model_name(session_config)
        if resolved_model_name:
            options.model = resolved_model_name
        external_completion_callback = self._build_external_completion_callback(
            task_spec=task_spec,
            runtime_paths=runtime_paths,
            workspace=workspace,
            tool_policy=tool_policy,
            readonly_before=readonly_before,
            completion_policy=completion_policy,
            timeout_seconds=model_timeout_seconds,
        )
        if stop_oracle == "hidden_judge_submit":
            external_completion_callback = None
        result: Optional[dict[str, Any]] = None
        pending_error: Optional[ClaudeSDKExecutionError] = None
        vendor_session_ref: dict[str, Any] = {
            "sdk_backend": "claude_sdk",
            "session_id": claude_session_id,
        }
        async with ClaudeSDKClient(options) as client:
            try:
                await client.query(prompt, session_id=claude_session_id)
                result = await self._consume_claude_response(
                    client.receive_response(),
                    total_timeout_seconds=model_timeout_seconds,
                    hook_events=hook_events,
                    result_message_type=ResultMessage,
                    submission_state=submission_state,
                    external_completion_callback=external_completion_callback,
                )
            except ClaudeSDKExecutionError as exc:
                diagnostics = dict(exc.diagnostics)
                diagnostics.setdefault("hook_events", [self._json_safe(item) for item in hook_events])
                diagnostics["submission_state"] = dict(self._merge_submission_state({}, submission_state))
                vendor_session_ref = self._build_vendor_session_ref(
                    client, workspace, sdk_messages=exc.sdk_messages
                )
                diagnostics["vendor_session_ref"] = vendor_session_ref
                pending_error = ClaudeSDKExecutionError(
                    str(exc),
                    error_type=self._normalize_protocol_failure_reason(exc.error_type),
                    sdk_messages=exc.sdk_messages,
                    result_text=exc.result_text,
                    diagnostics=diagnostics,
                    private_state=self._merge_private_submission_state(
                        private_submission_state,
                        getattr(exc, "private_state", {}),
                    ),
                )
            except Exception as exc:
                vendor_session_ref = self._build_vendor_session_ref(
                    client, workspace, sdk_messages=[]
                )
                pending_error = ClaudeSDKExecutionError(
                    f"Claude SDK query failed: {exc}",
                    error_type="sdk_error",
                    sdk_messages=[],
                    result_text="",
                    diagnostics={
                        "hook_events": [self._json_safe(item) for item in hook_events],
                        "vendor_session_ref": vendor_session_ref,
                        "timeout_occurred": False,
                        "sdk_result": self._extract_sdk_result_metadata(None),
                        "submission_state": dict(self._merge_submission_state({}, submission_state)),
                    },
                    private_state=dict(private_submission_state),
                )
            else:
                if result is not None and "vendor_session_ref" not in result:
                    vendor_session_ref = self._build_vendor_session_ref(
                        client, workspace, sdk_messages=result.get("messages", [])
                    )
                    result["vendor_session_ref"] = vendor_session_ref
                elif result is not None:
                    vendor_session_ref = dict(result.get("vendor_session_ref") or vendor_session_ref)
                if result is not None:
                    result.setdefault("sdk_diagnostics", {})
                    result["sdk_diagnostics"].setdefault(
                        "hook_events", [self._json_safe(item) for item in hook_events]
                    )
                    result["sdk_diagnostics"]["submission_state"] = dict(
                        self._merge_submission_state({}, submission_state)
                    )
                    result["submission_state"] = dict(
                        self._merge_submission_state({}, submission_state)
                    )
                    result["private_submission_state"] = dict(private_submission_state)

        session_cleanup = self._delete_claude_session_history(
            str(vendor_session_ref.get("session_id", "") or ""),
            workspace,
        )
        vendor_session_ref = self._attach_session_cleanup(
            vendor_session_ref,
            session_cleanup=session_cleanup,
        )
        if pending_error is not None:
            pending_error.diagnostics = dict(pending_error.diagnostics)
            pending_error.diagnostics["vendor_session_ref"] = vendor_session_ref
            raise pending_error
        if result is None:
            raise ClaudeSDKExecutionError(
                "Claude SDK query returned no result payload",
                error_type="sdk_error",
                sdk_messages=[],
                result_text="",
                diagnostics={"vendor_session_ref": vendor_session_ref},
            )
        result["vendor_session_ref"] = vendor_session_ref
        result.setdefault("sdk_diagnostics", {})
        result["sdk_diagnostics"]["vendor_session_ref"] = vendor_session_ref
        result["private_submission_state"] = dict(
            self._merge_private_submission_state(
                result.get("private_submission_state"),
                private_submission_state,
            )
        )
        return result

    def _build_workspace_system_prompt(
        self,
        tool_policy: dict[str, Any],
        *,
        stop_oracle: Optional[str] = None,
    ) -> dict[str, Any]:
        stop_oracle = str(stop_oracle or self.DEFAULT_STOP_ORACLE)
        network_access_enabled = bool(tool_policy.get("network_access", False))
        prohibited_side_effects_line = (
            "Prohibited external side effects include network access, package or environment installation, version control operations, privilege escalation, system or service control, and killing processes."
            if not network_access_enabled
            else "Prohibited external side effects still include package or environment installation, version control operations, privilege escalation, system or service control, and killing processes."
        )
        network_lines = self._workspace_system_network_lines(network_access_enabled)
        if stop_oracle == "hidden_judge_submit":
            append_lines = [
                "You are an execution agent inside a harness, not an open-ended research assistant.",
                "Work only inside the provided runtime workspace.",
                "Your primary objective is to complete the workspace contract and stop cleanly.",
                "README_public.md is the authoritative task specification narrative, and task_contract.public.json is the authoritative interface contract. Read both first.",
                "Prefer relative paths from the current working directory instead of guessed absolute paths.",
                "Use the preconfigured `python` from the workspace environment; do not switch interpreters.",
                "Allowed write roots: " + ", ".join(tool_policy["write_roots"]),
                "Prefer the `Write` tool for creating or editing source files under work/; use Bash mainly for running programs, inspection, debugging, and workspace-local file operations.",
                "Read-only roots: " + ", ".join(tool_policy["read_roots"]),
                "Any Bash command that stays inside the workspace, respects read-only roots, and avoids prohibited external side effects is allowed.",
                "Absolute or relative paths are acceptable when they resolve inside the workspace; writes must stay under "
                + ", ".join(tool_policy["write_roots"])
                + ".",
                prohibited_side_effects_line,
                *network_lines,
                "The hidden judge is available only through `submit_result(...)` and returns only pass or fail.",
                f"Before any code edits or solver runs in a round, update `{self.WORKSPACE_PLAN_PATH}` first.",
                f"On later rounds, revise `{self.WORKSPACE_PLAN_PATH}` before making more changes.",
                "Use the workspace tools to debug locally within this same session if `python work/main.py` fails.",
                "When you have a candidate output, call `submit_result(...)` with the final structured summary.",
                "If `submit_result(...)` returns `{ \"status\": \"fail\" }`, continue debugging locally and try again.",
                "If `submit_result(...)` returns `{ \"status\": \"pass\" }`, immediately return ONLY the structured summary and stop all further tool use.",
                "Do not expect hidden metrics, failure tags, or any other judge detail.",
            ]
        elif stop_oracle == "submit_tool":
            append_lines = [
                "You are an execution agent inside a harness, not an open-ended research assistant.",
                "Work only inside the provided runtime workspace.",
                "Your primary objective is to complete the public workspace contract and stop cleanly.",
                "README_public.md is the authoritative task specification narrative, and task_contract.public.json is the authoritative interface contract. Read both first.",
                "Prefer relative paths from the current working directory instead of guessed absolute paths.",
                "Use the preconfigured `python` from the workspace environment; do not switch interpreters.",
                "Allowed write roots: " + ", ".join(tool_policy["write_roots"]),
                "Prefer the `Write` tool for creating or editing source files under work/; use Bash mainly for running programs, inspection, debugging, and workspace-local file operations.",
                "Read-only roots: " + ", ".join(tool_policy["read_roots"]),
                "Any Bash command that stays inside the workspace, respects read-only roots, and avoids prohibited external side effects is allowed.",
                "Absolute or relative paths are acceptable when they resolve inside the workspace; writes must stay under "
                + ", ".join(tool_policy["write_roots"])
                + ".",
                prohibited_side_effects_line,
                *network_lines,
                "Hidden judge signals are not available.",
                f"Before any code edits or solver runs in a round, update `{self.WORKSPACE_PLAN_PATH}` first.",
                f"On later rounds, revise `{self.WORKSPACE_PLAN_PATH}` before making more changes.",
                "Use the workspace tools to debug locally within this same session if `python work/main.py` fails.",
                "Use `check_ready()` as the authoritative public completion oracle before stopping.",
                "If `python work/main.py` succeeds and the required outputs exist, your next action must be `check_ready()`, not more analysis, parameter sweeps, plotting, or model refinement.",
                "When `check_ready()` returns `ready=true`, call `submit_result(...)` with the final summary.",
                "If `submit_result(...)` is accepted, immediately return ONLY the structured summary and stop all further tool use.",
                "Do not continue scientific exploration once the public contract is satisfied; the harness decides completion, not your subjective confidence.",
            ]
        else:
            append_lines = [
                "You are an execution agent inside a harness, not an open-ended research assistant.",
                "Work only inside the provided runtime workspace.",
                "Your primary objective is to complete the public workspace contract and stop cleanly.",
                "README_public.md is the authoritative task specification narrative, and task_contract.public.json is the authoritative interface contract. Read both first.",
                "Prefer relative paths from the current working directory instead of guessed absolute paths.",
                "Use the preconfigured `python` from the workspace environment; do not switch interpreters.",
                "Allowed write roots: " + ", ".join(tool_policy["write_roots"]),
                "Prefer the `Write` tool for creating or editing source files under work/; use Bash mainly for running programs, inspection, debugging, and workspace-local file operations.",
                "Read-only roots: " + ", ".join(tool_policy["read_roots"]),
                "Any Bash command that stays inside the workspace, respects read-only roots, and avoids prohibited external side effects is allowed.",
                "Absolute or relative paths are acceptable when they resolve inside the workspace; writes must stay under "
                + ", ".join(tool_policy["write_roots"])
                + ".",
                prohibited_side_effects_line,
                *network_lines,
                "Hidden judge signals are not available.",
                f"Before any code edits or solver runs in a round, update `{self.WORKSPACE_PLAN_PATH}` first.",
                f"On later rounds, revise `{self.WORKSPACE_PLAN_PATH}` before making more changes.",
                "Use the workspace tools to debug locally within this same session if `python work/main.py` fails.",
                "Use `python evaluation/self_eval.py` as the authoritative public completion oracle before stopping.",
                "If `python work/main.py` succeeds, your next action must be `python evaluation/self_eval.py`, not more analysis, parameter sweeps, plotting, or model refinement.",
                "If `python evaluation/self_eval.py` exits successfully, immediately return ONLY the structured summary and stop all further tool use.",
                "Do not continue scientific exploration once the public evaluator passes.",
            ]
        append_text = "\n".join(append_lines)
        return {
            "type": "preset",
            "preset": "claude_code",
            "append": append_text,
        }

    def _workspace_system_network_lines(self, network_access_enabled: bool) -> list[str]:
        if not network_access_enabled:
            return ["Network access is disabled by the harness."]
        return [
            "Network access is enabled only because the harness allowed it; other prohibited external side effects remain forbidden.",
            "Before writing code, do one brief paper-first external search that is directly relevant to the task method.",
            "Prioritize papers, project pages, and paper abstract pages; consult official or author implementations only when the papers leave implementation details unclear.",
            "Prefer Claude SDK web tools such as `WebSearch` and `WebFetch` for external lookup instead of Bash-based web access.",
            "Keep external lookup bounded: use it to form a solver strategy, not to conduct an open-ended literature review. If a short search does not reveal strong sources, proceed with the best justified local assumptions.",
        ]

    def _workspace_summary_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "solver_summary": {"type": "string"},
                "declared_outputs": {"type": "array", "items": {"type": "string"}},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "files_written": {"type": "array", "items": {"type": "string"}},
                "commands_run": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "solver_summary",
                "declared_outputs",
                "assumptions",
                "files_written",
                "commands_run",
            ],
            "additionalProperties": True,
        }

    def _all_allowed_tools(
        self,
        stop_oracle: Optional[str] = None,
        *,
        tool_policy: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        stop_oracle = str(stop_oracle or self.DEFAULT_STOP_ORACLE)
        allowed_tools = list(self.DEFAULT_ALLOWED_TOOLS)
        if bool((tool_policy or {}).get("network_access", False)):
            allowed_tools.extend(self.NETWORK_ENABLED_ALLOWED_TOOLS)
        if stop_oracle == "submit_tool":
            allowed_tools.extend(self.SUBMISSION_TOOL_NAMES)
        elif stop_oracle == "hidden_judge_submit":
            allowed_tools.extend(self.HIDDEN_JUDGE_TOOL_NAMES)
        return list(dict.fromkeys(allowed_tools))

    def _all_disallowed_tools(self, *, tool_policy: Optional[dict[str, Any]] = None) -> list[str]:
        disallowed_tools = list(self.DEFAULT_DISALLOWED_TOOLS)
        if bool((tool_policy or {}).get("network_access", False)):
            disallowed_tools = [
                item
                for item in disallowed_tools
                if item not in set(self.NETWORK_ENABLED_ALLOWED_TOOLS)
            ]
        return list(dict.fromkeys(disallowed_tools))

    def _resolve_workspace_stop_oracle(self, session_config: ExecutorSessionConfig) -> str:
        raw_value = session_config.provider_extras.get(
            "workspace_stop_oracle",
            session_config.provider_extras.get("stop_oracle", self.DEFAULT_STOP_ORACLE),
        )
        normalized = str(raw_value or self.DEFAULT_STOP_ORACLE).strip().lower()
        aliases = {
            "public_self_eval": "public_self_eval",
            "self_eval": "public_self_eval",
            "submit_tool": "submit_tool",
            "submit": "submit_tool",
            "hidden_judge_submit": "hidden_judge_submit",
            "hidden_submit": "hidden_judge_submit",
        }
        if normalized not in aliases:
            raise RuntimeError(
                "workspace_stop_oracle must be one of: public_self_eval, submit_tool, hidden_judge_submit"
            )
        return aliases[normalized]

    def _agent_stop_policy(self, stop_oracle: str) -> str:
        if stop_oracle == "hidden_judge_submit":
            return "submit_hidden_judge_until_pass"
        if stop_oracle == "submit_tool":
            return "submit_tool"
        return "run_self_eval_then_summary"

    def _workspace_feedback_scope(self, stop_oracle: str) -> str:
        if stop_oracle == "hidden_judge_submit":
            return self.HIDDEN_JUDGE_FEEDBACK_SCOPE
        return self.WORKSPACE_FEEDBACK_SCOPE

    def _workspace_harness_feedback_mode(self, stop_oracle: str) -> str:
        if stop_oracle == "hidden_judge_submit":
            return self.HIDDEN_JUDGE_HARNESS_FEEDBACK_MODE
        return self.WORKSPACE_HARNESS_FEEDBACK_MODE

    def _empty_submission_state(self) -> dict[str, Any]:
        return {
            "submission_attempted": False,
            "submission_accepted": False,
            "submission_rejection_reasons": [],
            "submission_id": "",
            "submission_attempt_count": 0,
            "last_submission_status": "",
            "check_ready_calls": [],
            "submit_result_calls": [],
            "submission_events": [],
        }

    def _empty_private_submission_state(self) -> dict[str, Any]:
        return {
            "judge_results": [],
        }

    def _merge_json_safe_event_lists(
        self,
        base_items: Sequence[Any],
        incoming_items: Sequence[Any],
    ) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for item in [*(base_items or []), *(incoming_items or [])]:
            safe_item = self._json_safe(item)
            marker = json.dumps(safe_item, sort_keys=True, ensure_ascii=False)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(safe_item)
        return merged

    def _merge_submission_state(
        self,
        base_state: Optional[dict[str, Any]],
        incoming_state: Any,
    ) -> dict[str, Any]:
        merged = dict(base_state or self._empty_submission_state())
        if not isinstance(incoming_state, dict):
            return merged
        merged["submission_attempted"] = bool(
            incoming_state.get("submission_attempted", merged.get("submission_attempted", False))
        )
        merged["submission_accepted"] = bool(
            incoming_state.get("submission_accepted", merged.get("submission_accepted", False))
        )
        merged["submission_id"] = str(
            incoming_state.get("submission_id", merged.get("submission_id", "")) or ""
        )
        rejection_reasons = list(merged.get("submission_rejection_reasons", []))
        rejection_reasons.extend(
            str(item)
            for item in incoming_state.get("submission_rejection_reasons", []) or []
            if str(item)
        )
        merged["submission_rejection_reasons"] = list(dict.fromkeys(rejection_reasons))
        for field_name in ("check_ready_calls", "submit_result_calls", "submission_events"):
            merged[field_name] = self._merge_json_safe_event_lists(
                list(merged.get(field_name, []) or []),
                list(incoming_state.get(field_name, []) or []),
            )
        merged["submission_attempt_count"] = max(
            int(merged.get("submission_attempt_count", 0) or 0),
            int(incoming_state.get("submission_attempt_count", 0) or 0),
            len(list(merged.get("submit_result_calls", []) or [])),
        )
        merged["last_submission_status"] = str(
            incoming_state.get("last_submission_status", merged.get("last_submission_status", "")) or ""
        )
        return merged

    def _merge_private_submission_state(
        self,
        base_state: Optional[dict[str, Any]],
        incoming_state: Any,
    ) -> dict[str, Any]:
        merged = dict(base_state or self._empty_private_submission_state())
        if not isinstance(incoming_state, dict):
            return merged
        merged["judge_results"] = self._merge_json_safe_event_lists(
            list(merged.get("judge_results", []) or []),
            list(incoming_state.get("judge_results", []) or []),
        )
        return merged

    def _resolve_submission_state(
        self,
        *,
        response_metadata: dict[str, Any],
        sdk_diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._merge_submission_state({}, sdk_diagnostics.get("submission_state"))
        return self._merge_submission_state(state, response_metadata.get("submission_state"))

    def _submission_metadata(
        self,
        submission_state: dict[str, Any],
        *,
        stop_oracle: str,
    ) -> dict[str, Any]:
        return {
            "stop_oracle": stop_oracle,
            "submission_attempted": bool(submission_state.get("submission_attempted", False)),
            "submission_accepted": bool(submission_state.get("submission_accepted", False)),
            "submission_rejection_reasons": list(
                submission_state.get("submission_rejection_reasons", [])
            ),
            "submission_id": str(submission_state.get("submission_id", "") or ""),
            "submission_attempt_count": int(submission_state.get("submission_attempt_count", 0) or 0),
            "last_submission_status": str(
                submission_state.get("last_submission_status", "") or ""
            ),
        }

    def _write_submission_artifacts(
        self,
        *,
        workspace: Path,
        round_index: int,
        submission_state: dict[str, Any],
        stop_oracle: str,
    ) -> None:
        artifact_fields = [
            (f"submit_result_calls_round_{round_index}.json", "submit_result_calls"),
            (f"submission_events_round_{round_index}.json", "submission_events"),
        ]
        if stop_oracle == "submit_tool":
            artifact_fields.insert(0, (f"check_ready_calls_round_{round_index}.json", "check_ready_calls"))
        for filename, field_name in artifact_fields:
            (workspace / filename).write_text(
                json.dumps(
                    list(submission_state.get(field_name, [])),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def _serialize_judge_result(self, result: JudgeResult) -> dict[str, Any]:
        return {
            "task_id": result.task_id,
            "all_metrics_passed": bool(result.all_metrics_passed),
            "metrics_actual": dict(result.metrics_actual),
            "failed_metrics": list(result.failed_metrics),
            "failure_tags": list(result.failure_tags),
        }

    def _deserialize_judge_result(self, payload: Any) -> Optional[JudgeResult]:
        if not isinstance(payload, dict):
            return None
        task_id = str(payload.get("task_id", "") or "").strip()
        if not task_id:
            return None
        return JudgeResult(
            task_id=task_id,
            all_metrics_passed=bool(payload.get("all_metrics_passed", False)),
            metrics_actual=dict(payload.get("metrics_actual") or {}),
            failed_metrics=list(payload.get("failed_metrics") or []),
            failure_tags=list(payload.get("failure_tags") or []),
        )

    def _latest_private_judge_result(
        self,
        private_submission_state: Optional[dict[str, Any]],
    ) -> Optional[JudgeResult]:
        state = dict(private_submission_state or {})
        judge_results = list(state.get("judge_results", []) or [])
        if not judge_results:
            return None
        return self._deserialize_judge_result(judge_results[-1])

    def _latest_submission_request_summary(
        self,
        submission_state: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        state = dict(submission_state or {})
        submit_calls = list(state.get("submit_result_calls", []) or [])
        if not submit_calls:
            return {}
        latest_call = dict(submit_calls[-1] or {})
        request_payload = latest_call.get("request")
        return dict(request_payload) if isinstance(request_payload, dict) else {}

    def _hidden_judge_submission_failed(
        self,
        submission_state: Optional[dict[str, Any]],
    ) -> bool:
        state = dict(submission_state or {})
        return (
            int(state.get("submission_attempt_count", 0) or 0) > 0
            and str(state.get("last_submission_status", "") or "") == "fail"
            and not bool(state.get("submission_accepted", False))
        )

    def _hidden_judge_submission_passed(
        self,
        submission_state: Optional[dict[str, Any]],
    ) -> bool:
        state = dict(submission_state or {})
        return bool(state.get("submission_accepted", False)) and (
            str(state.get("last_submission_status", "") or "") == "pass"
        )

    def _build_hidden_judge_manifest(self, task_spec: Mapping[str, Any]) -> dict[str, Any]:
        spec = dict(task_spec or {})
        return {
            "task_id": str(spec.get("task_id", "") or ""),
            "family": str(spec.get("family", "") or ""),
            "primary_output_path": str(spec.get("primary_output_path", "") or ""),
            "task_contract_path": str(spec.get("task_contract_path", "") or ""),
            "judge_adapter_path": str(spec.get("judge_adapter_path", "") or ""),
            "runtime_env": dict(spec.get("runtime_env") or {}),
        }

    def _run_hidden_judge(
        self,
        *,
        task_bundle: TaskBundle,
        task_spec: Mapping[str, Any],
        run_id: str,
        workspace: Path,
        summary: dict[str, Any],
    ) -> JudgeResult:
        manifest = self._build_hidden_judge_manifest(task_spec)
        judge_path = task_bundle.hidden_bundle_dir / str(manifest.get("judge_adapter_path", "") or "")
        if not judge_path.exists():
            raise FileNotFoundError(f"hidden judge adapter not found: {judge_path}")
        payload = invoke_judge_runner(
            _task_python_executable(task_bundle),
            mode="evaluate",
            payload={
                "judge_path": str(judge_path.resolve()),
                "task_root": str(task_bundle.hidden_bundle_dir.resolve()),
                "manifest": manifest,
                "run_record": {
                    "run_id": run_id,
                    "task_id": task_bundle.task_id,
                    "provider": self.provider_name,
                    "env_hash": str((task_spec.get("runtime_env") or {}).get("env_hash", "") or ""),
                    "skills_active": [],
                    "workspace_root": str(Path(workspace).resolve()),
                    "metadata": {
                        "submitted_summary": self._json_safe(summary),
                    },
                },
            },
        )
        result_payload = dict(payload.get("judge_result") or {})
        result = self._deserialize_judge_result(result_payload)
        if result is None:
            raise RuntimeError("hidden judge runner returned an invalid judge_result payload")
        return result

    def _resolve_missing_result_failure_reason(
        self,
        diagnostics: dict[str, Any],
        *,
        submission_state: Optional[dict[str, Any]],
        timeout_occurred: bool,
    ) -> str:
        merged_state = self._merge_submission_state({}, submission_state)
        if bool(merged_state.get("submission_accepted", False)):
            return "accepted_submission_missing_result_message"
        if self._saw_protocol_stop_signal(diagnostics):
            return "protocol_incomplete_after_stop"
        return "request_timeout" if timeout_occurred else "missing_result_message"

    def _build_check_ready_payload(
        self,
        *,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        workspace: Path,
        tool_policy: dict[str, Any],
        readonly_before: dict[str, str],
    ) -> dict[str, Any]:
        output_state = self._inspect_output_contract_state(
            task_spec=task_spec,
            runtime_paths=runtime_paths,
        )
        readonly_after = self._snapshot_roots(workspace, tool_policy["read_roots"])
        readonly_violations = self._detect_snapshot_mutations(readonly_before, readonly_after)
        reasons: list[str] = []
        reasons.extend(f"missing output artifact: {item}" for item in output_state["missing_outputs"])
        reasons.extend(
            f"missing required field: {item}" for item in output_state["missing_output_fields"]
        )
        reasons.extend(str(item) for item in output_state["schema_warnings"])
        reasons.extend(f"read-only path modified: {item}" for item in readonly_violations)
        return {
            "ready": bool(output_state.get("output_schema_valid", False)) and not readonly_violations,
            "reasons": list(dict.fromkeys(reasons)),
            "required_outputs": list(output_state.get("required_outputs", [])),
            "existing_outputs": list(output_state.get("existing_outputs", [])),
            "missing_outputs": list(output_state.get("missing_outputs", [])),
            "missing_fields": list(output_state.get("missing_output_fields", [])),
            "schema_warnings": list(output_state.get("schema_warnings", [])),
            "readonly_violations": list(readonly_violations),
        }

    def _build_submission_tool_result(
        self,
        payload: dict[str, Any],
        *,
        is_error: bool = False,
    ) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                }
            ],
            "is_error": is_error,
        }

    def _evaluate_submission_request(
        self,
        *,
        summary_payload: Any,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        workspace: Path,
        tool_policy: dict[str, Any],
        readonly_before: dict[str, str],
    ) -> tuple[bool, dict[str, Any], dict[str, Any]]:
        summary_valid = True
        summary_error = ""
        try:
            normalized_summary = self._coerce_sdk_summary(summary_payload)
        except RuntimeError as exc:
            summary_valid = False
            summary_error = str(exc)
            normalized_summary = {}
        readiness_payload = self._build_check_ready_payload(
            task_spec=task_spec,
            runtime_paths=runtime_paths,
            workspace=workspace,
            tool_policy=tool_policy,
            readonly_before=readonly_before,
        )
        reasons = list(readiness_payload.get("reasons", []))
        if not summary_valid and summary_error:
            reasons.append(summary_error)
        response_payload = {
            "accepted": bool(readiness_payload.get("ready", False)) and summary_valid,
            "reasons": list(dict.fromkeys(reasons)),
            "submission_id": "",
        }
        return bool(response_payload["accepted"]), normalized_summary, response_payload

    def _build_submission_mcp_servers(
        self,
        *,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        workspace: Path,
        tool_policy: dict[str, Any],
        readonly_before: dict[str, str],
        submission_state: dict[str, Any],
    ) -> dict[str, Any]:
        from claude_agent_sdk import create_sdk_mcp_server, tool

        @tool(
            "check_ready",
            "Check whether the public output contract is satisfied and the run is ready for submission.",
            {"type": "object", "properties": {}},
        )
        async def check_ready(_args):
            payload = self._build_check_ready_payload(
                task_spec=task_spec,
                runtime_paths=runtime_paths,
                workspace=workspace,
                tool_policy=tool_policy,
                readonly_before=readonly_before,
            )
            call_record = {
                "tool": "check_ready",
                "request": {},
                "response": payload,
            }
            submission_state["check_ready_calls"].append(self._json_safe(call_record))
            submission_state["submission_events"].append(self._json_safe(call_record))
            return self._build_submission_tool_result(payload)

        @tool(
            "submit_result",
            "Submit the final structured summary after the public output contract is ready.",
            self._workspace_summary_schema(),
        )
        async def submit_result(args):
            accepted, normalized_summary, response_payload = self._evaluate_submission_request(
                summary_payload=args,
                task_spec=task_spec,
                runtime_paths=runtime_paths,
                workspace=workspace,
                tool_policy=tool_policy,
                readonly_before=readonly_before,
            )
            submission_state["submission_attempted"] = True
            if accepted:
                submission_id = (
                    f"round-1-submit-{len(submission_state['submit_result_calls']) + 1}"
                )
                submission_state["submission_accepted"] = True
                submission_state["submission_id"] = submission_id
                submission_state["submission_rejection_reasons"] = []
                response_payload["submission_id"] = submission_id
            else:
                submission_state["submission_rejection_reasons"] = list(
                    response_payload.get("reasons", [])
                )
            call_record = {
                "tool": "submit_result",
                "request": self._json_safe(normalized_summary if normalized_summary else args),
                "response": response_payload,
            }
            submission_state["submit_result_calls"].append(self._json_safe(call_record))
            submission_state["submission_events"].append(self._json_safe(call_record))
            return self._build_submission_tool_result(response_payload, is_error=not accepted)

        server = create_sdk_mcp_server(
            name=self.SUBMISSION_SERVER_NAME,
            version="1.0.0",
            tools=[check_ready, submit_result],
        )
        return {self.SUBMISSION_SERVER_NAME: server}

    def _build_hidden_judge_submission_mcp_servers(
        self,
        *,
        task_bundle: TaskBundle,
        session_config: ExecutorSessionConfig,
        task_spec: dict[str, Any],
        workspace: Path,
        submission_state: dict[str, Any],
        private_submission_state: dict[str, Any],
    ) -> dict[str, Any]:
        from claude_agent_sdk import create_sdk_mcp_server, tool

        @tool(
            "submit_result",
            "Submit the current candidate output to the hidden judge. The response is only pass or fail.",
            self._workspace_summary_schema(),
        )
        async def submit_result(args):
            try:
                normalized_summary = self._coerce_sdk_summary(args)
            except RuntimeError as exc:
                return self._build_submission_tool_result(
                    {"error": str(exc)},
                    is_error=True,
                )

            judge_result = self._run_hidden_judge(
                task_bundle=task_bundle,
                task_spec=task_spec,
                run_id=session_config.run_id,
                workspace=workspace,
                summary=normalized_summary,
            )
            submission_state["submission_attempted"] = True
            submission_state["submission_attempt_count"] = int(
                submission_state.get("submission_attempt_count", 0) or 0
            ) + 1
            attempt_index = int(submission_state["submission_attempt_count"])
            status = "pass" if judge_result.all_metrics_passed else "fail"
            submission_state["last_submission_status"] = status
            response_payload = {"status": status}
            submission_state["submission_accepted"] = bool(judge_result.all_metrics_passed)
            if judge_result.all_metrics_passed:
                submission_state["submission_rejection_reasons"] = []
                submission_state["submission_id"] = f"hidden-submit-{attempt_index}"
            else:
                submission_state["submission_id"] = ""
            call_record = {
                "tool": "submit_result",
                "attempt_index": attempt_index,
                "request": self._json_safe(normalized_summary),
                "response": response_payload,
            }
            submission_state["submit_result_calls"].append(self._json_safe(call_record))
            submission_state["submission_events"].append(self._json_safe(call_record))
            private_submission_state["judge_results"] = self._merge_json_safe_event_lists(
                list(private_submission_state.get("judge_results", []) or []),
                [self._serialize_judge_result(judge_result)],
            )
            return self._build_submission_tool_result(response_payload)

        server = create_sdk_mcp_server(
            name=self.HIDDEN_JUDGE_SERVER_NAME,
            version="1.0.0",
            tools=[submit_result],
        )
        return {self.HIDDEN_JUDGE_SERVER_NAME: server}

    def _coerce_sdk_summary(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("Claude SDK workspace agent did not return a structured summary")

        def _require_string_list(field_name: str) -> list[str]:
            value = payload.get(field_name, [])
            if value is None:
                return []
            if not isinstance(value, list):
                raise RuntimeError(
                    f"Claude SDK workspace agent summary field '{field_name}' must be an array"
                )
            return [str(item) for item in value]

        return {
            "solver_summary": str(payload.get("solver_summary", "") or ""),
            "declared_outputs": _require_string_list("declared_outputs"),
            "assumptions": _require_string_list("assumptions"),
            "files_written": _require_string_list("files_written"),
            "commands_run": _require_string_list("commands_run"),
        }

    def _extract_sdk_summary(
        self,
        last_message: Any,
        sdk_messages: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        structured_output = getattr(last_message, "structured_output", None)
        if isinstance(structured_output, dict):
            return structured_output
        result_text = self._extract_sdk_result_text(last_message)
        error_type = self._extract_sdk_error_type(last_message)
        text = result_text.strip()
        if getattr(last_message, "is_error", False):
            if error_type == "max_turns":
                raise ClaudeSDKExecutionError(
                    "Claude SDK workspace agent hit the max_turns limit before returning a structured summary",
                    error_type="max_turns",
                    sdk_messages=sdk_messages,
                    result_text=result_text,
                )
            detail = text or str(getattr(last_message, "subtype", "") or "Claude SDK result error")
            raise ClaudeSDKExecutionError(
                "Claude SDK workspace agent returned an error ResultMessage: " + detail,
                error_type="result_error",
                sdk_messages=sdk_messages,
                result_text=result_text,
            )
        if not text:
            raise ClaudeSDKExecutionError(
                "Claude SDK workspace agent did not return a structured summary",
                error_type="invalid_structured_summary",
                sdk_messages=sdk_messages,
                result_text=result_text,
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClaudeSDKExecutionError(
                "Claude SDK workspace agent did not return valid structured JSON",
                error_type="invalid_structured_summary",
                sdk_messages=sdk_messages,
                result_text=result_text,
            ) from exc
        if not isinstance(parsed, dict):
            raise ClaudeSDKExecutionError(
                "Claude SDK workspace agent did not return a JSON object summary",
                error_type="invalid_structured_summary",
                sdk_messages=sdk_messages,
                result_text=result_text,
            )
        return parsed

    def _append_protocol_failure_stderr(self, stderr: str, message: str) -> str:
        failure_line = f"Claude SDK protocol failure: {message}\n"
        if not stderr:
            return failure_line
        suffix = "" if stderr.endswith("\n") else "\n"
        return stderr + suffix + failure_line

    def _extract_sdk_result_text(self, last_message: Any) -> str:
        result_text = getattr(last_message, "result", "")
        if isinstance(result_text, str):
            return result_text
        if result_text is None:
            return ""
        return str(result_text)

    def _extract_sdk_error_type(self, last_message: Any) -> str:
        subtype = str(getattr(last_message, "subtype", "") or "").strip().lower()
        if subtype == "error_max_turns":
            return "max_turns"
        if subtype:
            return subtype
        if getattr(last_message, "is_error", False):
            return "result_error"
        return ""

    def _extract_sdk_result_metadata(self, result_message: Any) -> dict[str, Any]:
        if result_message is None:
            return {
                "subtype": "",
                "stop_reason": "",
                "is_error": False,
                "num_turns": 0,
                "session_id": "",
            }
        return {
            "subtype": str(getattr(result_message, "subtype", "") or ""),
            "stop_reason": str(getattr(result_message, "stop_reason", "") or ""),
            "is_error": bool(getattr(result_message, "is_error", False)),
            "num_turns": int(getattr(result_message, "num_turns", 0) or 0),
            "session_id": str(getattr(result_message, "session_id", "") or ""),
        }

    def _build_protocol_metadata(
        self,
        *,
        sdk_diagnostics: dict[str, Any],
        protocol_status: str,
        protocol_failure_reason: str,
        sdk_completion_source: str = "result_message",
    ) -> dict[str, Any]:
        sdk_result = sdk_diagnostics.get("sdk_result", {})
        if not isinstance(sdk_result, dict):
            sdk_result = {}
        return {
            "protocol_status": protocol_status,
            "protocol_failure_reason": str(protocol_failure_reason or ""),
            "sdk_completion_source": str(sdk_completion_source or "result_message"),
            "sdk_result_subtype": str(sdk_result.get("subtype", "") or ""),
            "sdk_result_stop_reason": str(sdk_result.get("stop_reason", "") or ""),
            "sdk_result_is_error": bool(sdk_result.get("is_error", False)),
            "sdk_result_num_turns": int(sdk_result.get("num_turns", 0) or 0),
            "sdk_result_session_id": str(sdk_result.get("session_id", "") or ""),
        }

    def _build_sdk_failure_metadata(
        self,
        *,
        sdk_messages: list[Any],
        bridge_mode: str,
        model_provider_kind: str,
        raw_response_text: str,
        sdk_diagnostics: Optional[dict[str, Any]] = None,
        vendor_session_ref: Optional[dict[str, Any]] = None,
        protocol_failure_reason: str,
        submission_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        normalized_diagnostics = dict(sdk_diagnostics or {})
        normalized_diagnostics.setdefault("sdk_result", self._extract_sdk_result_metadata(None))
        normalized_submission_state = self._merge_submission_state(
            normalized_diagnostics.get("submission_state"),
            submission_state,
        )
        normalized_diagnostics["submission_state"] = dict(normalized_submission_state)
        return {
            "bridge_mode": bridge_mode,
            "model_provider_kind": model_provider_kind,
            "sdk_error_type": protocol_failure_reason,
            "raw_response_preview": self._preview_text(raw_response_text),
            "raw_response_text": raw_response_text,
            "response_format": "claude_sdk_workspace",
            "response_candidate_count": 1,
            "response_selected_source": "claude_sdk_protocol_failure",
            "prompt_contract_version": self.WORKSPACE_PROMPT_VERSION,
            "sdk_messages": sdk_messages,
            "sdk_diagnostics": normalized_diagnostics,
            "submission_state": dict(normalized_submission_state),
            "vendor_session_ref": dict(vendor_session_ref or {}),
            **self._build_protocol_metadata(
                sdk_diagnostics=normalized_diagnostics,
                protocol_status="failed",
                protocol_failure_reason=protocol_failure_reason,
            ),
        }

    def _workspace_python_path_entries(self, python_executable: str | Path) -> list[str]:
        value = str(python_executable or "").strip()
        if not value:
            return []
        return python_executable_path_entries(Path(value))

    def _prepend_env_path_entries(self, existing_path: str, entries: Sequence[str]) -> str:
        merged_entries: list[str] = []
        seen: set[str] = set()
        incoming = list(entries)
        if existing_path:
            incoming.extend(segment for segment in existing_path.split(os.pathsep) if segment)
        for item in incoming:
            normalized = os.path.normcase(os.path.normpath(str(item)))
            if normalized in seen:
                continue
            seen.add(normalized)
            merged_entries.append(str(item))
        return os.pathsep.join(merged_entries)

    def _build_workspace_sdk_env(
        self,
        runtime_env: Mapping[str, str],
        provider_env: Optional[Mapping[str, str]] = None,
    ) -> dict[str, str]:
        sdk_env = dict(provider_env or {})
        task_python_executable = str(
            runtime_env.get("MYEVOSKILL_TASK_PYTHON")
            or runtime_env.get("MYEVOSKILL_PYTHON_EXE")
            or sys.executable
        ).strip() or sys.executable
        sdk_env["PATH"] = self._prepend_env_path_entries(
            str(runtime_env.get("PATH", "") or ""),
            self._workspace_python_path_entries(task_python_executable),
        )
        for key in (
            "MYEVOSKILL_RUNTIME_ROOT",
            "MYEVOSKILL_PUBLIC_BUNDLE",
            "MYEVOSKILL_WORK_DIR",
            "MYEVOSKILL_OUTPUT_DIR",
            "MYEVOSKILL_CHECKPOINT_DIR",
            "MYEVOSKILL_WORKSPACE",
            "MYEVOSKILL_TASK_ID",
            "MYEVOSKILL_TASK_PYTHON",
            "MYEVOSKILL_TASK_ENV_HASH",
            "MYEVOSKILL_TASK_ENV_BACKEND",
            "PYTHONIOENCODING",
            "PYTHONUTF8",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        ):
            value = runtime_env.get(key)
            if value:
                sdk_env[key] = str(value)
        sdk_env["MYEVOSKILL_PYTHON_EXE"] = str(Path(task_python_executable).resolve())
        conda_prefix = str(runtime_env.get("CONDA_PREFIX", "") or "")
        if not conda_prefix:
            task_prefix = str(runtime_env.get("VIRTUAL_ENV", "") or "").strip()
            if task_prefix:
                prefix = Path(task_prefix).resolve()
            else:
                prefix = _venv_root_from_python_executable(Path(task_python_executable))
            if prefix.exists() and (prefix / "conda-meta").exists():
                conda_prefix = str(prefix)
        if conda_prefix:
            sdk_env["CONDA_PREFIX"] = conda_prefix
        return sdk_env

    def _build_claude_sdk_options_kwargs(
        self,
        *,
        session_config: ExecutorSessionConfig,
        workspace: Path,
        system_prompt: dict[str, Any],
        stop_oracle: str,
        tool_policy: Optional[dict[str, Any]] = None,
        mcp_servers: Optional[dict[str, Any]] = None,
        sdk_env: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        allowed_tools = self._all_allowed_tools(stop_oracle, tool_policy=tool_policy)
        disallowed_tools = self._all_disallowed_tools(tool_policy=tool_policy)
        raw_max_turns = session_config.provider_extras.get("claude_max_turns", 50)
        if raw_max_turns is None:
            max_turns = None
        else:
            try:
                parsed_max_turns = int(raw_max_turns)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("claude_max_turns must be an integer or null") from exc
            max_turns = parsed_max_turns if parsed_max_turns > 0 else None
        return {
            "tools": list(allowed_tools),
            "allowed_tools": list(allowed_tools),
            "disallowed_tools": list(disallowed_tools),
            "system_prompt": system_prompt,
            "output_format": {"type": "json_schema", "schema": self._workspace_summary_schema()},
            "setting_sources": ["user", "project"],
            "permission_mode": "acceptEdits",
            "add_dirs": [str(workspace)],
            "cwd": str(workspace),
            "effort": str(session_config.provider_extras.get("claude_effort", "low") or "low"),
            "max_turns": max_turns,
            "mcp_servers": dict(mcp_servers or {}),
            "env": dict(sdk_env or {}),
        }

    def _extract_sdk_session_id(self, sdk_messages: Sequence[Any]) -> str:
        for message in sdk_messages:
            if isinstance(message, dict):
                session_id = str(message.get("session_id", "") or "").strip()
                if session_id:
                    return session_id
                data = message.get("data")
                if isinstance(data, dict):
                    nested = str(data.get("session_id", "") or "").strip()
                    if nested:
                        return nested
        return ""

    def _build_vendor_session_ref(
        self,
        client: Any,
        workspace: Path,
        *,
        sdk_messages: Optional[Sequence[Any]] = None,
    ) -> dict[str, Any]:
        session_id = str(
            getattr(client, "session_id", None)
            or getattr(client, "conversation_id", None)
            or getattr(client, "request_id", None)
            or ""
        ).strip()
        if not session_id and sdk_messages:
            session_id = self._extract_sdk_session_id(sdk_messages)
        native_ref = self._locate_claude_native_trace(workspace, session_id=session_id)
        return {
            "sdk_backend": "claude_sdk",
            "session_id": session_id,
            **native_ref,
        }

    def _new_claude_session_id(self) -> str:
        """Allocate a fresh Claude session UUID so each run is conversation-isolated."""

        return str(uuid.uuid4())

    def _delete_claude_session_history(self, session_id: str, workspace: Path) -> dict[str, Any]:
        """Best-effort hard-delete of the Claude local session transcript for this workspace."""

        resolved_workspace = Path(workspace).resolve()
        cleanup = {
            "requested": bool(str(session_id or "").strip()),
            "deleted": False,
            "error": "",
            "directory": str(resolved_workspace),
        }
        normalized_session_id = str(session_id or "").strip()
        if not normalized_session_id:
            cleanup["error"] = "missing_session_id"
            return cleanup
        try:
            from claude_agent_sdk import delete_session
        except Exception as exc:  # pragma: no cover - optional SDK import
            cleanup["error"] = f"sdk_import_failed: {exc}"
            return cleanup
        try:
            delete_session(normalized_session_id, directory=str(resolved_workspace))
            cleanup["deleted"] = True
            return cleanup
        except FileNotFoundError:
            cleanup["error"] = "session_not_found"
            return cleanup
        except Exception as exc:  # pragma: no cover - defensive
            cleanup["error"] = f"{type(exc).__name__}: {exc}"
            return cleanup

    def _attach_session_cleanup(
        self,
        vendor_session_ref: Optional[Mapping[str, Any]],
        *,
        session_cleanup: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Attach cleanup status and strip native trace pointers once the session is deleted."""

        finalized = dict(vendor_session_ref or {})
        cleanup = dict(session_cleanup or {})
        finalized["session_cleanup"] = cleanup
        if bool(cleanup.get("deleted")):
            finalized["matched_native_path"] = ""
            finalized["matched_native_exists"] = False
        return finalized

    def _locate_claude_native_trace(self, workspace: Path, *, session_id: str = "") -> dict[str, Any]:
        projects_root = Path.home() / ".claude" / "projects"
        project_key = self._claude_project_key(workspace)
        candidate_dir = projects_root / project_key
        candidate_path: Optional[Path] = None
        if candidate_dir.exists():
            if session_id:
                exact = candidate_dir / f"{session_id}.jsonl"
                if exact.exists():
                    candidate_path = exact
            if candidate_path is None:
                jsonl_files = sorted(candidate_dir.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
                if jsonl_files:
                    candidate_path = jsonl_files[0]
        return {
            "projects_root": str(projects_root),
            "project_key": project_key,
            "matched_native_path": str(candidate_path) if candidate_path is not None else "",
            "matched_native_exists": bool(candidate_path is not None and candidate_path.exists()),
        }

    def _claude_project_key(self, workspace: Path) -> str:
        resolved = str(workspace.resolve())
        return "".join(char if char.isalnum() else "-" for char in resolved)

    def _build_sdk_metadata(
        self,
        *,
        sdk_messages: list[Any],
        parsed_summary: dict[str, Any],
        bridge_mode: str,
        model_provider_kind: str,
        raw_response_text: str,
        sdk_diagnostics: Optional[dict[str, Any]] = None,
        submission_state: Optional[dict[str, Any]] = None,
        completion_source: str = "result_message",
    ) -> dict[str, Any]:
        commands_from_trace = self._extract_sdk_commands(sdk_messages)
        if commands_from_trace:
            parsed_summary["commands_run"] = commands_from_trace
        normalized_diagnostics = dict(sdk_diagnostics or {})
        normalized_diagnostics.setdefault("sdk_result", self._extract_sdk_result_metadata(None))
        normalized_submission_state = self._merge_submission_state(
            normalized_diagnostics.get("submission_state"),
            submission_state,
        )
        normalized_diagnostics["submission_state"] = dict(normalized_submission_state)
        return {
            "bridge_mode": bridge_mode,
            "model_provider_kind": model_provider_kind,
            "sdk_error_type": "",
            "raw_response_preview": self._preview_text(raw_response_text),
            "raw_response_text": raw_response_text,
            "response_format": "claude_sdk_workspace",
            "response_candidate_count": 1,
            "response_selected_source": "claude_sdk_result_message",
            "parsed_response": parsed_summary,
            "prompt_contract_version": self.WORKSPACE_PROMPT_VERSION,
            "sdk_messages": sdk_messages,
            "sdk_diagnostics": normalized_diagnostics,
            "submission_state": dict(normalized_submission_state),
            **self._build_protocol_metadata(
                sdk_diagnostics=normalized_diagnostics,
                protocol_status="completed",
                protocol_failure_reason="",
                sdk_completion_source=completion_source,
            ),
        }

    def _normalize_shell_command(self, command: Any) -> str:
        text = str(command or "")
        return re.sub(r"\s+", " ", text.replace("\\", "/")).strip().lower()

    def _extract_tool_use_entries(self, payload: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                entry_id = str(node.get("id", "") or "").strip()
                tool_name = str(node.get("name", "") or node.get("tool_name", "") or "").strip()
                tool_input = node.get("input")
                if entry_id and tool_name and isinstance(tool_input, dict):
                    entries.append(
                        {
                            "id": entry_id,
                            "name": tool_name,
                            "input": dict(tool_input),
                        }
                    )
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)
        return entries

    def _extract_tool_result_entries(self, payload: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                if "tool_use_result" in node:
                    tool_result = node.get("tool_use_result")
                    content_text = ""
                    tool_use_id = ""
                    is_error = False
                    for item in node.get("content", []) or []:
                        if not isinstance(item, dict):
                            continue
                        candidate_tool_use_id = str(item.get("tool_use_id", "") or "").strip()
                        if candidate_tool_use_id and not tool_use_id:
                            tool_use_id = candidate_tool_use_id
                        if item.get("is_error") is True:
                            is_error = True
                        if not content_text and item.get("content") is not None:
                            content_text = str(item.get("content") or "")
                    if not tool_use_id:
                        candidate_tool_use_id = str(node.get("parent_tool_use_id", "") or "").strip()
                        if candidate_tool_use_id:
                            tool_use_id = candidate_tool_use_id
                    if isinstance(tool_result, str) and tool_result.lower().startswith("error:"):
                        is_error = True
                    entries.append(
                        {
                            "tool_use_id": tool_use_id,
                            "is_error": bool(is_error),
                            "tool_use_result": tool_result,
                            "content": content_text,
                        }
                    )
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)
        return entries

    def _parse_tool_result_returncode(self, result_entry: dict[str, Any]) -> int:
        tool_result = result_entry.get("tool_use_result")
        if isinstance(tool_result, dict) and tool_result.get("returncode") is not None:
            try:
                return int(tool_result.get("returncode"))
            except (TypeError, ValueError):
                pass
        combined_text = " ".join(
            str(item or "")
            for item in (
                result_entry.get("content"),
                tool_result if isinstance(tool_result, str) else "",
            )
            if item is not None
        )
        match = re.search(r"(?:Error:\s*)?Exit code\s+(-?\d+)", combined_text)
        if match:
            return int(match.group(1))
        return 1 if bool(result_entry.get("is_error", False)) else 0

    def _find_latest_bash_command_result(
        self,
        sdk_messages: Sequence[Any],
        expected_command: str,
    ) -> Optional[dict[str, Any]]:
        expected_command_norm = self._normalize_shell_command(expected_command)
        tool_uses: dict[str, dict[str, Any]] = {}
        latest_match: Optional[dict[str, Any]] = None

        for message in sdk_messages:
            for entry in self._extract_tool_use_entries(message):
                entry_id = str(entry.get("id", "") or "").strip()
                if not entry_id:
                    continue
                tool_uses[entry_id] = {
                    "name": str(entry.get("name", "") or ""),
                    "command": str((entry.get("input") or {}).get("command", "") or ""),
                    "description": str((entry.get("input") or {}).get("description", "") or ""),
                }
            for result_entry in self._extract_tool_result_entries(message):
                tool_use_id = str(result_entry.get("tool_use_id", "") or "").strip()
                if not tool_use_id:
                    continue
                tool_use = tool_uses.get(tool_use_id)
                if not tool_use:
                    continue
                if str(tool_use.get("name", "")).strip().lower() != "bash":
                    continue
                command = str(tool_use.get("command", "") or "")
                if expected_command_norm not in self._normalize_shell_command(command):
                    continue
                tool_result = result_entry.get("tool_use_result")
                stdout = ""
                stderr = ""
                interrupted = False
                if isinstance(tool_result, dict):
                    stdout = str(tool_result.get("stdout", "") or "")
                    stderr = str(tool_result.get("stderr", "") or "")
                    interrupted = bool(tool_result.get("interrupted", False))
                elif isinstance(tool_result, str):
                    stderr = str(tool_result or "")
                returncode = self._parse_tool_result_returncode(result_entry)
                latest_match = {
                    "tool_use_id": tool_use_id,
                    "command": command,
                    "description": str(tool_use.get("description", "") or ""),
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "interrupted": interrupted,
                    "is_error": bool(result_entry.get("is_error", False)),
                    "succeeded": returncode == 0 and not bool(result_entry.get("is_error", False)),
                }
        return latest_match

    def _default_workspace_summary(
        self,
        task_spec: dict[str, Any],
        *,
        solver_summary: str,
    ) -> dict[str, Any]:
        return {
            "solver_summary": str(solver_summary or ""),
            "declared_outputs": [
                str(item.get("path", "") or "").strip()
                for item in self._required_output_specs(task_spec)
                if str(item.get("path", "") or "").strip()
            ],
            "assumptions": [],
            "files_written": [],
            "commands_run": [],
        }

    def _write_workspace_trajectory_artifacts(
        self,
        *,
        workspace: Path,
        task_id: str,
        model_provider: str,
        model_name: str,
        trajectory_rounds: Sequence[dict[str, Any]],
        final_status: str,
        final_error_info: dict[str, Any],
    ) -> None:
        vendor_session_ref: dict[str, Any] = {}
        native_lines: list[str] = []
        normalized_rounds: list[dict[str, Any]] = []

        for round_trace in trajectory_rounds:
            round_index = int(round_trace.get("round_index", 0) or 0)
            sdk_messages = list(round_trace.get("sdk_messages", []))
            if not vendor_session_ref and round_trace.get("vendor_session_ref"):
                vendor_session_ref = dict(round_trace["vendor_session_ref"])
            for message in sdk_messages:
                native_lines.append(
                    json.dumps(
                        {"round_index": round_index, "message": message},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            normalized_rounds.append(
                {
                    "round_index": round_index,
                    "prompt_path": round_trace.get("prompt_path", ""),
                    "commands_run": list(round_trace.get("commands_run", [])),
                    "files_written": list(round_trace.get("files_written", [])),
                    "post_run_audit": round_trace.get(
                        "post_run_audit",
                        round_trace.get("public_self_check", {}),
                    ),
                    "returncode": round_trace.get("returncode"),
                    "protocol_status": round_trace.get("protocol_status", ""),
                    "protocol_failure_reason": round_trace.get("protocol_failure_reason", ""),
                    "sdk_result": round_trace.get("sdk_result", {}),
                    "submission_state": round_trace.get("submission_state", {}),
                    "error_type": round_trace.get("error_type", ""),
                    "error_message": round_trace.get("error_message", ""),
                    "claude_diagnostics": round_trace.get("claude_diagnostics", {}),
                }
            )

        native_path = workspace / "trajectory_native.jsonl"
        matched_native_value = str(vendor_session_ref.get("matched_native_path", "") or "").strip()
        matched_native_path = Path(matched_native_value) if matched_native_value else None
        copied_vendor_native = False
        if matched_native_path is not None and matched_native_path.exists():
            shutil.copy2(matched_native_path, native_path)
            copied_vendor_native = True
        else:
            native_path.write_text(
                ("\n".join(native_lines) + ("\n" if native_lines else "")),
                encoding="utf-8",
            )

        normalized_payload = {
            "sdk_backend": "claude_sdk",
            "task_id": task_id,
            "model_provider": model_provider,
            "model_name": model_name,
            "final_status": final_status,
            "final_error_info": final_error_info,
            "rounds": normalized_rounds,
        }
        (workspace / "trajectory_normalized.json").write_text(
            json.dumps(normalized_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (workspace / "trajectory_summary.json").write_text(
            json.dumps(
                {
                    "sdk_backend": "claude_sdk",
                    "final_status": final_status,
                    "round_count": len(normalized_rounds),
                    "message_count": sum(len(list(item.get("sdk_messages", []))) for item in trajectory_rounds),
                    "final_error_type": final_error_info.get("error_type", ""),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (workspace / "trajectory_redaction_report.json").write_text(
            json.dumps(
                {
                    "policy": "Secrets are not persisted; only safe config names and SDK traces are stored.",
                    "redacted_fields": [],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        vendor_payload = {
            **vendor_session_ref,
            "native_trace_copied": copied_vendor_native,
            "workspace_native_path": str(native_path),
        }
        (workspace / "vendor_session_ref.json").write_text(
            json.dumps(vendor_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _serialize_sdk_messages(self, messages: list[Any]) -> list[Any]:
        return [self._json_safe(message) for message in messages]

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return self._json_safe(value.model_dump())
            except TypeError:
                pass
        if hasattr(value, "__dict__"):
            return self._json_safe(vars(value))
        return str(value)

    def _resolve_mock_sdk_response(self, mock_value: Any, round_index: int) -> dict[str, Any]:
        if isinstance(mock_value, (list, tuple)):
            if not mock_value:
                return {}
            value = mock_value[round_index - 1] if round_index - 1 < len(mock_value) else mock_value[-1]
        else:
            value = mock_value
        if isinstance(value, str):
            return json.loads(value)
        return dict(value or {})

    def _apply_mock_workspace_files(self, workspace_root: Path, files: dict[str, Any]) -> None:
        for relative_path, content in sorted(files.items()):
            target = workspace_root / str(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content or ""), encoding="utf-8")

    def _seed_workspace_scaffold(self, runtime_paths: dict[str, Path]) -> None:
        scaffold_files = {
            "work/main.py": (
                '"""Entrypoint scaffold for the workspace solver."""\n'
                'from pathlib import Path\n\n'
                'def main() -> None:\n'
                '    Path("output").mkdir(parents=True, exist_ok=True)\n'
                '    raise NotImplementedError("Replace this scaffold with a working solver")\n\n'
                'if __name__ == "__main__":\n'
                '    main()\n'
            ),
        }
        runtime_root = runtime_paths["runtime_root"]
        for relative_path, content in scaffold_files.items():
            target = runtime_root / relative_path
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _coerce_tool_policy(self, session_config: ExecutorSessionConfig) -> dict[str, Any]:
        incoming = dict(session_config.tool_policy)
        network_access = bool(incoming.get("network_access", False))
        bash_denied_tokens = [
            str(item)
            for item in incoming.get("bash_denied_tokens", list(self.DEFAULT_BASH_DENIED_TOKENS))
        ]
        if network_access:
            bash_denied_tokens = [
                token
                for token in bash_denied_tokens
                if str(token).strip().lower() not in {"curl", "wget"}
            ]
        # Keep the workspace policy provider-agnostic so the harness remains the
        # source of truth across Claude SDK, OpenHands, and future integrations.
        return {
            "read_roots": [str(item) for item in incoming.get("read_roots", ["data", "public_bundle", "evaluation", "README_public.md", "requirements.txt"])],
            "write_roots": [str(item) for item in incoming.get("write_roots", ["work", "output", "checkpoints"])],
            "bash_allowed_prefixes": [str(item) for item in incoming.get("bash_allowed_prefixes", list(self.DEFAULT_BASH_ALLOWED_PREFIXES))],
            "bash_denied_tokens": bash_denied_tokens,
            "network_access": network_access,
        }

    def _summarize_tool_policy(self, tool_policy: dict[str, Any]) -> dict[str, Any]:
        return {
            "read_roots": list(tool_policy["read_roots"]),
            "write_roots": list(tool_policy["write_roots"]),
            "bash_allowed_prefixes": list(tool_policy["bash_allowed_prefixes"]),
            "bash_denied_tokens": list(tool_policy["bash_denied_tokens"]),
            "network_access": bool(tool_policy["network_access"]),
        }

    def _snapshot_roots(self, workspace_root: Path, roots: Sequence[str]) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for root_name in roots:
            target = workspace_root / root_name
            if target.is_file():
                snapshot[root_name] = self._hash_path(target)
                continue
            if not target.exists():
                continue
            for child in sorted(path for path in target.rglob("*") if path.is_file()):
                rel = child.relative_to(workspace_root).as_posix()
                snapshot[rel] = self._hash_path(child)
        return snapshot

    def _hash_path(self, path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    def _detect_snapshot_mutations(
        self,
        before: dict[str, str],
        after: dict[str, str],
    ) -> list[str]:
        return [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]

    def _collect_changed_paths(
        self,
        before: dict[str, str],
        after: dict[str, str],
    ) -> list[str]:
        return self._detect_snapshot_mutations(before, after)

    def _extract_sdk_commands(self, sdk_messages: list[Any]) -> list[str]:
        commands: list[str] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                name = str(node.get("name", "") or node.get("tool_name", "")).lower()
                if name == "bash":
                    command = node.get("command")
                    if command:
                        commands.append(str(command))
                    payload = node.get("input")
                    if isinstance(payload, dict) and payload.get("command"):
                        commands.append(str(payload["command"]))
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(sdk_messages)
        return list(dict.fromkeys(commands))

    def _extract_denied_bash_attempts(
        self,
        hook_events: Sequence[dict[str, Any]],
        *,
        workspace_root: Path,
        tool_policy: dict[str, Any],
    ) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        seen: set[tuple[str, str, tuple[tuple[str, str, str, str, str], ...]]] = set()
        for raw_event in hook_events:
            if not isinstance(raw_event, Mapping):
                continue
            if str(raw_event.get("hook_event_name", "") or "") != "PreToolUse":
                continue
            if str(raw_event.get("permission_decision", "") or "").lower() != "deny":
                continue
            tool_name = str(raw_event.get("tool_name", "") or "").strip().lower()
            input_payload = raw_event.get("input")
            if not tool_name and isinstance(input_payload, Mapping):
                tool_name = str(input_payload.get("tool_name", "") or "").strip().lower()
            if tool_name and tool_name != "bash":
                continue
            command = str(raw_event.get("command", "") or "").strip()
            if not command and isinstance(input_payload, Mapping):
                tool_input = input_payload.get("tool_input")
                if isinstance(tool_input, Mapping):
                    command = str(tool_input.get("command", "") or "").strip()
            violations = raw_event.get("violations")
            if not isinstance(violations, Sequence) or isinstance(violations, (str, bytes)):
                violations = self._validate_bash_commands([command], workspace_root, tool_policy)
            normalized_violations = self._dedupe_workspace_violations(
                [dict(item) for item in violations if isinstance(item, Mapping)]
            )
            violation_key = tuple(
                self._workspace_violation_identity(item) for item in normalized_violations
            )
            dedupe_key = (
                str(raw_event.get("tool_use_id", "") or ""),
                command,
                violation_key,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            attempts.append(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_use_id": str(raw_event.get("tool_use_id", "") or ""),
                    "tool_name": "Bash",
                    "command": command,
                    "permission_decision": "deny",
                    "permission_decision_reason": str(
                        raw_event.get("permission_decision_reason", "") or ""
                    ),
                    "violations": normalized_violations,
                }
            )
        return attempts

    def _filter_denied_bash_commands(
        self,
        commands: Sequence[str],
        denied_attempts: Sequence[dict[str, Any]],
    ) -> list[str]:
        denied_commands = {
            self._normalize_shell_command(item.get("command", ""))
            for item in denied_attempts
            if str(item.get("command", "") or "").strip()
        }
        filtered: list[str] = []
        for command in commands:
            normalized = self._normalize_shell_command(command)
            if normalized and normalized in denied_commands:
                continue
            filtered.append(str(command))
        return list(dict.fromkeys(filtered))

    def _normalize_workspace_relative_path(
        self,
        path_value: Any,
        *,
        workspace_root: Path,
        current_dir: Optional[Path] = None,
    ) -> str:
        text = self._strip_matching_quotes(str(path_value or "").strip())
        if not text:
            return ""
        path_ref = self._resolve_workspace_path_reference(
            text,
            current_dir=(current_dir or workspace_root),
            workspace_root=workspace_root,
        )
        if not path_ref.get("inside_workspace", False):
            return ""
        return Path(str(path_ref.get("relative_path", "") or "")).as_posix()

    def _workspace_plan_actions(
        self,
        sdk_messages: Sequence[Any],
        *,
        workspace_root: Path,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        workspace_root = Path(workspace_root).resolve()
        entrypoint_command = self._normalize_shell_command(self._workspace_entrypoint_command())
        for message in sdk_messages:
            for entry in self._extract_tool_use_entries(message):
                tool_name = str(entry.get("name", "") or "").strip()
                tool_input = dict(entry.get("input") or {})
                if tool_name == "Write":
                    file_path = self._normalize_workspace_relative_path(
                        tool_input.get("file_path", ""),
                        workspace_root=workspace_root,
                    )
                    if file_path:
                        actions.append({"tool": "Write", "kind": "write", "target": file_path})
                elif tool_name == "Bash":
                    command = str(tool_input.get("command", "") or "").strip()
                    if not command:
                        continue
                    if self._normalize_shell_command(command) == entrypoint_command:
                        actions.append(
                            {
                                "tool": "Bash",
                                "kind": "run",
                                "target": self._workspace_entrypoint_command(),
                            }
                        )
                        continue
                    for segment in self._split_shell_segments(command):
                        for effect in self._extract_segment_path_effects(segment):
                            if str(effect.get("mode", "") or "") != "write":
                                continue
                            target = self._normalize_workspace_relative_path(
                                effect.get("path", ""),
                                workspace_root=workspace_root,
                            )
                            if not target or target in {"work", "work/src", "output", "checkpoints"}:
                                continue
                            actions.append(
                                {
                                    "tool": "Bash",
                                    "kind": "write",
                                    "target": target,
                                    "command": command,
                                }
                            )
                            break
                elif tool_name == "submit_result":
                    actions.append({"tool": "submit_result", "kind": "submit", "target": "submit_result"})
        return actions

    def _evaluate_workspace_plan_requirement(
        self,
        *,
        sdk_messages: Sequence[Any],
        workspace_root: Path,
        files_written: Sequence[str],
        submission_state: Mapping[str, Any],
    ) -> Optional[dict[str, Any]]:
        normalized_files_written = [
            Path(str(item or "")).as_posix()
            for item in files_written
            if str(item or "").strip()
        ]
        actions = self._workspace_plan_actions(sdk_messages, workspace_root=workspace_root)
        requires_plan = bool(
            actions
            or normalized_files_written
            or bool(submission_state.get("submission_attempted", False))
        )
        if not requires_plan:
            return None
        if self.WORKSPACE_PLAN_PATH not in normalized_files_written:
            return self._build_workspace_plan_feedback()
        first_material_action = next(
            (
                action
                for action in actions
                if action.get("kind") in {"write", "run", "submit"}
            ),
            None,
        )
        if first_material_action is None:
            return None
        if first_material_action.get("kind") == "write" and (
            str(first_material_action.get("target", "") or "") == self.WORKSPACE_PLAN_PATH
        ):
            return None
        return self._build_workspace_plan_feedback()

    def _split_handled_bash_policy_violations(
        self,
        violations: Sequence[dict[str, Any]],
        denied_attempts: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        handled_keys = {
            self._workspace_violation_identity(violation)
            for attempt in denied_attempts
            for violation in attempt.get("violations", [])
            if isinstance(violation, Mapping)
        }
        handled: list[dict[str, Any]] = []
        unhandled: list[dict[str, Any]] = []
        for violation in violations:
            target = handled if self._workspace_violation_identity(violation) in handled_keys else unhandled
            target.append(dict(violation))
        return self._dedupe_workspace_violations(handled), self._dedupe_workspace_violations(unhandled)

    def _validate_bash_commands(
        self,
        commands: Sequence[str],
        workspace_root: Path,
        tool_policy: dict[str, Any],
    ) -> list[dict[str, Any]]:
        workspace_root_resolved = workspace_root.resolve()
        violations: list[dict[str, Any]] = []
        for command in commands:
            normalized = str(command or "").strip()
            if not normalized:
                continue
            command_violations, _ = self._validate_bash_expression(
                normalized,
                original_command=normalized,
                current_dir=workspace_root_resolved,
                workspace_root=workspace_root_resolved,
                tool_policy=tool_policy,
            )
            violations.extend(command_violations)
        return self._dedupe_workspace_violations(violations)

    def _validate_bash_expression(
        self,
        expression: str,
        *,
        original_command: str,
        current_dir: Path,
        workspace_root: Path,
        tool_policy: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], Path]:
        violations: list[dict[str, Any]] = []
        active_dir = current_dir
        for segment in self._split_shell_segments(expression):
            segment_violations, active_dir = self._validate_bash_segment(
                segment,
                original_command=original_command,
                current_dir=active_dir,
                workspace_root=workspace_root,
                tool_policy=tool_policy,
            )
            violations.extend(segment_violations)
        return violations, active_dir

    def _validate_bash_segment(
        self,
        segment: str,
        *,
        original_command: str,
        current_dir: Path,
        workspace_root: Path,
        tool_policy: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], Path]:
        normalized = str(segment or "").strip()
        if not normalized:
            return [], current_dir
        policy_segment = self._strip_heredoc_body(normalized)

        cd_target = self._parse_cd_target(policy_segment)
        if cd_target is not None:
            if not cd_target:
                return [], current_dir
            path_ref = self._resolve_workspace_path_reference(
                cd_target,
                current_dir=current_dir,
                workspace_root=workspace_root,
            )
            if not path_ref.get("inside_workspace", False):
                return [
                    self._make_workspace_violation(
                        original_command,
                        "outside_workspace_path",
                        segment=policy_segment,
                        path=str(path_ref.get("raw", cd_target)),
                        detail="cd target resolves outside the workspace",
                    )
                ], current_dir
            return [], Path(path_ref["resolved"])

        wrapped_expression = self._unwrap_shell_wrapper(policy_segment)
        if wrapped_expression is not None:
            return self._validate_bash_expression(
                wrapped_expression,
                original_command=original_command,
                current_dir=current_dir,
                workspace_root=workspace_root,
                tool_policy=tool_policy,
            )

        denied = self._match_denied_bash_category(policy_segment, tool_policy=tool_policy)
        if denied is not None:
            category, detail = denied
            return [
                self._make_workspace_violation(
                    original_command,
                    category,
                    segment=policy_segment,
                    detail=detail,
                )
            ], current_dir

        return (
            self._validate_segment_path_effects(
                policy_segment,
                original_command=original_command,
                current_dir=current_dir,
                workspace_root=workspace_root,
                write_roots=tool_policy["write_roots"],
            ),
            current_dir,
        )

    def _make_workspace_violation(
        self,
        command: str,
        category: str,
        *,
        segment: str = "",
        path: str = "",
        detail: str = "",
    ) -> dict[str, Any]:
        violation = {
            "command": str(command or "").strip(),
            "category": str(category or "").strip(),
            "segment": str(segment or "").strip(),
            "detail": str(detail or "").strip(),
        }
        if path:
            violation["path"] = str(path).strip()
        return violation

    def _dedupe_workspace_violations(
        self,
        violations: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for item in violations:
            key = (
                str(item.get("command", "") or ""),
                str(item.get("segment", "") or ""),
                str(item.get("category", "") or ""),
                str(item.get("path", "") or ""),
                str(item.get("detail", "") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(dict(item))
        return unique

    def _workspace_violation_identity(
        self,
        violation: Mapping[str, Any],
    ) -> tuple[str, str, str, str, str]:
        return (
            str(violation.get("command", "") or ""),
            str(violation.get("segment", "") or ""),
            str(violation.get("category", "") or ""),
            str(violation.get("path", "") or ""),
            str(violation.get("detail", "") or ""),
        )

    def _format_workspace_violations(
        self,
        violations: Sequence[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        for item in violations:
            command = str(item.get("command", "") or item.get("segment", "") or "<unknown>").strip()
            category = str(item.get("category", "violation") or "violation").strip()
            details: list[str] = []
            if item.get("path"):
                details.append(f"path={item['path']}")
            if item.get("detail"):
                details.append(str(item["detail"]))
            if details:
                parts.append(f"{command} [{category}: {'; '.join(details)}]")
            else:
                parts.append(f"{command} [{category}]")
        return ", ".join(parts)

    def _split_shell_segments(self, expression: str) -> list[str]:
        segments: list[str] = []
        buffer: list[str] = []
        in_single = False
        in_double = False
        index = 0
        while index < len(expression):
            char = expression[index]
            if char == "'" and not in_double:
                in_single = not in_single
                buffer.append(char)
                index += 1
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                buffer.append(char)
                index += 1
                continue
            if not in_single and not in_double:
                operator = None
                if expression.startswith("&&", index):
                    operator = "&&"
                elif expression.startswith("||", index):
                    operator = "||"
                elif char in {";", "|"}:
                    operator = char
                if operator is not None:
                    segment = "".join(buffer).strip()
                    if segment:
                        segments.append(segment)
                    buffer = []
                    index += len(operator)
                    continue
            buffer.append(char)
            index += 1
        trailing = "".join(buffer).strip()
        if trailing:
            segments.append(trailing)
        return segments

    def _parse_cd_target(self, segment: str) -> str | None:
        try:
            parts = shlex.split(segment, posix=False)
        except ValueError:
            return None
        if not parts or parts[0].lower() != "cd":
            return None
        candidates = [part for part in parts[1:] if part.lower() != "/d"]
        if not candidates:
            return ""
        return candidates[-1]

    def _unwrap_shell_wrapper(self, segment: str) -> str | None:
        timeout_match = re.match(
            r"^timeout\s+(?:/t\s+)?\d+\s+(?P<inner>.+)$",
            segment,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if timeout_match:
            return self._strip_matching_quotes(timeout_match.group("inner").strip())

        cmd_match = re.match(
            r"^cmd(?:\.exe)?\s+/c\s+(?P<inner>.+)$",
            segment,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if cmd_match:
            return self._strip_matching_quotes(cmd_match.group("inner").strip())

        powershell_match = re.match(
            r"^(?:powershell|pwsh)(?:\.exe)?\s+-Command\s+(?P<inner>.+)$",
            segment,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if powershell_match:
            return self._strip_matching_quotes(powershell_match.group("inner").strip())
        return None

    def _strip_matching_quotes(self, value: str) -> str:
        stripped = str(value or "").strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
            return stripped[1:-1]
        return stripped

    def _looks_like_shell_redirection_token(self, token: str) -> bool:
        normalized = self._strip_matching_quotes(str(token or "").strip())
        if not normalized:
            return False
        return bool(re.match(r"^\d*(?:>>?|<<?|<<<|>&|<&).*$", normalized))

    def _strip_heredoc_body(self, segment: str) -> str:
        in_single = False
        in_double = False
        index = 0
        while index < len(segment) - 1:
            char = segment[index]
            if char == "'" and not in_double:
                in_single = not in_single
                index += 1
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                index += 1
                continue
            if not in_single and not in_double and segment.startswith("<<", index):
                return segment[:index].strip()
            index += 1
        return segment

    def _match_denied_bash_category(
        self,
        segment: str,
        *,
        tool_policy: dict[str, Any],
    ) -> tuple[str, str] | None:
        normalized = str(segment or "").strip()
        lowered = normalized.lower()

        if (
            not bool(tool_policy.get("network_access", False))
            and re.match(
                r"^(?:curl|wget|invoke-webrequest|iwr|irm|ftp|sftp|ssh|scp)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        ):
            return "denied_category", "network access is disabled by the harness"

        if re.match(r"^git\b", normalized, flags=re.IGNORECASE):
            return "denied_category", "version control operations are prohibited"

        if re.match(
            r"^(?:pip|pip3)\s+(?:install|uninstall|download|wheel)\b",
            normalized,
            flags=re.IGNORECASE,
        ) or re.match(
            r"^(?:python|python3|py)\s+-m\s+pip\s+(?:install|uninstall|download|wheel)\b",
            normalized,
            flags=re.IGNORECASE,
        ) or re.match(
            r"^(?:conda|mamba|micromamba)\s+(?:install|update|upgrade|remove|create|env)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return "denied_category", "package or environment installation is prohibited"

        if re.match(r"^(?:sudo|runas)\b", normalized, flags=re.IGNORECASE) or re.search(
            r"-verb\s+runas\b",
            lowered,
        ):
            return "denied_category", "privilege escalation is prohibited"

        if re.match(
            r"^(?:taskkill|pkill|kill|stop-process)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return "denied_category", "killing processes is prohibited"

        if re.match(
            r"^(?:remove-service|start-service|restart-service|stop-service|sc(?:\.exe)?|netsh|set-executionpolicy)\b",
            normalized,
            flags=re.IGNORECASE,
        ) or re.match(
            r"^net\s+(?:start|stop)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return "denied_category", "system or service control is prohibited"

        for token in tool_policy.get("bash_denied_tokens", []):
            denied = str(token or "").strip().lower()
            if denied and lowered.startswith(denied):
                return "denied_category", f"matched denied token: {token}"

        return None

    def _resolve_workspace_path_reference(
        self,
        raw_path: str,
        *,
        current_dir: Path,
        workspace_root: Path,
    ) -> dict[str, Any]:
        raw_value = self._strip_matching_quotes(str(raw_path or "").strip())
        if not raw_value:
            return {
                "raw": "",
                "resolved": None,
                "relative_path": None,
                "inside_workspace": False,
                "special_sink": False,
            }
        if self._is_special_shell_sink(raw_value):
            return {
                "raw": raw_value,
                "resolved": None,
                "relative_path": None,
                "inside_workspace": True,
                "special_sink": True,
            }
        path_value = Path(raw_value)
        candidate = path_value if path_value.is_absolute() else current_dir / path_value
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        try:
            relative_path = resolved.relative_to(workspace_root)
            inside_workspace = True
        except ValueError:
            relative_path = None
            inside_workspace = False
        return {
            "raw": raw_value,
            "resolved": resolved,
            "relative_path": relative_path,
            "inside_workspace": inside_workspace,
            "special_sink": False,
        }

    def _is_special_shell_sink(self, value: str) -> bool:
        normalized = self._strip_matching_quotes(str(value or "").strip()).lower()
        return normalized in {"nul", "nul:", "/dev/null"}

    def _path_is_within_declared_roots(
        self,
        relative_path: Path,
        roots: Sequence[str],
    ) -> bool:
        relative_parts = relative_path.parts
        for root in roots:
            root_parts = Path(str(root)).parts
            if root_parts and relative_parts[: len(root_parts)] == root_parts:
                return True
        return False

    def _validate_segment_path_effects(
        self,
        segment: str,
        *,
        original_command: str,
        current_dir: Path,
        workspace_root: Path,
        write_roots: Sequence[str],
    ) -> list[dict[str, Any]]:
        effects = self._extract_segment_path_effects(segment)
        violations: list[dict[str, Any]] = []
        for effect in effects:
            path_ref = self._resolve_workspace_path_reference(
                str(effect.get("path", "") or ""),
                current_dir=current_dir,
                workspace_root=workspace_root,
            )
            if path_ref.get("special_sink", False):
                continue
            if not path_ref.get("inside_workspace", False):
                violations.append(
                    self._make_workspace_violation(
                        original_command,
                        "outside_workspace_path",
                        segment=segment,
                        path=str(path_ref.get("raw", effect.get("path", ""))),
                        detail="referenced path resolves outside the workspace",
                    )
                )
                continue
            if effect.get("mode") == "write" and not self._path_is_within_declared_roots(
                Path(path_ref["relative_path"]),
                write_roots,
            ):
                violations.append(
                    self._make_workspace_violation(
                        original_command,
                        "outside_write_roots",
                        segment=segment,
                        path=Path(path_ref["relative_path"]).as_posix(),
                        detail="write target is outside writable roots",
                    )
                )
        return violations

    def _extract_segment_path_effects(self, segment: str) -> list[dict[str, str]]:
        effects: list[dict[str, str]] = list(self._extract_shell_redirection_effects(segment))
        try:
            tokens = shlex.split(segment, posix=False)
        except ValueError:
            return effects
        if not tokens:
            return effects

        command = tokens[0].lower()
        args = tokens[1:]

        if command == "touch":
            for target in self._collect_cli_path_arguments(args):
                effects.append({"mode": "write", "path": target})
        elif command == "mkdir":
            for target in self._collect_cli_path_arguments(args):
                effects.append({"mode": "write", "path": target})
        elif command in {"cp", "copy", "mv", "move"}:
            path_args = self._collect_cli_path_arguments(args)
            if len(path_args) >= 2:
                for source in path_args[:-1]:
                    effects.append({"mode": "read", "path": source})
                effects.append({"mode": "write", "path": path_args[-1]})
        elif command in {"rm", "del"}:
            for target in self._collect_cli_path_arguments(args):
                effects.append({"mode": "write", "path": target})
        elif command == "ren":
            path_args = self._collect_cli_path_arguments(args)
            if len(path_args) >= 2:
                source = path_args[0]
                target = path_args[1]
                effects.append({"mode": "write", "path": source})
                if not re.search(r"[\\/]", target):
                    source_parent = Path(self._strip_matching_quotes(source)).parent
                    target = str(source_parent / target)
                effects.append({"mode": "write", "path": target})
        elif command == "new-item":
            target = self._powershell_argument_value(tokens, "-Path", "-LiteralPath")
            if not target:
                positional = self._collect_cli_path_arguments(args)
                target = positional[-1] if positional else ""
            if target:
                effects.append({"mode": "write", "path": target})
        elif command in {"set-content", "add-content", "out-file"}:
            target = self._powershell_argument_value(
                tokens,
                "-Path",
                "-LiteralPath",
                "-FilePath",
            )
            if target:
                effects.append({"mode": "write", "path": target})
        elif command in {"copy-item", "move-item"}:
            source = self._powershell_argument_value(tokens, "-Path", "-LiteralPath")
            target = self._powershell_argument_value(tokens, "-Destination")
            if source:
                effects.append({"mode": "read", "path": source})
            if target:
                effects.append({"mode": "write", "path": target})
        elif command == "remove-item":
            target = self._powershell_argument_value(tokens, "-Path", "-LiteralPath")
            if target:
                effects.append({"mode": "write", "path": target})

        return effects

    def _collect_cli_path_arguments(self, tokens: Sequence[str]) -> list[str]:
        paths: list[str] = []
        for token in tokens:
            normalized = self._strip_matching_quotes(str(token or "").strip())
            if (
                not normalized
                or self._looks_like_shell_flag(normalized)
                or self._looks_like_shell_redirection_token(normalized)
            ):
                continue
            paths.append(normalized)
        return paths

    def _looks_like_shell_flag(self, token: str) -> bool:
        normalized = str(token or "").strip()
        if not normalized:
            return False
        if normalized.startswith("-"):
            return True
        return bool(re.match(r"^/[A-Za-z?]+$", normalized))

    def _powershell_argument_value(self, tokens: Sequence[str], *names: str) -> str:
        lowered_names = {name.lower() for name in names}
        for index, token in enumerate(tokens[:-1]):
            if token.lower() in lowered_names:
                return self._strip_matching_quotes(str(tokens[index + 1] or "").strip())
        return ""

    def _extract_shell_redirection_effects(self, segment: str) -> list[dict[str, str]]:
        effects: list[dict[str, str]] = []
        in_single = False
        in_double = False
        index = 0
        while index < len(segment):
            char = segment[index]
            if char == "'" and not in_double:
                in_single = not in_single
                index += 1
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                index += 1
                continue
            if in_single or in_double or char not in {">", "<"}:
                index += 1
                continue

            operator = char
            if index + 1 < len(segment) and segment[index + 1] == char:
                operator += char
                index += 1

            fd_index = index - len(operator)
            while fd_index > 0 and segment[fd_index - 1].isdigit():
                fd_index -= 1

            cursor = index + 1
            while cursor < len(segment) and segment[cursor].isspace():
                cursor += 1

            target, cursor = self._read_shell_token(segment, cursor)
            target = self._strip_matching_quotes(target)
            operator_token = self._strip_matching_quotes(segment[fd_index : index + 1].strip())
            if (
                target
                and operator_token not in {"<<", "<<<"}
                and not target.startswith("&")
                and not self._looks_like_shell_redirection_token(target)
                and not self._is_special_shell_sink(target)
            ):
                effects.append(
                    {
                        "mode": "write" if operator.startswith(">") else "read",
                        "path": target,
                    }
                )
            index = cursor
        return effects

    def _read_shell_token(self, segment: str, start_index: int) -> tuple[str, int]:
        if start_index >= len(segment):
            return "", start_index
        quote = segment[start_index] if segment[start_index] in {"'", '"'} else ""
        buffer: list[str] = []
        index = start_index
        if quote:
            index += 1
            while index < len(segment) and segment[index] != quote:
                buffer.append(segment[index])
                index += 1
            if index < len(segment) and segment[index] == quote:
                index += 1
            return "".join(buffer), index

        while index < len(segment):
            char = segment[index]
            if char.isspace() or char in {"&", "|", ";"}:
                break
            buffer.append(char)
            index += 1
        return "".join(buffer), index

    def _completed_process_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _generate_legacy_workspace_response(
        self,
        provider_adapter,
        session_config: ExecutorSessionConfig,
        prompt: str,
        model_timeout_seconds: int,
        round_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if "mock_llm_response" in session_config.provider_extras:
            provider_adapter.resolve_api_key()
            raw_response = self._resolve_mock_response(
                session_config.provider_extras["mock_llm_response"],
                round_index,
            )
            return self._parse_legacy_workspace_model_response(
                raw_content=raw_response,
                bridge_mode="workspace_mock_llm",
                model_provider_kind=provider_adapter.safe_log_config().get("kind", ""),
            )
        messages = [
            {"role": "system", "content": self._build_legacy_workspace_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        url, headers, payload = provider_adapter.build_request(messages)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=model_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        raw_content = self._extract_model_content(body)
        return self._parse_legacy_workspace_model_response(
            raw_content=raw_content,
            bridge_mode="workspace_openai_compatible",
            model_provider_kind=provider_adapter.safe_log_config().get("kind", ""),
        )

    def _build_legacy_workspace_system_prompt(self) -> str:
        return "\n".join(
            [
                "You are a Claude-style workspace coding agent.",
                "Return a single JSON object with exactly these top-level keys:",
                '- "files": required object mapping relative file path to full file content',
                '- "entrypoint": required relative path for execution',
                '- "declared_outputs": optional list',
                '- "assumptions": optional list',
                '- "solver_summary": optional short string',
                "Do not include markdown fences.",
                "Do not include explanatory prose outside the JSON object.",
                "Every path in files must be relative and stay under work/.",
            ]
        )

    def _parse_legacy_workspace_model_response(
        self,
        raw_content: str,
        bridge_mode: str,
        model_provider_kind: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        structured_payload, parse_error, candidate_count, selected_source = (
            self._extract_structured_payload(raw_content)
        )
        if structured_payload is None:
            raise ModelResponseParseError(
                "workspace agent requires a structured JSON response",
                raw_response=raw_content,
                parse_error=parse_error or "workspace_json_parse_failed",
                candidate_count=candidate_count,
                selected_source=selected_source or "workspace_json",
            )
        files = self._coerce_workspace_files(structured_payload.get("files"))
        entrypoint = structured_payload.get("entrypoint", "work/main.py")
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            raise ModelResponseParseError(
                "structured workspace response missing required field 'entrypoint'",
                raw_response=raw_content,
                parse_error="workspace_json_missing_entrypoint",
                candidate_count=candidate_count,
                selected_source=selected_source,
            )
        declared_outputs = self._coerce_list_field(structured_payload.get("declared_outputs"), default=[])
        assumptions = self._coerce_list_field(structured_payload.get("assumptions"), default=[])
        solver_summary = structured_payload.get("solver_summary", "")
        payload = {
            "files": files,
            "entrypoint": str(entrypoint),
            "declared_outputs": declared_outputs,
            "assumptions": assumptions,
            "solver_summary": str(solver_summary or ""),
        }
        metadata = {
            "bridge_mode": bridge_mode,
            "model_provider_kind": model_provider_kind,
            "sdk_error_type": "",
            "raw_response_preview": self._preview_text(raw_content),
            "raw_response_text": raw_content,
            "response_format": "structured_json_workspace",
            "response_candidate_count": candidate_count,
            "response_selected_source": selected_source,
            "parsed_response": {
                "entrypoint": payload["entrypoint"],
                "declared_outputs": declared_outputs,
                "assumptions": assumptions,
                "solver_summary": payload["solver_summary"],
                "files": sorted(files.keys()),
            },
            "prompt_contract_version": self.LEGACY_WORKSPACE_PROMPT_VERSION,
        }
        return payload, metadata

    def _coerce_workspace_files(self, value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            files = {str(path): str(content) for path, content in value.items()}
        elif isinstance(value, list):
            files = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                path = item.get("path")
                content = item.get("content")
                if path:
                    files[str(path)] = str(content or "")
        else:
            files = {}
        if not files:
            raise ModelResponseParseError(
                "structured workspace response missing required field 'files'"
            )
        return files

    def _validate_workspace_paths(
        self,
        files: dict[str, str],
        entrypoint: str,
    ) -> list[str]:
        violations: list[str] = []
        for path in list(files.keys()) + [entrypoint]:
            normalized = path.replace("\\", "/").strip()
            if not normalized or normalized.startswith("/") or normalized.startswith("../"):
                violations.append(normalized or "<empty>")
                continue
            parts = [part for part in normalized.split("/") if part not in ("", ".")]
            if any(part == ".." for part in parts):
                violations.append(normalized)
                continue
            if not normalized.startswith("work/"):
                violations.append(normalized)
        return sorted(set(violations))

    def _write_workspace_files(
        self,
        workspace_root: Path,
        files: dict[str, str],
    ) -> list[str]:
        written: list[str] = []
        for relative_path, content in sorted(files.items()):
            target = workspace_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(relative_path)
        return written

    def _public_self_check(
        self,
        *,
        task_spec: dict[str, Any],
        runtime_paths: dict[str, Path],
        completed: subprocess.CompletedProcess[str],
        timed_out: bool,
        entrypoint: str,
        env: dict[str, str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        output_state = self._inspect_output_contract_state(
            task_spec=task_spec,
            runtime_paths=runtime_paths,
        )
        self_eval_completed, self_eval_timed_out = _run_subprocess(
            ["python", "evaluation/self_eval.py"],
            cwd=runtime_paths["runtime_root"],
            env=env,
            timeout_seconds=timeout_seconds,
        )
        parse_error = ""
        self_eval_payload: dict[str, Any]
        try:
            self_eval_payload = json.loads(self_eval_completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
            self_eval_payload = {
                "passed": False,
                "checks": [],
                "errors": ["invalid public self eval output"],
                "warnings": [],
            }
        if not isinstance(self_eval_payload, dict):
            parse_error = parse_error or "public self eval did not return a JSON object"
            self_eval_payload = {
                "passed": False,
                "checks": [],
                "errors": ["invalid public self eval output"],
                "warnings": [],
            }
        output_exists = output_state["primary_output_exists"]
        schema_warnings = list(output_state["schema_warnings"])
        if output_state["missing_outputs"] and not output_exists:
            schema_warnings = ["missing output artifact", *schema_warnings]
        completed_stdout = self._completed_process_text(completed.stdout)
        completed_stderr = self._completed_process_text(completed.stderr)
        self_eval_stdout = self._completed_process_text(self_eval_completed.stdout)
        self_eval_stderr = self._completed_process_text(self_eval_completed.stderr)
        public_self_eval_passed = (
            self_eval_completed.returncode == 0
            and not self_eval_timed_out
            and bool(self_eval_payload.get("passed", False))
        )
        return {
            "run_succeeded": completed.returncode == 0 and not timed_out,
            "returncode": completed.returncode,
            "timed_out": timed_out,
            "entrypoint": entrypoint,
            "output_exists": output_exists,
            "missing_outputs": list(output_state["missing_outputs"]),
            "missing_output_fields": list(output_state["missing_output_fields"]),
            "schema_warnings": schema_warnings,
            "stdout_tail": completed_stdout[-2000:],
            "stderr_tail": completed_stderr[-2000:],
            "public_self_eval_command": self._workspace_self_eval_command(),
            "public_self_eval_returncode": self_eval_completed.returncode,
            "public_self_eval_timed_out": self_eval_timed_out,
            "public_self_eval_parse_error": parse_error,
            "public_self_eval_passed": public_self_eval_passed,
            "public_self_eval_checks": list(self_eval_payload.get("checks", [])),
            "public_self_eval_errors": list(self_eval_payload.get("errors", [])),
            "public_self_eval_warnings": list(self_eval_payload.get("warnings", [])),
            "public_self_eval_stdout": self_eval_stdout,
            "public_self_eval_stderr": self_eval_stderr,
            "self_check_passed": (
                completed.returncode == 0
                and not timed_out
                and public_self_eval_passed
            ),
        }

    def _resolve_mock_response(self, mock_value: Any, round_index: int) -> str:
        if isinstance(mock_value, (list, tuple)):
            if not mock_value:
                return ""
            if round_index - 1 < len(mock_value):
                return str(mock_value[round_index - 1])
            return str(mock_value[-1])
        return str(mock_value)

    def _safe_read_text(self, path: Path) -> str:
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

class ClaudeAdapter(ClaudeWorkspaceAdapter):
    """Compatibility wrapper for the Claude-style workspace agent."""

    provider_name = "claude"
