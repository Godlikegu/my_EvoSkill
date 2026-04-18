"""Claude-backed registration draft helpers."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .executor import ClaudeSDKExecutionError, ClaudeWorkspaceAdapter
from .model_provider import ClaudeSDKAdapter
from .models import ExecutorSessionConfig, ModelConfig

REGISTRATION_INPUT_FILENAME = "registration_input.json"
_RESOURCE_DECLARATION_FIELDS = (
    "task_description_resources",
    "public_input_resources",
    "public_metadata_resources",
    "public_eval_resources",
    "evaluation_logic_resources",
    "hidden_reference_resources",
    "hidden_metric_config_resources",
)
_OPTIONAL_LIST_FIELDS = {
    "public_eval_resources",
    "hidden_reference_resources",
    "hidden_metric_config_resources",
}
_ALLOWED_AUTHORITIES = {"authoritative", "supplementary"}
_ALLOWED_METRIC_KINDS = {"standard", "script"}
_ALLOWED_OPERATORS = {"<=", ">="}


def resolve_registration_input_path(
    task_root: Path,
    registration_input_path: Optional[Path] = None,
) -> Path:
    """Resolve the task-local registration input path."""

    task_root = Path(task_root).resolve()
    if registration_input_path is None:
        return task_root / "evaluation" / REGISTRATION_INPUT_FILENAME
    candidate = Path(registration_input_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (task_root / candidate).resolve()


def load_registration_input(
    task_root: Path,
    *,
    registration_input_path: Optional[Path] = None,
) -> tuple[dict[str, Any], Path]:
    """Load registration_input.json from the task evaluation directory."""

    resolved_path = resolve_registration_input_path(
        task_root,
        registration_input_path=registration_input_path,
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"registration input not found: {resolved_path}")
    return json.loads(resolved_path.read_text(encoding="utf-8")), resolved_path


def validate_registration_input(payload: Mapping[str, Any]) -> list[str]:
    """Validate the user-supplied registration input payload."""

    errors: list[str] = []
    normalized = dict(payload or {})
    task_id = str(normalized.get("task_id", "") or "").strip()
    if not task_id:
        errors.append("registration_input.task_id is required")

    family = normalized.get("family")
    if family is not None and not isinstance(family, str):
        errors.append("registration_input.family must be a string when provided")

    for field_name in _RESOURCE_DECLARATION_FIELDS:
        required = field_name not in _OPTIONAL_LIST_FIELDS
        value = normalized.get(field_name)
        if value is None:
            if required:
                errors.append(f"registration_input.{field_name} is required")
            continue
        if not isinstance(value, list):
            errors.append(f"registration_input.{field_name} must be a list")
            continue
        for index, raw_item in enumerate(value):
            item = raw_item if isinstance(raw_item, Mapping) else {}
            prefix = f"registration_input.{field_name}[{index}]"
            path = str(item.get("path", "") or "").strip()
            authority = str(item.get("authority", "") or "").strip()
            notes = item.get("notes")
            if not path:
                errors.append(f"{prefix}.path is required")
            elif Path(path).is_absolute():
                errors.append(f"{prefix}.path must be relative")
            if authority not in _ALLOWED_AUTHORITIES:
                errors.append(
                    f"{prefix}.authority must be one of: {sorted(_ALLOWED_AUTHORITIES)}"
                )
            if notes is not None and not isinstance(notes, str):
                errors.append(f"{prefix}.notes must be a string when provided")

    pass_metrics = normalized.get("pass_metrics")
    if not isinstance(pass_metrics, list) or not pass_metrics:
        errors.append("registration_input.pass_metrics must be a non-empty list")
    else:
        for index, raw_metric in enumerate(pass_metrics):
            metric = raw_metric if isinstance(raw_metric, Mapping) else {}
            prefix = f"registration_input.pass_metrics[{index}]"
            name = str(metric.get("name", "") or "").strip()
            description = str(metric.get("description", "") or "").strip()
            kind = str(metric.get("kind", "") or "").strip()
            operator = str(metric.get("operator", "") or "").strip()
            threshold = metric.get("threshold")
            if not name:
                errors.append(f"{prefix}.name is required")
            if not description:
                errors.append(f"{prefix}.description is required")
            if kind not in _ALLOWED_METRIC_KINDS:
                errors.append(
                    f"{prefix}.kind must be one of: {sorted(_ALLOWED_METRIC_KINDS)}"
                )
            if operator not in _ALLOWED_OPERATORS:
                errors.append(
                    f"{prefix}.operator must be one of: {sorted(_ALLOWED_OPERATORS)}"
                )
            if threshold is None or isinstance(threshold, bool):
                errors.append(f"{prefix}.threshold is required")
            else:
                try:
                    float(threshold)
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.threshold must be numeric")
            result_key = metric.get("result_key")
            if result_key is not None and not isinstance(result_key, str):
                errors.append(f"{prefix}.result_key must be a string when provided")
            if kind == "script":
                source_path = str(metric.get("source_path", "") or "").strip()
                callable_name = str(metric.get("callable", "") or "").strip()
                if not source_path:
                    errors.append(f"{prefix}.source_path is required for script metrics")
                elif Path(source_path).is_absolute():
                    errors.append(f"{prefix}.source_path must be relative")
                if not callable_name:
                    errors.append(f"{prefix}.callable is required for script metrics")

    execution_hints = normalized.get("execution_hints")
    if execution_hints is not None and not isinstance(execution_hints, Mapping):
        errors.append("registration_input.execution_hints must be an object when provided")

    return sorted(dict.fromkeys(errors))


def normalize_registration_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize registration input for prompt construction and notes."""

    normalized = dict(payload or {})
    result: dict[str, Any] = {
        "task_id": str(normalized.get("task_id", "") or "").strip(),
        "family": str(normalized.get("family", "") or "").strip(),
    }
    for field_name in _RESOURCE_DECLARATION_FIELDS:
        entries: list[dict[str, str]] = []
        for raw_item in normalized.get(field_name, []) or []:
            if not isinstance(raw_item, Mapping):
                continue
            entries.append(
                {
                    "path": str(raw_item.get("path", "") or "").strip().replace("\\", "/"),
                    "authority": str(raw_item.get("authority", "") or "").strip(),
                    "notes": str(raw_item.get("notes", "") or "").strip(),
                }
            )
        result[field_name] = entries
    metrics: list[dict[str, Any]] = []
    for raw_metric in normalized.get("pass_metrics", []) or []:
        if not isinstance(raw_metric, Mapping):
            continue
        metric = {
            "name": str(raw_metric.get("name", "") or "").strip(),
            "description": str(raw_metric.get("description", "") or "").strip(),
            "kind": str(raw_metric.get("kind", "") or "").strip(),
            "operator": str(raw_metric.get("operator", "") or "").strip(),
            "threshold": raw_metric.get("threshold"),
        }
        for optional_name in ("source_path", "callable", "result_key"):
            value = str(raw_metric.get(optional_name, "") or "").strip()
            if value:
                metric[optional_name] = value.replace("\\", "/")
        metrics.append(metric)
    result["pass_metrics"] = metrics

    raw_hints = normalized.get("execution_hints") or {}
    execution_hints = raw_hints if isinstance(raw_hints, Mapping) else {}
    result["execution_hints"] = {
        "read_first": [
            str(item).replace("\\", "/").strip()
            for item in execution_hints.get("read_first", []) or []
            if str(item or "").strip()
        ],
        "suggested_entrypoint": str(
            execution_hints.get("suggested_entrypoint", "") or ""
        ).strip().replace("\\", "/"),
        "suggested_output_path": str(
            execution_hints.get("suggested_output_path", "") or ""
        ).strip().replace("\\", "/"),
    }
    return result


def registration_agent_output_schema() -> dict[str, Any]:
    """Return the JSON schema expected from the registration agent."""

    warning_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "field": {"type": "string"},
            "user_value": {},
            "agent_value": {},
            "evidence_paths": {"type": "array", "items": {"type": "string"}},
            "message": {"type": "string"},
        },
        "required": ["message"],
        "additionalProperties": True,
    }
    return {
        "type": "object",
        "properties": {
            "registration_contract_draft": {"type": "object"},
            "contract_generation_notes": {
                "type": "object",
                "properties": {
                    "agent_summary": {"type": "string"},
                    "warnings": {"type": "array", "items": warning_schema},
                    "open_questions": {"type": "array", "items": {"type": "string"}},
                    "declared_inputs_used": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "resource_validation": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                },
                "additionalProperties": True,
            },
            "judge_recommendations": {
                "type": "object",
                "properties": {
                    "metrics": {"type": "array", "items": {"type": "object"}},
                    "generation_hints": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
        },
        "required": ["registration_contract_draft"],
        "additionalProperties": True,
    }


def coerce_registration_agent_output(payload: Any) -> dict[str, Any]:
    """Normalize and minimally validate agent output payload."""

    if not isinstance(payload, Mapping):
        raise RuntimeError("registration agent did not return a JSON object")

    draft = payload.get("registration_contract_draft")
    if not isinstance(draft, Mapping) or not dict(draft):
        raise RuntimeError("registration agent output is missing registration_contract_draft")

    notes = payload.get("contract_generation_notes")
    notes_payload = dict(notes) if isinstance(notes, Mapping) else {}
    warnings: list[dict[str, Any]] = []
    for raw_warning in notes_payload.get("warnings", []) or []:
        if isinstance(raw_warning, Mapping):
            warning = {
                "type": str(raw_warning.get("type", "") or "").strip()
                or "agent_warning",
                "field": str(raw_warning.get("field", "") or "").strip(),
                "user_value": raw_warning.get("user_value"),
                "agent_value": raw_warning.get("agent_value"),
                "evidence_paths": [
                    str(item).replace("\\", "/").strip()
                    for item in raw_warning.get("evidence_paths", []) or []
                    if str(item or "").strip()
                ],
                "message": str(raw_warning.get("message", "") or "").strip(),
            }
            if warning["message"]:
                warnings.append(warning)
            continue
        text = str(raw_warning or "").strip()
        if not text:
            continue
        warnings.append(
            {
                "type": "agent_warning",
                "field": "",
                "user_value": None,
                "agent_value": None,
                "evidence_paths": [],
                "message": text,
            }
        )

    judge = payload.get("judge_recommendations")
    judge_payload = dict(judge) if isinstance(judge, Mapping) else {}
    judge_metrics: list[dict[str, Any]] = []
    for raw_metric in judge_payload.get("metrics", []) or []:
        if not isinstance(raw_metric, Mapping):
            continue
        judge_metrics.append(
            {
                "name": str(raw_metric.get("name", "") or "").strip(),
                "kind": str(raw_metric.get("kind", "") or "").strip(),
                "source_path": str(raw_metric.get("source_path", "") or "").strip().replace(
                    "\\", "/"
                ),
                "callable": str(raw_metric.get("callable", "") or "").strip(),
                "result_key": str(raw_metric.get("result_key", "") or "").strip(),
                "rationale": str(raw_metric.get("rationale", "") or "").strip(),
            }
        )

    return {
        "registration_contract_draft": dict(draft),
        "contract_generation_notes": {
            "agent_summary": str(notes_payload.get("agent_summary", "") or "").strip(),
            "warnings": warnings,
            "open_questions": [
                str(item).strip()
                for item in notes_payload.get("open_questions", []) or []
                if str(item or "").strip()
            ],
            "declared_inputs_used": [
                str(item).replace("\\", "/").strip()
                for item in notes_payload.get("declared_inputs_used", []) or []
                if str(item or "").strip()
            ],
            "resource_validation": [
                dict(item)
                for item in notes_payload.get("resource_validation", []) or []
                if isinstance(item, Mapping)
            ],
        },
        "judge_recommendations": {
            "metrics": judge_metrics,
            "generation_hints": [
                str(item).strip()
                for item in judge_payload.get("generation_hints", []) or []
                if str(item or "").strip()
            ],
        },
    }


def build_registration_agent_system_prompt(
    registration_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the fixed system prompt for the registration agent."""

    normalized_input = normalize_registration_input(registration_input)
    append_text = "\n".join(
        [
            "You are the MyEvoSkill task-registration agent.",
            "Your job is to inspect the task directory and produce a structured registration contract draft.",
            "Do not restate README or source file contents verbatim.",
            "You may explore files in the task directory using read-only tools.",
            "The user-provided registration input is high-priority guidance for resource semantics and evaluation conventions.",
            "If your file-based inference disagrees with a user declaration, do not silently overwrite it.",
            "Report the disagreement in contract_generation_notes.warnings as a structured object with keys: type, field, user_value, agent_value, evidence_paths, message.",
            "contract_generation_notes.warnings must always be an explicit array; return [] when there are no conflicts.",
            "Stop as soon as you can return a schema-valid JSON object.",
            "Return one JSON object whose top-level keys are registration_contract_draft, contract_generation_notes, judge_recommendations.",
            "",
            "User-declared registration input:",
            json.dumps(normalized_input, indent=2, sort_keys=True, ensure_ascii=False),
        ]
    )
    return {
        "type": "preset",
        "preset": "claude_code",
        "append": append_text,
    }


def build_registration_agent_prompt(
    task_root: Path,
    registration_input: Mapping[str, Any],
    *,
    registration_input_path: Path,
    repair_feedback: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build the user prompt for the registration agent query."""

    normalized_input = normalize_registration_input(registration_input)
    read_first = normalized_input.get("execution_hints", {}).get("read_first", [])
    read_first_text = ", ".join(read_first) if read_first else "(no explicit read-first list)"
    lines = [
        "Inspect the task directory and return ONLY a JSON object that matches the provided schema.",
        f"task_root={Path(task_root).resolve()}",
        f"registration_input_path={registration_input_path}",
        f"Declared read-first resources: {read_first_text}",
        "",
        "Expectations:",
        "1. Produce registration_contract_draft with task_id, family, resources, output_contract, judge_contract, execution_conventions.",
        "2. Use resource roles from the contract vocabulary: task_description, public_input_data, public_metadata, public_eval_script, metric_helper, hidden_reference, hidden_metric_config.",
        "3. Every judge_contract.metrics entry must define pass_condition {operator, threshold}. Do not use threshold_key or a top-level threshold field.",
        "4. Metric input mappings must preserve shape semantics. Only use index/scalar_index when the referenced output_field or resource is explicitly batched or scalar-like.",
        "5. contract_generation_notes.warnings must contain structured conflicts when your inference disagrees with the user's declaration, and must be [] when there are no conflicts.",
        "6. judge_recommendations should explain the metric source file, callable, result_key, and why it is recommended.",
        "7. Do not emit prose before or after the JSON object.",
    ]
    if repair_feedback:
        lines.extend(
            [
                "",
                "Repair round:",
                "8. The previous attempt failed local registration-contract validation. Fix the returned JSON using the feedback below.",
                "9. Keep already-correct parts when possible instead of rewriting the contract arbitrarily.",
                "10. Return a complete replacement JSON object that satisfies the schema and the validation feedback.",
                json.dumps(
                    dict(repair_feedback),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            ]
        )
    return "\n".join(lines)


class _RegistrationClaudeAdapter(ClaudeWorkspaceAdapter):
    """Thin Claude workspace adapter reused for registration-agent protocols."""

    WORKSPACE_PROMPT_VERSION = "v1_registration_agent"
    DEFAULT_ALLOWED_TOOLS = ["Read", "Glob", "Grep"]
    DEFAULT_DISALLOWED_TOOLS = ["Write", "Bash", "WebFetch", "WebSearch", "TodoWrite"]

    def _workspace_summary_schema(self) -> dict[str, Any]:
        return registration_agent_output_schema()

    def _coerce_sdk_summary(self, payload: Any) -> dict[str, Any]:
        return coerce_registration_agent_output(payload)


def _extract_registration_candidate_from_message(
    adapter: _RegistrationClaudeAdapter,
    message: Any,
    *,
    assistant_only: bool = True,
) -> Optional[dict[str, Any]]:
    structured_output = getattr(message, "structured_output", None)
    if isinstance(structured_output, Mapping):
        try:
            return adapter._coerce_sdk_summary(structured_output)
        except RuntimeError:
            pass

    role = str(getattr(message, "role", "") or "").strip().lower()
    message_type = message.__class__.__name__.lower()
    assistant_like = (
        role == "assistant"
        or "assistant" in message_type
        or "resultmessage" in message_type
    )
    if assistant_only and not assistant_like:
        return None

    serialized = adapter._json_safe(message)
    text_candidates: list[str] = []

    def _collect_strings(node: Any, *, field_name: str = "") -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_name = str(key or "")
                if key_name in {"result", "text"} and isinstance(value, str):
                    text_candidates.append(value)
                    continue
                if key_name == "content":
                    if isinstance(value, str):
                        text_candidates.append(value)
                    else:
                        _collect_strings(value, field_name=key_name)
                    continue
                _collect_strings(value, field_name=key_name)
            return
        if isinstance(node, list):
            for item in node:
                _collect_strings(item, field_name=field_name)

    _collect_strings(serialized)
    for candidate_text in reversed(text_candidates):
        parsed, _, _, _ = adapter._extract_structured_payload(candidate_text)
        if not isinstance(parsed, Mapping):
            continue
        try:
            return adapter._coerce_sdk_summary(parsed)
        except RuntimeError:
            continue
    return None


async def _consume_registration_agent_response(
    adapter: _RegistrationClaudeAdapter,
    message_stream: Any,
    *,
    total_timeout_seconds: int,
    hook_events: Sequence[dict[str, Any]],
    result_message_type: Any,
) -> dict[str, Any]:
    start = time.monotonic()
    iterator = message_stream.__aiter__()
    messages: list[Any] = []
    result_text = ""

    while True:
        remaining_total = total_timeout_seconds - (time.monotonic() - start)
        if remaining_total <= 0:
            diagnostics = adapter._build_protocol_diagnostics(
                messages,
                hook_events=hook_events,
                timeout_occurred=True,
            )
            raise ClaudeSDKExecutionError(
                f"Claude SDK registration agent timed out after {total_timeout_seconds} seconds",
                error_type="request_timeout",
                sdk_messages=adapter._serialize_sdk_messages(messages),
                result_text=result_text,
                diagnostics=diagnostics,
            )

        try:
            message = await asyncio.wait_for(
                iterator.__anext__(),
                timeout=max(remaining_total, 0.01),
            )
        except StopAsyncIteration:
            diagnostics = adapter._build_protocol_diagnostics(
                messages,
                hook_events=hook_events,
                timeout_occurred=False,
            )
            raise ClaudeSDKExecutionError(
                "Claude SDK registration agent response stream ended without a valid structured result",
                error_type="missing_result_message",
                sdk_messages=adapter._serialize_sdk_messages(messages),
                result_text=result_text,
                diagnostics=diagnostics,
            )
        except asyncio.TimeoutError as exc:
            diagnostics = adapter._build_protocol_diagnostics(
                messages,
                hook_events=hook_events,
                timeout_occurred=True,
            )
            raise ClaudeSDKExecutionError(
                f"Claude SDK registration agent timed out after {total_timeout_seconds} seconds",
                error_type="request_timeout",
                sdk_messages=adapter._serialize_sdk_messages(messages),
                result_text=result_text,
                diagnostics=diagnostics,
            ) from exc

        messages.append(message)
        serialized_messages = adapter._serialize_sdk_messages(messages)

        if adapter._is_claude_result_message(message, result_message_type=result_message_type):
            result_text = adapter._extract_sdk_result_text(message)
            diagnostics = adapter._build_protocol_diagnostics(
                messages,
                hook_events=hook_events,
                result_message=message,
                timeout_occurred=False,
            )
            try:
                summary_payload = adapter._coerce_sdk_summary(
                    adapter._extract_sdk_summary(message, serialized_messages)
                )
            except ClaudeSDKExecutionError:
                raise
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

        candidate = _extract_registration_candidate_from_message(adapter, message)
        if candidate is not None:
            diagnostics = adapter._build_protocol_diagnostics(
                messages,
                hook_events=hook_events,
                timeout_occurred=False,
            )
            diagnostics["external_completion"] = {
                "policy": "result_message_with_schema_fallback",
                "matched_message_type": message.__class__.__name__,
            }
            return {
                "messages": serialized_messages,
                "summary": candidate,
                "result_text": result_text,
                "sdk_diagnostics": diagnostics,
                "completion_source": "external_registration_schema",
            }


async def _run_registration_agent_async(
    *,
    task_root: Path,
    registration_input: Mapping[str, Any],
    registration_input_path: Path,
    timeout_seconds: int,
    repair_feedback: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        from claude_agent_sdk.types import ResultMessage
    except Exception as exc:  # pragma: no cover - import depends on optional SDK
        raise RuntimeError(
            "claude_agent_sdk is required for registration draft generation"
        ) from exc

    model = ModelConfig(
        provider_name="claude-sdk",
        model_name="",
        api_key_env="ANTHROPIC_API_KEY",
        temperature=0.0,
    )
    provider = ClaudeSDKAdapter(model)
    sdk_env = provider.build_sdk_env()
    adapter = _RegistrationClaudeAdapter()
    session_config = ExecutorSessionConfig(
        run_id=f"registration-draft-{task_root.name}-{int(time.time())}",
        env_hash="registration-agent",
        workspace_root=task_root,
        model_config=model,
        provider_extras={
            "claude_effort": "low",
            "claude_max_turns": 40,
        },
    )
    hook_events: list[dict[str, Any]] = []
    system_prompt = build_registration_agent_system_prompt(registration_input)
    prompt = build_registration_agent_prompt(
        task_root,
        registration_input,
        registration_input_path=registration_input_path,
        repair_feedback=repair_feedback,
    )
    options_kwargs = adapter._build_claude_sdk_options_kwargs(
        session_config=session_config,
        workspace=task_root,
        system_prompt=system_prompt,
        stop_oracle="public_self_eval",
        mcp_servers=None,
        sdk_env=sdk_env,
    )
    options_kwargs["hooks"] = adapter._build_claude_sdk_hooks(hook_events)
    options = ClaudeAgentOptions(**options_kwargs)
    resolved_model_name = adapter._resolved_model_name(session_config)
    if resolved_model_name:
        options.model = resolved_model_name

    async with ClaudeSDKClient(options) as client:
        try:
            await client.query(prompt)
            result = await _consume_registration_agent_response(
                adapter,
                client.receive_response(),
                total_timeout_seconds=timeout_seconds,
                hook_events=hook_events,
                result_message_type=ResultMessage,
            )
        except ClaudeSDKExecutionError as exc:
            diagnostics = dict(exc.diagnostics)
            diagnostics.setdefault("hook_events", [adapter._json_safe(item) for item in hook_events])
            vendor_session_ref = adapter._build_vendor_session_ref(
                client,
                task_root,
                sdk_messages=exc.sdk_messages,
            )
            diagnostics["vendor_session_ref"] = vendor_session_ref
            raise ClaudeSDKExecutionError(
                str(exc),
                error_type=exc.error_type,
                sdk_messages=exc.sdk_messages,
                result_text=exc.result_text,
                diagnostics=diagnostics,
            ) from exc
        if "vendor_session_ref" not in result:
            result["vendor_session_ref"] = adapter._build_vendor_session_ref(
                client,
                task_root,
                sdk_messages=result.get("messages", []),
            )
        return result


def run_registration_agent(
    task_root: Path,
    *,
    registration_input: Mapping[str, Any],
    registration_input_path: Path,
    timeout_seconds: int = 300,
    repair_feedback: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run the mandatory Claude registration agent and return structured output."""

    task_root = Path(task_root).resolve()
    registration_input_path = Path(registration_input_path).resolve()
    return asyncio.run(
        _run_registration_agent_async(
            task_root=task_root,
            registration_input=registration_input,
            registration_input_path=registration_input_path,
            timeout_seconds=timeout_seconds,
            repair_feedback=repair_feedback,
        )
    )


__all__ = [
    "REGISTRATION_INPUT_FILENAME",
    "build_registration_agent_prompt",
    "build_registration_agent_system_prompt",
    "coerce_registration_agent_output",
    "load_registration_input",
    "normalize_registration_input",
    "registration_agent_output_schema",
    "resolve_registration_input_path",
    "run_registration_agent",
    "validate_registration_input",
]
