"""Contract-driven task registration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .envs import EnvManager, build_task_env_spec, resolve_env_cache_root
from .judge_runner import invoke_judge_runner
from .manifest_bootstrap import (
    DEFAULT_JUDGE_ADAPTER_PATH,
    DEFAULT_METRICS_CONFIG_PATH,
    DEFAULT_OUTPUT_PATH,
    _build_data_summary,
    _build_evaluation_summary,
    _build_output_detection,
    _build_public_eval_spec,
    _build_public_policy,
    _infer_same_shape_fields,
    _read_readme_info,
    _registry_tasks_root,
    _relative_source_task_dir,
    _resolve_output_root,
)
from .registration_agent import (
    REGISTRATION_INPUT_FILENAME,
    load_registration_input,
    normalize_registration_input,
    run_registration_agent,
    validate_registration_input,
)
from .models import ContractDraftResult, EnvCacheRecord, TaskRegistrationResult
from .task_runtime import (
    coerce_runtime_layout,
    coerce_runtime_policy,
)

CONTRACT_DRAFT_FILENAME = "registration_contract.draft.json"
CONTRACT_FILENAME = "registration_contract.json"
CONTRACT_NOTES_FILENAME = "contract_generation.notes.json"
GENERATED_JUDGE_SENTINEL = "MYEVOSKILL_GENERATED_READY_JUDGE_V1"
READY_JUDGE_MARKER = "GENERATED_JUDGE_READY = True"

RESOURCE_ROLES = {
    "task_description",
    "public_input_data",
    "public_metadata",
    "public_eval_script",
    "metric_helper",
    "hidden_reference",
    "hidden_metric_config",
}
RESOURCE_VISIBILITIES = {"public", "private"}
RESOURCE_AUTHORITIES = {"authoritative", "supplementary"}


def registration_contract_paths(task_root: Path) -> dict[str, Path]:
    """Resolve all task-local registration contract artifacts."""

    evaluation_dir = Path(task_root).resolve() / "evaluation"
    return {
        "evaluation_dir": evaluation_dir,
        "input_path": evaluation_dir / REGISTRATION_INPUT_FILENAME,
        "draft_path": evaluation_dir / CONTRACT_DRAFT_FILENAME,
        "confirmed_path": evaluation_dir / CONTRACT_FILENAME,
        "notes_path": evaluation_dir / CONTRACT_NOTES_FILENAME,
        "judge_path": Path(task_root).resolve() / DEFAULT_JUDGE_ADAPTER_PATH,
    }


def task_requirements_path(task_root: Path) -> Path:
    """Resolve one task-local requirements.txt path."""

    return Path(task_root).resolve() / "requirements.txt"


def ensure_task_runtime_env(
    task_root: Path,
    *,
    output_root: Path,
    task_id: str,
    family: str,
) -> EnvCacheRecord:
    """Build or reuse the cached task runtime environment from requirements.txt."""

    requirements_path = task_requirements_path(task_root)
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements.txt not found: {requirements_path}")
    manager = EnvManager(resolve_env_cache_root(output_root))
    spec = build_task_env_spec(
        task_id=task_id,
        family=family,
        requirements_path=requirements_path,
    )
    return manager.ensure_env(spec)


def runtime_env_manifest_payload(
    cache_record: EnvCacheRecord,
    *,
    output_root: Path,
    task_root: Path,
) -> dict[str, Any]:
    """Serialize cached runtime environment metadata into the task manifest."""

    return {
        "backend": "venv_pip",
        "env_hash": cache_record.env_hash,
        "requirements_path": str(task_requirements_path(task_root)),
        "python_executable": str(cache_record.python_executable),
        "ready": bool(cache_record.ready),
        "build_log_path": str(cache_record.build_log_path),
        "install_report_path": str(cache_record.install_report_path),
    }


def ensure_manifest_runtime_env(
    manifest: Mapping[str, Any],
    *,
    task_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Ensure the runtime environment declared in a ready manifest exists locally."""

    task_id = str(manifest.get("task_id", "") or Path(task_root).name)
    family = str(manifest.get("family", "") or "")
    cache_record = ensure_task_runtime_env(
        task_root,
        output_root=output_root,
        task_id=task_id,
        family=family,
    )
    runtime_env = runtime_env_manifest_payload(
        cache_record,
        output_root=output_root,
        task_root=task_root,
    )
    expected_hash = str((manifest.get("runtime_env") or {}).get("env_hash", "") or "").strip()
    if expected_hash and expected_hash != runtime_env["env_hash"]:
        raise RuntimeError(
            "live run refused: manifest runtime_env.env_hash does not match rebuilt environment"
        )
    return runtime_env


def create_registration_contract_draft(
    task_root: Path,
    *,
    output_root: Path,
    registration_input_path: Optional[Path] = None,
) -> ContractDraftResult:
    """Generate a registration contract draft from one raw task directory."""

    task_root = Path(task_root).resolve()
    output_root = _resolve_output_root(output_root)
    task_id = task_root.name
    paths = registration_contract_paths(task_root)
    paths["evaluation_dir"].mkdir(parents=True, exist_ok=True)
    registration_input, resolved_input_path = load_registration_input(
        task_root,
        registration_input_path=registration_input_path,
    )
    input_validation_errors = validate_registration_input(registration_input)
    if input_validation_errors:
        raise ValueError(
            "registration_input.json validation failed:\n- "
            + "\n- ".join(input_validation_errors)
        )
    normalized_input = normalize_registration_input(registration_input)
    declared_task_id = str(normalized_input.get("task_id", "") or "").strip()
    if declared_task_id and declared_task_id != task_id:
        raise ValueError(
            "registration_input.json task_id does not match task directory name: "
            f"{declared_task_id!r} != {task_id!r}"
        )

    attempt_summaries: list[dict[str, Any]] = []
    repair_feedback: Optional[dict[str, Any]] = None
    final_agent_result: Optional[dict[str, Any]] = None
    contract: dict[str, Any] = {}

    for attempt_index in (1, 2):
        try:
            agent_result = run_registration_agent(
                task_root,
                registration_input=normalized_input,
                registration_input_path=resolved_input_path,
                repair_feedback=repair_feedback,
            )
        except Exception as exc:
            attempt_summaries.append(
                _registration_attempt_summary(
                    attempt_index=attempt_index,
                    error=exc,
                )
            )
            if attempt_index == 1 or repair_feedback is None:
                raise
            raise ValueError(
                _format_registration_draft_attempt_failure(attempt_summaries)
            ) from exc

        contract = dict(agent_result["summary"]["registration_contract_draft"])
        contract_validation_errors = validate_registration_contract(contract, require_complete=False)
        attempt_summaries.append(
            _registration_attempt_summary(
                attempt_index=attempt_index,
                agent_result=agent_result,
                validation_errors=contract_validation_errors,
            )
        )
        if not contract_validation_errors:
            final_agent_result = agent_result
            break
        if attempt_index == 1:
            repair_feedback = _build_registration_repair_feedback(
                validation_errors=contract_validation_errors,
                agent_result=agent_result,
            )
            continue
        raise ValueError(_format_registration_draft_attempt_failure(attempt_summaries))

    if final_agent_result is None:
        raise RuntimeError("registration draft generation did not produce a valid result")

    notes_output = dict(final_agent_result["summary"].get("contract_generation_notes") or {})
    judge_recommendations = dict(final_agent_result["summary"].get("judge_recommendations") or {})
    structured_warnings = list(notes_output.get("warnings", []) or [])
    warning_messages = _warning_messages(structured_warnings)
    missing_items = _contract_missing_items(contract)
    notes_payload = _build_notes_payload(
        task_id=task_id,
        task_root=task_root,
        output_root=output_root,
        registration_input_path=resolved_input_path,
        registration_input=normalized_input,
        draft_path=paths["draft_path"],
        confirmed_path=paths["confirmed_path"],
        manifest_path=None,
        judge_path=None,
        contract=contract,
        missing_items=missing_items,
        warnings=structured_warnings,
        completion_source=str(final_agent_result.get("completion_source", "") or ""),
        vendor_session_ref=dict(final_agent_result.get("vendor_session_ref") or {}),
        contract_generation_notes=notes_output,
        judge_recommendations=judge_recommendations,
        judge_validation_checks=[],
        contract_status="draft_generated",
        judge_status="not_started",
        attempt_summaries=attempt_summaries,
    )

    paths["draft_path"].write_text(
        json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["notes_path"].write_text(
        json.dumps(notes_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_registry_notes_copy(task_id, output_root=output_root, payload=notes_payload)
    return ContractDraftResult(
        task_id=task_id,
        draft_path=paths["draft_path"],
        notes_path=paths["notes_path"],
        missing_items=missing_items,
        warnings=warning_messages,
        attempt_count=len(attempt_summaries),
        attempt_summaries=attempt_summaries,
    )


def _build_registration_repair_feedback(
    *,
    validation_errors: Sequence[str],
    agent_result: Mapping[str, Any],
) -> dict[str, Any]:
    summary = dict(agent_result.get("summary") or {})
    notes = dict(summary.get("contract_generation_notes") or {})
    return {
        "previous_attempt": {
            "validation_errors": list(validation_errors),
            "registration_contract_draft": dict(summary.get("registration_contract_draft") or {}),
            "contract_generation_notes": {
                "warnings": list(notes.get("warnings", []) or []),
                "open_questions": list(notes.get("open_questions", []) or []),
            },
            "completion_source": str(agent_result.get("completion_source", "") or ""),
        }
    }


def _registration_attempt_summary(
    *,
    attempt_index: int,
    agent_result: Optional[Mapping[str, Any]] = None,
    validation_errors: Sequence[str] = (),
    error: Optional[BaseException] = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"attempt": int(attempt_index)}
    if agent_result is not None:
        summary["completion_source"] = str(agent_result.get("completion_source", "") or "")
        summary["vendor_session_ref"] = dict(agent_result.get("vendor_session_ref") or {})
        diagnostics = dict(agent_result.get("sdk_diagnostics") or {})
        if diagnostics.get("external_completion"):
            summary["external_completion"] = dict(diagnostics.get("external_completion") or {})
    if validation_errors:
        summary["status"] = "invalid_contract"
        summary["validation_errors"] = list(validation_errors)
    elif error is not None:
        summary["status"] = "agent_error"
        summary["error_type"] = type(error).__name__
        summary["error_message"] = str(error)
    else:
        summary["status"] = "succeeded"
    return summary


def _format_registration_draft_attempt_failure(
    attempt_summaries: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["registration agent failed to produce a valid registration_contract_draft:"]
    for raw_attempt in attempt_summaries:
        attempt = dict(raw_attempt or {})
        attempt_index = int(attempt.get("attempt", 0) or 0)
        status = str(attempt.get("status", "") or "unknown")
        lines.append(f"- attempt {attempt_index}: status={status}")
        completion_source = str(attempt.get("completion_source", "") or "").strip()
        if completion_source:
            lines.append(f"  completion_source={completion_source}")
        validation_errors = [
            str(item).strip()
            for item in attempt.get("validation_errors", []) or []
            if str(item or "").strip()
        ]
        for error_text in validation_errors:
            lines.append(f"  validation_error={error_text}")
        error_message = str(attempt.get("error_message", "") or "").strip()
        if error_message:
            lines.append(f"  error={error_message}")
    return "\n".join(lines)


def load_registration_contract(
    task_root: Path,
    *,
    confirmed: bool = True,
) -> dict[str, Any]:
    """Load a registration contract from the task-local evaluation directory."""

    paths = registration_contract_paths(task_root)
    target = paths["confirmed_path"] if confirmed else paths["draft_path"]
    if not target.exists():
        raise FileNotFoundError(f"registration contract not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def validate_registration_contract(
    contract: Mapping[str, Any],
    *,
    require_complete: bool,
) -> list[str]:
    """Validate a registration contract payload."""

    errors: list[str] = []
    payload = dict(contract or {})
    for field_name in (
        "task_id",
        "family",
        "resources",
        "output_contract",
        "judge_contract",
        "execution_conventions",
    ):
        if field_name not in payload:
            errors.append(f"missing required field: {field_name}")

    if require_complete and _contains_placeholder_token(payload.get("family")):
        errors.append("family must be confirmed in registration_contract.json")

    resources = payload.get("resources")
    if not isinstance(resources, list) or not resources:
        errors.append("resources must be a non-empty list")
    else:
        for index, raw_resource in enumerate(resources):
            resource = raw_resource if isinstance(raw_resource, Mapping) else {}
            prefix = f"resources[{index}]"
            path = str(resource.get("path", "") or "").strip()
            role = str(resource.get("role", "") or "").strip()
            visibility = str(resource.get("visibility", "") or "").strip()
            semantics = str(resource.get("semantics", "") or "").strip()
            authority = str(resource.get("authority", "") or "").strip()
            if not path or Path(path).is_absolute():
                errors.append(f"{prefix}.path must be a non-empty relative path")
            if role not in RESOURCE_ROLES:
                errors.append(f"{prefix}.role must be one of: {sorted(RESOURCE_ROLES)}")
            if visibility not in RESOURCE_VISIBILITIES:
                errors.append(
                    f"{prefix}.visibility must be one of: {sorted(RESOURCE_VISIBILITIES)}"
                )
            if authority not in RESOURCE_AUTHORITIES:
                errors.append(
                    f"{prefix}.authority must be one of: {sorted(RESOURCE_AUTHORITIES)}"
                )
            if not semantics:
                errors.append(f"{prefix}.semantics is required")

    output_contract = payload.get("output_contract")
    if not isinstance(output_contract, Mapping):
        errors.append("output_contract must be an object")
    else:
        for field_name in ("path", "format", "required_fields", "numeric_fields", "same_shape_fields"):
            if field_name not in output_contract:
                errors.append(f"missing output_contract.{field_name}")
        if require_complete:
            for field_name in ("path", "format", "required_fields"):
                if _contains_placeholder_token(output_contract.get(field_name)):
                    errors.append(f"output_contract.{field_name} must be confirmed")

    judge_contract = payload.get("judge_contract")
    if not isinstance(judge_contract, Mapping):
        errors.append("judge_contract must be an object")
    else:
        metrics = judge_contract.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            errors.append("judge_contract.metrics must be a non-empty list")
        else:
            for index, raw_metric in enumerate(metrics):
                metric = raw_metric if isinstance(raw_metric, Mapping) else {}
                prefix = f"judge_contract.metrics[{index}]"
                name = str(metric.get("name", "") or "").strip()
                kind = str(metric.get("kind", "") or "").strip()
                description = str(metric.get("description", "") or "").strip()
                pass_condition = metric.get("pass_condition")
                if not name:
                    errors.append(f"{prefix}.name is required")
                if kind not in {"standard", "script"}:
                    errors.append(f"{prefix}.kind must be either 'standard' or 'script'")
                if not description:
                    errors.append(f"{prefix}.description is required")
                if not isinstance(pass_condition, Mapping):
                    errors.append(f"{prefix}.pass_condition is required")
                    pass_condition = None
                if "threshold_key" in metric:
                    errors.append(
                        f"{prefix}.threshold_key is not supported; use pass_condition.threshold"
                    )
                if "threshold" in metric:
                    errors.append(
                        f"{prefix}.threshold is not supported; use pass_condition.threshold"
                    )
                if isinstance(pass_condition, Mapping):
                    operator = str(pass_condition.get("operator", "") or "").strip()
                    threshold = pass_condition.get("threshold")
                    if operator not in {"<=", ">="}:
                        errors.append(
                            f"{prefix}.pass_condition.operator must be one of: ['<=', '>=']"
                        )
                    if threshold is None or isinstance(threshold, bool):
                        errors.append(f"{prefix}.pass_condition.threshold must be numeric")
                    else:
                        try:
                            float(threshold)
                        except (TypeError, ValueError):
                            errors.append(f"{prefix}.pass_condition.threshold must be numeric")
                if kind == "script":
                    if not str(metric.get("source_path", "") or "").strip():
                        errors.append(f"{prefix}.source_path is required for script metrics")
                    if not str(metric.get("callable", "") or "").strip():
                        errors.append(f"{prefix}.callable is required for script metrics")
                    inputs = metric.get("inputs")
                    if require_complete and not isinstance(inputs, Mapping):
                        errors.append(f"{prefix}.inputs must be provided for script metrics")
                    if require_complete and _contains_placeholder_token(metric.get("result_key")):
                        errors.append(f"{prefix}.result_key must be confirmed for script metrics")
                if kind == "standard" and require_complete:
                    if _contains_placeholder_token(metric.get("mode")):
                        errors.append(f"{prefix}.mode must be confirmed for standard metrics")
                    if _contains_placeholder_token(metric.get("output_field")):
                        errors.append(f"{prefix}.output_field must be confirmed for standard metrics")
                    if _contains_placeholder_token(metric.get("reference_resource_path")):
                        errors.append(
                            f"{prefix}.reference_resource_path must be confirmed for standard metrics"
                        )
                    if _contains_placeholder_token(metric.get("reference_field")):
                        errors.append(
                            f"{prefix}.reference_field must be confirmed for standard metrics"
                        )
                if require_complete and (
                    not isinstance(pass_condition, Mapping)
                    or _contains_placeholder_token(pass_condition.get("operator"))
                    or _contains_placeholder_token(pass_condition.get("threshold"))
                ):
                    errors.append(f"{prefix}.pass_condition must be confirmed")

    execution = payload.get("execution_conventions")
    if not isinstance(execution, Mapping):
        errors.append("execution_conventions must be an object")
    else:
        for field_name in ("read_first", "readable_paths", "writable_paths", "entrypoint"):
            if field_name not in execution:
                errors.append(f"missing execution_conventions.{field_name}")
        if require_complete:
            for field_name in ("read_first", "readable_paths", "writable_paths", "entrypoint"):
                if _contains_placeholder_token(execution.get(field_name)):
                    errors.append(f"execution_conventions.{field_name} must be confirmed")

    return sorted(dict.fromkeys(errors))


def _validate_task_local_file_reference(
    task_root: Path,
    *,
    field_name: str,
    raw_path: Any,
) -> Optional[str]:
    normalized = str(raw_path or "").strip().replace("\\", "/")
    if not normalized:
        return None
    candidate = Path(normalized)
    if candidate.is_absolute():
        return f"{field_name} must be a relative task-local path: {normalized}"
    resolved_root = Path(task_root).resolve()
    resolved_path = (resolved_root / candidate).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return f"{field_name} escapes the task root: {normalized}"
    if not resolved_path.exists():
        return f"{field_name} does not exist: {normalized}"
    if not resolved_path.is_file():
        return f"{field_name} must point to a file: {normalized}"
    return None


def _iter_metric_input_paths(
    node: Any,
    *,
    prefix: str,
) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if isinstance(node, Mapping):
        resource_path = str(node.get("resource_path", "") or "").strip()
        if resource_path:
            paths.append((f"{prefix}.resource_path", resource_path))
        reference_path = str(node.get("reference_resource_path", "") or "").strip()
        if reference_path:
            paths.append((f"{prefix}.reference_resource_path", reference_path))
        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_iter_metric_input_paths(value, prefix=child_prefix))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            paths.extend(_iter_metric_input_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def validate_registration_contract_task_paths(
    task_root: Path,
    contract: Mapping[str, Any],
) -> list[str]:
    """Validate confirmed contract references that must resolve inside task_root."""

    errors: list[str] = []
    task_root = Path(task_root).resolve()
    for index, raw_resource in enumerate(contract.get("resources", []) or []):
        if not isinstance(raw_resource, Mapping):
            continue
        error = _validate_task_local_file_reference(
            task_root,
            field_name=f"resources[{index}].path",
            raw_path=raw_resource.get("path"),
        )
        if error:
            errors.append(error)

    for index, raw_metric in enumerate((contract.get("judge_contract") or {}).get("metrics", []) or []):
        if not isinstance(raw_metric, Mapping):
            continue
        prefix = f"judge_contract.metrics[{index}]"
        for field_name in ("source_path", "source_hint", "reference_resource_path"):
            error = _validate_task_local_file_reference(
                task_root,
                field_name=f"{prefix}.{field_name}",
                raw_path=raw_metric.get(field_name),
            )
            if error:
                errors.append(error)
        for nested_field, nested_path in _iter_metric_input_paths(
            raw_metric.get("inputs"),
            prefix=f"{prefix}.inputs",
        ):
            error = _validate_task_local_file_reference(
                task_root,
                field_name=nested_field,
                raw_path=nested_path,
            )
            if error:
                errors.append(error)
    return sorted(dict.fromkeys(errors))


def register_confirmed_task(
    task_root: Path,
    *,
    output_root: Path,
) -> TaskRegistrationResult:
    """Generate manifest and ready judge from a confirmed registration contract."""

    task_root = Path(task_root).resolve()
    output_root = _resolve_output_root(output_root)
    task_id = task_root.name
    paths = registration_contract_paths(task_root)
    contract = load_registration_contract(task_root, confirmed=True)
    normalized_input: dict[str, Any] = {}
    if paths["input_path"].exists():
        raw_registration_input, _ = load_registration_input(
            task_root,
            registration_input_path=paths["input_path"],
        )
        normalized_input = normalize_registration_input(raw_registration_input)
    validation_errors = validate_registration_contract(contract, require_complete=True)
    if validation_errors:
        raise ValueError(
            "registration_contract.json validation failed:\n- " + "\n- ".join(validation_errors)
        )
    task_path_errors = validate_registration_contract_task_paths(task_root, contract)
    if task_path_errors:
        raise ValueError(
            "registration_contract.json task-local path validation failed:\n- "
            + "\n- ".join(task_path_errors)
        )

    readme_info = _read_readme_info(task_root / "README.md")
    data_summary = _build_data_summary(task_root / "data")
    evaluation_summary = _build_evaluation_summary(task_root / "evaluation")
    output_detection = _build_output_detection(task_root, data_summary, readme_info, task_id)
    cache_record = ensure_task_runtime_env(
        task_root,
        output_root=output_root,
        task_id=task_id,
        family=str(contract.get("family", "") or ""),
    )
    runtime_env = runtime_env_manifest_payload(
        cache_record,
        output_root=output_root,
        task_root=task_root,
    )
    manifest = _build_manifest_from_contract(
        task_root=task_root,
        output_root=output_root,
        contract=contract,
        readme_info=readme_info,
        data_summary=data_summary,
        evaluation_summary=evaluation_summary,
        output_detection=output_detection,
        ready=False,
        validation_checks=[],
        runtime_env=runtime_env,
    )
    judge_source = render_ready_judge(contract)
    paths["judge_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["judge_path"].write_text(judge_source, encoding="utf-8")

    validation_checks = validate_generated_judge(
        task_root=task_root,
        judge_path=paths["judge_path"],
        manifest=manifest,
        python_executable=cache_record.python_executable,
    )
    manifest = _build_manifest_from_contract(
        task_root=task_root,
        output_root=output_root,
        contract=contract,
        readme_info=readme_info,
        data_summary=data_summary,
        evaluation_summary=evaluation_summary,
        output_detection=output_detection,
        ready=True,
        validation_checks=validation_checks,
        runtime_env=runtime_env,
    )
    registry_root = _registry_tasks_root(output_root)
    manifest_path = registry_root / f"{task_id}.json"
    registry_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    missing_items = _contract_missing_items(contract)
    notes_payload = _build_notes_payload(
        task_id=task_id,
        task_root=task_root,
        output_root=output_root,
        registration_input_path=paths.get("input_path"),
        registration_input=normalized_input,
        draft_path=paths["draft_path"],
        confirmed_path=paths["confirmed_path"],
        manifest_path=manifest_path,
        judge_path=paths["judge_path"],
        contract=contract,
        missing_items=missing_items,
        warnings=[],
        readme_info=readme_info,
        data_summary=data_summary,
        evaluation_summary=evaluation_summary,
        output_detection=output_detection,
        metric_source_candidates=_metric_source_paths_from_contract(contract),
        completion_source="confirmed_contract",
        judge_validation_checks=validation_checks,
        contract_status="confirmed_registered",
        judge_status="ready",
        runtime_env={
            "status": "ready",
            "env_hash": cache_record.env_hash,
            "python_executable": str(cache_record.python_executable),
            "install_report_path": str(cache_record.install_report_path),
            "build_log_path": str(cache_record.build_log_path),
        },
    )
    paths["notes_path"].write_text(
        json.dumps(notes_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_registry_notes_copy(task_id, output_root=output_root, payload=notes_payload)
    return TaskRegistrationResult(
        task_id=task_id,
        manifest_path=manifest_path,
        judge_path=paths["judge_path"],
        notes_path=paths["notes_path"],
        missing_items=missing_items,
        warnings=[],
    )


def validate_generated_judge(
    *,
    task_root: Path,
    judge_path: Path,
    manifest: Mapping[str, Any],
    python_executable: Path,
) -> list[str]:
    """Validate that a generated judge is importable and marked ready."""

    checks: list[str] = []
    source = Path(judge_path).read_text(encoding="utf-8")
    if GENERATED_JUDGE_SENTINEL not in source:
        raise ValueError("judge_adapter.py is missing the generated ready sentinel")
    if READY_JUDGE_MARKER not in source:
        raise ValueError("judge_adapter.py is not marked as ready")
    if "judge_not_implemented" in source:
        raise ValueError("judge_adapter.py still contains a stub marker")

    validation_result = invoke_judge_runner(
        python_executable,
        mode="validate",
        payload={"judge_path": str(Path(judge_path).resolve())},
    )
    if not bool(validation_result.get("callable_present", False)):
        raise AttributeError("generated judge is missing evaluate_run(task_root, run_record, manifest)")
    checks.append("judge_importable")
    checks.append("judge_callable_present")
    if not bool(validation_result.get("generated_ready", False)):
        raise ValueError("judge_adapter.py is not marked as ready")

    generated_metrics = list(validation_result.get("generated_metric_names", []) or [])
    manifest_metrics = list((manifest.get("judge_spec") or {}).get("metrics") or [])
    if generated_metrics != manifest_metrics:
        raise ValueError(
            "generated judge metric list does not match manifest judge_spec.metrics"
        )
    checks.append("judge_metrics_match_manifest")

    generated_contract_path = str(validation_result.get("generated_contract_path", "") or "")
    manifest_contract_path = str(
        (manifest.get("judge_spec") or {}).get("registration_contract_path", "") or ""
    )
    if generated_contract_path != manifest_contract_path:
        raise ValueError(
            "generated judge contract path does not match manifest judge_spec.registration_contract_path"
        )
    checks.append("judge_contract_path_match_manifest")
    checks.append("judge_ready_marker_present")
    return checks


def render_ready_judge(contract: Mapping[str, Any]) -> str:
    """Render a ready-to-import judge adapter from a confirmed contract."""

    metric_names = [
        str(item.get("name", "") or "")
        for item in (contract.get("judge_contract") or {}).get("metrics", [])
        if str(item.get("name", "") or "")
    ]
    contract_path = f"evaluation/{CONTRACT_FILENAME}"
    metric_names_literal = json.dumps(metric_names, ensure_ascii=False)
    contract_path_literal = json.dumps(contract_path, ensure_ascii=False)
    sentinel_literal = json.dumps(GENERATED_JUDGE_SENTINEL, ensure_ascii=False)
    return "\n".join(
        [
            '"""Task-local ready judge adapter generated from registration_contract.json."""',
            "",
            f"# {GENERATED_JUDGE_SENTINEL}",
            "from __future__ import annotations",
            "",
            "import importlib.util",
            "import json",
            "from pathlib import Path",
            "from typing import Any, Mapping",
            "",
            "import numpy as np",
            "",
            "from myevoskill.judging import HiddenJudge, MetricRequirement",
            "from myevoskill.models import JudgeResult, RunRecord",
            "from myevoskill.task_runtime import resolve_primary_output_path",
            "",
            "GENERATED_JUDGE_READY = True",
            f"GENERATED_METRIC_NAMES = {metric_names_literal}",
            f"GENERATED_REGISTRATION_CONTRACT_PATH = {contract_path_literal}",
            f"GENERATED_SENTINEL = {sentinel_literal}",
            "",
            "",
            "def _load_contract(task_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:",
            "    judge_spec = dict(manifest.get('judge_spec') or {})",
            "    contract_rel = str(",
            "        judge_spec.get('registration_contract_path') or GENERATED_REGISTRATION_CONTRACT_PATH",
            "    )",
            "    contract_path = task_root / contract_rel",
            "    return json.loads(contract_path.read_text(encoding='utf-8'))",
            "",
            "",
            "def _load_json(path: Path) -> Any:",
            "    return json.loads(path.read_text(encoding='utf-8'))",
            "",
            "",
            "def _load_resource_value(task_root: Path, spec: Mapping[str, Any], *, output_payload=None) -> Any:",
            "    if 'value' in spec:",
            "        return spec['value']",
            "    if output_payload is not None and spec.get('output_field'):",
            "        field = str(spec['output_field'])",
            "        if field not in output_payload.files:",
            "            raise KeyError(f'missing output field: {field}')",
            "        value = np.asarray(output_payload[field])",
            "    else:",
            "        resource_path = str(spec.get('resource_path') or spec.get('reference_resource_path') or '').strip()",
            "        field = str(spec.get('field') or spec.get('reference_field') or '').strip()",
            "        path = task_root / resource_path",
            "        suffix = path.suffix.lower()",
            "        if suffix == '.npz':",
            "            with np.load(path, allow_pickle=False) as payload:",
            "                if field not in payload.files:",
            "                    raise KeyError(f'missing field {field!r} in {resource_path}')",
            "                value = np.asarray(payload[field])",
            "        elif suffix == '.json':",
            "            payload = _load_json(path)",
            "            value = payload if not field else payload[field]",
            "        else:",
            "            raise RuntimeError(f'unsupported resource type for judge input: {resource_path}')",
            "    if isinstance(value, np.ndarray):",
            "        if 'index' in spec:",
            "            value = value[int(spec['index'])]",
            "        if 'scalar_index' in spec:",
            "            flat = np.asarray(value).reshape(-1)",
            "            value = float(flat[int(spec['scalar_index'])])",
            "        if bool(spec.get('squeeze', False)):",
            "            value = np.squeeze(value)",
            "    return value",
            "",
            "",
            "def _resolve_inputs(task_root: Path, input_spec: Mapping[str, Any], *, output_payload) -> dict[str, Any]:",
            "    resolved: dict[str, Any] = {}",
            "    for name, spec in dict(input_spec or {}).items():",
            "        if isinstance(spec, Mapping) and (",
            "            'output_field' in spec",
            "            or 'resource_path' in spec",
            "            or 'reference_resource_path' in spec",
            "            or 'value' in spec",
            "        ):",
            "            resolved[str(name)] = _load_resource_value(",
            "                task_root,",
            "                spec,",
            "                output_payload=output_payload,",
            "            )",
            "            continue",
            "        if isinstance(spec, Mapping):",
            "            nested: dict[str, Any] = {}",
            "            for nested_name, nested_spec in dict(spec).items():",
            "                nested[str(nested_name)] = _load_resource_value(",
            "                    task_root,",
            "                    nested_spec,",
            "                    output_payload=output_payload,",
            "                )",
            "            resolved[str(name)] = nested",
            "            continue",
            "        resolved[str(name)] = spec",
            "    return resolved",
            "",
            "",
            "def _load_helper_callable(task_root: Path, source_path: str, callable_name: str):",
            "    module_path = task_root / source_path",
            "    spec = importlib.util.spec_from_file_location(",
            "        f'myevoskill_metric_helper_{module_path.stem}',",
            "        module_path,",
            "    )",
            "    if spec is None or spec.loader is None:",
            "        raise RuntimeError(f'cannot import metric helper: {module_path}')",
            "    module = importlib.util.module_from_spec(spec)",
            "    spec.loader.exec_module(module)",
            "    helper = getattr(module, callable_name, None)",
            "    if not callable(helper):",
            "        raise AttributeError(f'metric helper not found: {source_path}:{callable_name}')",
            "    return helper",
            "",
            "",
            "def _operator_for_metric(metric: Mapping[str, Any]) -> str:",
            "    pass_condition = metric.get('pass_condition') or {}",
            "    operator = str(pass_condition.get('operator', '')).strip()",
            "    if operator not in {'<=', '>='}:",
            "        raise RuntimeError(f\"invalid pass_condition.operator for metric {metric.get('name', '')!r}: {operator!r}\")",
            "    return operator",
            "",
            "",
            "def _build_requirements(contract: Mapping[str, Any], task_root: Path) -> list[MetricRequirement]:",
            "    requirements: list[MetricRequirement] = []",
            "    for metric in contract.get('judge_contract', {}).get('metrics', []) or []:",
            "        pass_condition = metric.get('pass_condition') or {}",
            "        threshold_value = pass_condition.get('threshold')",
            "        if threshold_value is None:",
            "            raise RuntimeError(f\"missing pass_condition.threshold for metric {metric.get('name', '')!r}\")",
            "        requirements.append(",
            "            MetricRequirement(",
            "                str(metric['name']),",
            "                float(threshold_value),",
            "                _operator_for_metric(metric),",
            "            )",
            "        )",
            "    return requirements",
            "",
            "",
            "def _cosine_similarity(pred: np.ndarray, ref: np.ndarray) -> float:",
            "    pred_flat = np.asarray(pred, dtype=float).ravel()",
            "    ref_flat = np.asarray(ref, dtype=float).ravel()",
            "    denom = float(np.linalg.norm(pred_flat) * np.linalg.norm(ref_flat) + 1e-12)",
            "    return float(np.dot(pred_flat, ref_flat) / denom)",
            "",
            "",
            "def _nrmse(pred: np.ndarray, ref: np.ndarray) -> float:",
            "    pred_arr = np.asarray(pred, dtype=float)",
            "    ref_arr = np.asarray(ref, dtype=float)",
            "    dynamic_range = float(ref_arr.max() - ref_arr.min() + 1e-12)",
            "    return float(np.sqrt(np.mean((pred_arr - ref_arr) ** 2)) / dynamic_range)",
            "",
            "",
            "def _validate_output_contract_shapes(contract: Mapping[str, Any], output_payload) -> list[str]:",
            "    failures: list[str] = []",
            "    output_contract = dict(contract.get('output_contract') or {})",
            "    same_shape_fields = [str(item) for item in output_contract.get('same_shape_fields', []) or [] if str(item)]",
            "    if same_shape_fields:",
            "        shapes: dict[str, tuple[int, ...]] = {}",
            "        for field in same_shape_fields:",
            "            if field not in output_payload.files:",
            "                continue",
            "            shapes[field] = tuple(np.asarray(output_payload[field]).shape)",
            "        unique_shapes = sorted(set(shapes.values()))",
            "        if len(unique_shapes) > 1:",
            "            failures.append('invalid_output_shape')",
            "    return failures",
            "",
            "",
            "def _compute_standard_metric(metric: Mapping[str, Any], output_payload, task_root: Path) -> float:",
            "    pred = _load_resource_value(",
            "        task_root,",
            "        {'output_field': metric['output_field']},",
            "        output_payload=output_payload,",
            "    )",
            "    ref = _load_resource_value(",
            "        task_root,",
            "        {",
            "            'resource_path': metric['reference_resource_path'],",
            "            'field': metric['reference_field'],",
            "        },",
            "        output_payload=output_payload,",
            "    )",
            "    mode = str(metric.get('mode', '') or '').lower()",
            "    if mode == 'phase_ncc':",
            "        pred_phase = np.angle(np.asarray(pred))",
            "        ref_phase = np.angle(np.asarray(ref))",
            "        pred_phase = pred_phase - pred_phase.mean()",
            "        ref_phase = ref_phase - ref_phase.mean()",
            "        return _cosine_similarity(pred_phase, ref_phase)",
            "    if mode == 'phase_nrmse':",
            "        pred_phase = np.angle(np.asarray(pred))",
            "        ref_phase = np.angle(np.asarray(ref))",
            "        pred_phase = pred_phase - pred_phase.mean()",
            "        ref_phase = ref_phase - ref_phase.mean()",
            "        return _nrmse(pred_phase, ref_phase)",
            "    if mode == 'ncc':",
            "        return _cosine_similarity(pred, ref)",
            "    if mode == 'nrmse':",
            "        return _nrmse(pred, ref)",
            "    if mode == 'abs_error':",
            "        pred_scalar = float(np.asarray(pred).reshape(-1)[0])",
            "        ref_scalar = float(np.asarray(ref).reshape(-1)[0])",
            "        return float(abs(pred_scalar - ref_scalar))",
            "    raise RuntimeError(f'unsupported standard metric mode: {metric.get(\"mode\", \"\")!r}')",
            "",
            "",
            "def _compute_script_metric(metric: Mapping[str, Any], output_payload, task_root: Path) -> float:",
            "    helper = _load_helper_callable(",
            "        task_root,",
            "        str(metric['source_path']),",
            "        str(metric['callable']),",
            "    )",
            "    inputs = _resolve_inputs(task_root, metric.get('inputs') or {}, output_payload=output_payload)",
            "    result = helper(**inputs)",
            "    if not isinstance(result, Mapping):",
            "        raise RuntimeError(",
            "            f\"script metric helper must return a mapping: {metric['source_path']}:{metric['callable']}\"",
            "        )",
            "    result_key = str(metric['result_key'])",
            "    if result_key not in result:",
            "        raise KeyError(",
            "            f\"script metric result key not found: {metric['source_path']}:{metric['callable']} -> {result_key}\"",
            "        )",
            "    return float(result[result_key])",
            "",
            "",
            "def evaluate_run(",
            "    task_root: Path,",
            "    run_record: RunRecord,",
            "    manifest: Mapping[str, Any],",
            ") -> JudgeResult:",
            "    task_root = Path(task_root)",
            "    contract = _load_contract(task_root, manifest)",
            "    requirements = _build_requirements(contract, task_root)",
            "    output_path = resolve_primary_output_path(run_record.workspace_root, manifest)",
            "    if not output_path.exists():",
            "        return HiddenJudge().evaluate(",
            "            task_id=run_record.task_id,",
            "            metrics_actual={},",
            "            requirements=requirements,",
            "            failure_tags=['missing_output'],",
            "        )",
            "    metrics_actual: dict[str, float] = {}",
            "    try:",
            "        with np.load(output_path, allow_pickle=False) as output_payload:",
            "            required_fields = list(contract.get('output_contract', {}).get('required_fields', []))",
            "            missing_fields = [field for field in required_fields if field not in output_payload.files]",
            "            if missing_fields:",
            "                return HiddenJudge().evaluate(",
            "                    task_id=run_record.task_id,",
            "                    metrics_actual={},",
            "                    requirements=requirements,",
            "                    failure_tags=['missing_required_field'],",
            "                )",
            "            shape_failures = _validate_output_contract_shapes(contract, output_payload)",
            "            if shape_failures:",
            "                return HiddenJudge().evaluate(",
            "                    task_id=run_record.task_id,",
            "                    metrics_actual={},",
            "                    requirements=requirements,",
            "                    failure_tags=shape_failures,",
            "                )",
            "            for metric in contract.get('judge_contract', {}).get('metrics', []) or []:",
            "                kind = str(metric.get('kind', '') or '').lower()",
            "                if kind == 'script':",
            "                    metrics_actual[str(metric['name'])] = _compute_script_metric(",
            "                        metric,",
            "                        output_payload,",
            "                        task_root,",
            "                    )",
            "                    continue",
            "                metrics_actual[str(metric['name'])] = _compute_standard_metric(",
            "                    metric,",
            "                    output_payload,",
            "                    task_root,",
            "                )",
            "    except Exception as exc:",
            "        return HiddenJudge().evaluate(",
            "            task_id=run_record.task_id,",
            "            metrics_actual=metrics_actual,",
            "            requirements=requirements,",
            "            failure_tags=[f'judge_runtime_error:{type(exc).__name__}'],",
            "        )",
            "    return HiddenJudge().evaluate(",
            "        task_id=run_record.task_id,",
            "        metrics_actual=metrics_actual,",
            "        requirements=requirements,",
            "        failure_tags=[],",
            "    )",
            "",
        ]
    )


def ensure_live_ready_manifest(
    manifest: Mapping[str, Any],
    *,
    task_root: Path,
) -> dict[str, Any]:
    """Reject live execution when the manifest or confirmed contract is not ready."""

    judge_spec = dict(manifest.get("judge_spec") or {})
    if not bool(judge_spec.get("ready", False)):
        raise RuntimeError(
            "live run refused: manifest judge_spec.ready is false; run task registration first"
        )
    runtime_env = dict(manifest.get("runtime_env") or {})
    if not runtime_env:
        raise RuntimeError(
            "live run refused: manifest runtime_env is missing; rerun task registration"
        )
    if not bool(runtime_env.get("ready", False)):
        raise RuntimeError(
            "live run refused: manifest runtime_env.ready is false; rerun task registration"
        )
    contract_rel = str(
        judge_spec.get("registration_contract_path")
        or manifest.get("registration_contract_path")
        or ""
    ).strip()
    if not contract_rel:
        raise RuntimeError(
            "live run refused: manifest judge_spec.registration_contract_path is missing"
        )
    contract_path = Path(task_root).resolve() / contract_rel
    if not contract_path.exists():
        raise FileNotFoundError(f"live run refused: confirmed contract not found: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validation_errors = validate_registration_contract(contract, require_complete=True)
    if validation_errors:
        raise RuntimeError(
            "live run refused: registration_contract.json is invalid:\n- "
            + "\n- ".join(validation_errors)
        )
    task_path_errors = validate_registration_contract_task_paths(task_root, contract)
    if task_path_errors:
        raise RuntimeError(
            "live run refused: registration_contract.json has invalid task-local paths:\n- "
            + "\n- ".join(task_path_errors)
        )
    return contract


def load_task_registration_notes(
    task_id: str,
    *,
    output_root: Optional[Path] = None,
    task_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Load registration notes from the registry copy or task-local notes file."""

    if task_root is not None:
        path = registration_contract_paths(task_root)["notes_path"]
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    project_root = _resolve_output_root(output_root)
    registry_path = _registry_tasks_root(project_root) / f"{task_id}.notes.json"
    if registry_path.exists():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"registration notes not found for task_id={task_id!r}")


def _build_contract_resources(
    *,
    task_root: Path,
    data_summary: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
    metric_source_candidates: Sequence[str],
) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []

    def _append(
        relative_path: str,
        *,
        role: str,
        visibility: str,
        semantics: str,
        authority: str,
    ) -> None:
        normalized = relative_path.replace("\\", "/").strip()
        if not normalized:
            return
        resources.append(
            {
                "path": normalized,
                "role": role,
                "visibility": visibility,
                "semantics": semantics,
                "authority": authority,
            }
        )

    if (task_root / "README.md").exists():
        _append(
            "README.md",
            role="task_description",
            visibility="public",
            semantics="Task description and public constraints for the solver.",
            authority="authoritative",
        )
    if (task_root / "data" / "raw_data.npz").exists():
        _append(
            "data/raw_data.npz",
            role="public_input_data",
            visibility="public",
            semantics="Public observation data available to the execution agent.",
            authority="authoritative",
        )
    if (task_root / "data" / "meta_data.json").exists():
        _append(
            "data/meta_data.json",
            role="public_metadata",
            visibility="public",
            semantics="Public physical parameters, constants, and experiment configuration.",
            authority="authoritative",
        )
    if (task_root / "evaluation" / "self_eval.py").exists():
        _append(
            "evaluation/self_eval.py",
            role="public_eval_script",
            visibility="public",
            semantics="Public self-check script or evaluator available to the solver.",
            authority="supplementary",
        )
    if (task_root / "evaluation" / "metrics.json").exists():
        _append(
            "evaluation/metrics.json",
            role="hidden_metric_config",
            visibility="private",
            semantics="Metric threshold configuration used by the hidden judge.",
            authority="authoritative",
        )
    if (task_root / "data" / "ground_truth.npz").exists():
        _append(
            "data/ground_truth.npz",
            role="hidden_reference",
            visibility="private",
            semantics="Hidden ground-truth reference used for metric evaluation.",
            authority="authoritative",
        )
    if evaluation_summary.get("reference_outputs_exists", False):
        for item in evaluation_summary.get("reference_outputs_files", []) or []:
            _append(
                f"evaluation/{item}",
                role="hidden_reference",
                visibility="private",
                semantics="Hidden reference output retained for offline evaluation or parity checks.",
                authority="supplementary",
            )
    for relative_path in metric_source_candidates:
        if not (task_root / relative_path).exists():
            continue
        _append(
            relative_path,
            role="metric_helper",
            visibility="private",
            semantics="Task-private Python source that defines or computes evaluation metrics.",
            authority="supplementary",
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for resource in resources:
        key = (str(resource["path"]), str(resource["role"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resource)
    return deduped


def _build_contract_output_contract(output_detection: Mapping[str, Any]) -> dict[str, Any]:
    required_fields = [
        str(item)
        for item in output_detection.get("candidate_output_fields", []) or []
        if str(item)
    ]
    return {
        "path": str(output_detection.get("output_path") or DEFAULT_OUTPUT_PATH),
        "format": str(output_detection.get("output_format") or "npz"),
        "required_fields": required_fields or ["TODO_field_name"],
        "numeric_fields": list(required_fields or ["TODO_field_name"]),
        "same_shape_fields": _infer_same_shape_fields(required_fields or ["TODO_field_name"]),
    }


def _build_draft_judge_contract(
    *,
    task_root: Path,
    readme_info: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
    output_detection: Mapping[str, Any],
    metric_source_candidates: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    metrics_summary = dict(evaluation_summary.get("metrics_summary") or {})
    ground_truth_arrays = dict((data_summary.get("ground_truth") or {}).get("arrays") or {})
    output_fields = list(output_detection.get("candidate_output_fields") or [])
    metric_helper = _pick_metric_helper(task_root, metric_source_candidates)

    if (
        metric_helper
        and {"reconstructed_spectrum", "estimated_temperature_K"}.issubset(output_fields)
        and {"spectrum", "temperature"}.issubset(ground_truth_arrays)
    ):
        return (
            {
                "metrics": [
                    {
                        "name": "ncc_vs_ref",
                        "kind": "script",
                        "description": "Cosine similarity between reconstructed spectrum and hidden reference spectrum.",
                        "source_path": metric_helper,
                        "callable": "compute_metrics",
                        "result_key": "ncc",
                        "pass_condition": {
                            "operator": ">=",
                            "threshold": float(metrics_summary.get("ncc_boundary", 0.0)),
                        },
                        "inputs": {
                            "y_true": {
                                "resource_path": "data/raw_data.npz",
                                "field": "measurements",
                                "index": 0,
                            },
                            "y_pred": {
                                "output_field": "reconstructed_spectrum",
                                "index": 0,
                            },
                            "params_pred": {
                                "temperature": {
                                    "output_field": "estimated_temperature_K",
                                    "scalar_index": 0,
                                }
                            },
                            "params_true": {
                                "temperature": {
                                    "resource_path": "data/ground_truth.npz",
                                    "field": "temperature",
                                    "scalar_index": 0,
                                }
                            },
                        },
                    },
                    {
                        "name": "nrmse_vs_ref",
                        "kind": "script",
                        "description": "NRMSE between reconstructed spectrum and hidden reference spectrum.",
                        "source_path": metric_helper,
                        "callable": "compute_metrics",
                        "result_key": "nrmse",
                        "pass_condition": {
                            "operator": "<=",
                            "threshold": float(metrics_summary.get("nrmse_boundary", 0.0)),
                        },
                        "inputs": {
                            "y_true": {
                                "resource_path": "data/raw_data.npz",
                                "field": "measurements",
                                "index": 0,
                            },
                            "y_pred": {
                                "output_field": "reconstructed_spectrum",
                                "index": 0,
                            },
                            "params_pred": {
                                "temperature": {
                                    "output_field": "estimated_temperature_K",
                                    "scalar_index": 0,
                                }
                            },
                            "params_true": {
                                "temperature": {
                                    "resource_path": "data/ground_truth.npz",
                                    "field": "temperature",
                                    "scalar_index": 0,
                                }
                            },
                        },
                    },
                    {
                        "name": "temperature_error_K",
                        "kind": "script",
                        "description": "Absolute temperature error against the hidden ground truth.",
                        "source_path": metric_helper,
                        "callable": "compute_metrics",
                        "result_key": "temperature_error_K",
                        "pass_condition": {
                            "operator": "<=",
                            "threshold": float(
                                metrics_summary.get("temperature_error_K_boundary", 0.0)
                            ),
                        },
                        "inputs": {
                            "y_true": {
                                "resource_path": "data/raw_data.npz",
                                "field": "measurements",
                                "index": 0,
                            },
                            "y_pred": {
                                "output_field": "reconstructed_spectrum",
                                "index": 0,
                            },
                            "params_pred": {
                                "temperature": {
                                    "output_field": "estimated_temperature_K",
                                    "scalar_index": 0,
                                }
                            },
                            "params_true": {
                                "temperature": {
                                    "resource_path": "data/ground_truth.npz",
                                    "field": "temperature",
                                    "scalar_index": 0,
                                }
                            },
                        },
                    },
                ]
            },
            warnings,
        )

    domain_text = " ".join(
        str(item or "")
        for item in (
            task_root.name,
            readme_info.get("title", ""),
            readme_info.get("domain", ""),
        )
    ).lower()
    if "ptychograph" in domain_text and "object" in ground_truth_arrays and "object" in output_fields:
        return (
            {
                "metrics": [
                    {
                        "name": "phase_ncc",
                        "kind": "standard",
                        "description": str(
                            metrics_summary.get(
                                "ncc_definition",
                                "Cosine similarity on mean-centered phase maps.",
                            )
                        ),
                        "mode": "phase_ncc",
                        "output_field": "object",
                        "reference_resource_path": "data/ground_truth.npz",
                        "reference_field": "object",
                        "pass_condition": {
                            "operator": ">=",
                            "threshold": float(metrics_summary.get("ncc_boundary", 0.0)),
                        },
                        "source_hint": _prefer_existing_metric_source(
                            metric_source_candidates,
                            preferred=("main.py", "src/visualization.py"),
                        ),
                    },
                    {
                        "name": "phase_nrmse",
                        "kind": "standard",
                        "description": str(
                            metrics_summary.get(
                                "nrmse_definition",
                                "NRMSE on mean-centered phase maps.",
                            )
                        ),
                        "mode": "phase_nrmse",
                        "output_field": "object",
                        "reference_resource_path": "data/ground_truth.npz",
                        "reference_field": "object",
                        "pass_condition": {
                            "operator": "<=",
                            "threshold": float(metrics_summary.get("nrmse_boundary", 0.0)),
                        },
                        "source_hint": _prefer_existing_metric_source(
                            metric_source_candidates,
                            preferred=("main.py", "src/visualization.py"),
                        ),
                    },
                ]
            },
            warnings,
        )

    warnings.append(
        "draft judge contract could not fully infer metric mappings; confirm judge_contract.metrics manually before register"
    )
    boundary_keys = sorted(
        key for key in metrics_summary.keys() if str(key).endswith("_boundary")
    )
    metrics: list[dict[str, Any]] = []
    for key in boundary_keys or ["TODO_metric_boundary"]:
        name = str(key)[: -len("_boundary")] if str(key).endswith("_boundary") else "TODO_metric_name"
        threshold_value = metrics_summary.get(key)
        operator = "<=" if any(token in name.lower() for token in ("error", "rmse", "loss")) else ">="
        metrics.append(
            {
                "name": name,
                "kind": "standard",
                "description": str(metrics_summary.get(f"{name}_definition", f"TODO description for {name}")),
                "mode": "TODO_mode",
                "output_field": "TODO_output_field",
                "reference_resource_path": "TODO_reference_resource_path",
                "reference_field": "TODO_reference_field",
                "pass_condition": {
                    "operator": operator,
                    "threshold": float(threshold_value if threshold_value is not None else 0.0),
                },
                "source_hint": _prefer_existing_metric_source(metric_source_candidates, preferred=("main.py",)),
            }
        )
    return {"metrics": metrics}, warnings


def _build_execution_conventions(task_root: Path) -> dict[str, Any]:
    readable_paths = ["README_public.md"]
    for relative_path in ("data/raw_data.npz", "data/meta_data.json", "requirements.txt"):
        if relative_path == "README_public.md":
            continue
        raw_path = task_root / relative_path.replace("README_public.md", "README.md")
        if raw_path.exists():
            readable_paths.append(relative_path)
    return {
        "read_first": ["README_public.md"],
        "readable_paths": readable_paths,
        "writable_paths": ["work", "output", "checkpoints"],
        "entrypoint": "work/main.py",
    }


def _build_manifest_from_contract(
    *,
    task_root: Path,
    output_root: Path,
    contract: Mapping[str, Any],
    readme_info: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
    output_detection: Mapping[str, Any],
    ready: bool,
    validation_checks: Sequence[str],
    runtime_env: Mapping[str, Any],
) -> dict[str, Any]:
    public_policy = _build_public_policy(task_root, readme_info, data_summary, evaluation_summary)
    output_contract = dict(contract.get("output_contract") or {})
    public_eval_spec = _build_public_eval_spec(data_summary, output_detection)
    metrics = list((contract.get("judge_contract") or {}).get("metrics") or [])
    required_fields = list(output_contract.get("required_fields") or [])
    metric_config_path = _metric_config_path_from_contract(contract)
    metric_sources = _metric_sources_from_contract(contract)
    output_path = str(output_contract.get("path") or DEFAULT_OUTPUT_PATH)
    metric_names = [str(metric.get("name")) for metric in metrics]
    runtime_layout = coerce_runtime_layout(contract.get("runtime_layout"))
    runtime_policy = coerce_runtime_policy(contract.get("runtime_policy"))
    return {
        "task_id": str(contract["task_id"]),
        "family": str(contract["family"]),
        "source_task_dir": _relative_source_task_dir(task_root, output_root),
        "public_policy": public_policy,
        "runtime_layout": runtime_layout,
        "runtime_policy": runtime_policy,
        "runtime_env": dict(runtime_env or {}),
        "output_contract": {
            "required_outputs": [
                {
                    "path": str(output_contract.get("path") or DEFAULT_OUTPUT_PATH),
                    "format": str(output_contract.get("format") or "npz"),
                    "required_fields": required_fields,
                    "numeric_fields": list(output_contract.get("numeric_fields") or required_fields),
                    "same_shape_fields": list(
                        output_contract.get("same_shape_fields")
                        or _infer_same_shape_fields(required_fields)
                    ),
                }
            ]
        },
        "public_eval_spec": public_eval_spec,
        "proxy_spec": {
            "primary_output": output_path,
            "output_dtype": str(output_contract.get("format") or "npz"),
            "required_fields": required_fields,
            "numeric_fields": list(output_contract.get("numeric_fields") or required_fields),
            "same_shape_fields": list(
                output_contract.get("same_shape_fields") or _infer_same_shape_fields(required_fields)
            ),
        },
        "proxy_output_name": Path(output_path).name,
        "judge_metrics": metric_names,
        "execution_conventions": dict(contract.get("execution_conventions") or {}),
        "registration_contract_path": f"evaluation/{CONTRACT_FILENAME}",
        "registration_contract": dict(contract),
        "judge_spec": {
            "adapter_path": DEFAULT_JUDGE_ADAPTER_PATH,
            "callable": "evaluate_run",
            "metrics_config_path": metric_config_path,
            "metrics": metric_names,
            "required_fields": required_fields,
            "ready": bool(ready),
            "metric_sources": metric_sources,
            "registration_contract_path": f"evaluation/{CONTRACT_FILENAME}",
            "validation_checks": list(validation_checks),
        },
    }


def _warning_messages(warnings: Sequence[Any]) -> list[str]:
    messages: list[str] = []
    for item in warnings or []:
        if isinstance(item, Mapping):
            message = str(item.get("message", "") or "").strip()
            field = str(item.get("field", "") or "").strip()
            if field and message:
                messages.append(f"{field}: {message}")
                continue
            if message:
                messages.append(message)
                continue
        text = str(item or "").strip()
        if text:
            messages.append(text)
    return list(dict.fromkeys(messages))


def _build_notes_payload(
    *,
    task_id: str,
    task_root: Path,
    output_root: Path,
    registration_input_path: Optional[Path] = None,
    registration_input: Optional[Mapping[str, Any]] = None,
    draft_path: Path,
    confirmed_path: Path,
    manifest_path: Optional[Path],
    judge_path: Optional[Path],
    contract: Mapping[str, Any],
    missing_items: Sequence[str],
    warnings: Sequence[Any],
    readme_info: Optional[Mapping[str, Any]] = None,
    data_summary: Optional[Mapping[str, Any]] = None,
    evaluation_summary: Optional[Mapping[str, Any]] = None,
    output_detection: Optional[Mapping[str, Any]] = None,
    metric_source_candidates: Sequence[str] = (),
    completion_source: str = "",
    vendor_session_ref: Optional[Mapping[str, Any]] = None,
    contract_generation_notes: Optional[Mapping[str, Any]] = None,
    judge_recommendations: Optional[Mapping[str, Any]] = None,
    judge_validation_checks: Sequence[str],
    contract_status: str,
    judge_status: str,
    runtime_env: Optional[Mapping[str, Any]] = None,
    attempt_summaries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    normalized_notes = (
        dict(contract_generation_notes) if isinstance(contract_generation_notes, Mapping) else {}
    )
    normalized_warnings = [
        dict(item) if isinstance(item, Mapping) else {"message": str(item or "").strip()}
        for item in warnings or []
        if (
            isinstance(item, Mapping)
            and str(item.get("message", "") or "").strip()
        )
        or str(item or "").strip()
    ]
    return {
        "task_id": task_id,
        "task_root": str(task_root),
        "registration_input_path": (
            str(registration_input_path) if registration_input_path else ""
        ),
        "registration_input": dict(registration_input or {}),
        "draft_path": str(draft_path),
        "confirmed_path": str(confirmed_path),
        "manifest_path": str(manifest_path) if manifest_path else "",
        "judge_path": str(judge_path) if judge_path else "",
        "missing_items": list(missing_items),
        "warnings": normalized_warnings,
        "attempt_count": len(list(attempt_summaries)),
        "attempt_summaries": [dict(item) for item in attempt_summaries or [] if isinstance(item, Mapping)],
        "contract": dict(contract),
        "metric_source_candidates": list(metric_source_candidates),
        "readme_summary": dict(readme_info or {}),
        "data_summary": dict(data_summary or {}),
        "evaluation_summary": dict(evaluation_summary or {}),
        "output_detection": dict(output_detection or {}),
        "declared_inputs_used": [
            str(item).replace("\\", "/").strip()
            for item in normalized_notes.get("declared_inputs_used", []) or []
            if str(item or "").strip()
        ],
        "resource_validation": [
            dict(item)
            for item in normalized_notes.get("resource_validation", []) or []
            if isinstance(item, Mapping)
        ],
        "judge_recommendations": dict(judge_recommendations or {}),
        "completion_source": str(completion_source or ""),
        "vendor_session_ref": dict(vendor_session_ref or {}),
        "contract_generation": {
            "status": contract_status,
            "source_files": sorted(
                path.relative_to(task_root).as_posix()
                for path in task_root.rglob("*")
                if path.is_file()
            ),
            "agent_summary": str(normalized_notes.get("agent_summary", "") or "").strip(),
            "open_questions": [
                str(item).strip()
                for item in normalized_notes.get("open_questions", []) or []
                if str(item or "").strip()
            ],
            "completion_source": str(completion_source or ""),
            "vendor_session_ref": dict(vendor_session_ref or {}),
        },
        "judge_generation": {
            "status": judge_status,
            "validation_checks": list(judge_validation_checks),
        },
        "runtime_env": dict(runtime_env or {}),
        "registry_notes_path": str(_registry_tasks_root(output_root) / f"{task_id}.notes.json"),
    }


def _write_registry_notes_copy(
    task_id: str,
    *,
    output_root: Path,
    payload: Mapping[str, Any],
) -> None:
    registry_root = _registry_tasks_root(output_root)
    registry_root.mkdir(parents=True, exist_ok=True)
    (registry_root / f"{task_id}.notes.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _metric_config_path_from_contract(contract: Mapping[str, Any]) -> str:
    for resource in contract.get("resources", []) or []:
        if str(resource.get("role", "") or "") == "hidden_metric_config":
            return str(resource.get("path", "") or DEFAULT_METRICS_CONFIG_PATH)
    return DEFAULT_METRICS_CONFIG_PATH


def _metric_sources_from_contract(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for metric in (contract.get("judge_contract") or {}).get("metrics", []) or []:
        metric_name = str(metric.get("name", "") or "")
        if not metric_name:
            continue
        if str(metric.get("kind", "") or "") == "script":
            sources.append(
                {
                    "metric": metric_name,
                    "kind": "script",
                    "source_path": str(metric.get("source_path", "") or ""),
                    "callable": str(metric.get("callable", "") or ""),
                }
            )
            continue
        sources.append(
            {
                "metric": metric_name,
                "kind": "standard",
                "source_path": str(metric.get("source_hint", "") or ""),
                "callable": "",
            }
        )
    return sources


def _metric_source_paths_from_contract(contract: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for metric in (contract.get("judge_contract") or {}).get("metrics", []) or []:
        for field_name in ("source_path", "source_hint"):
            value = str(metric.get(field_name, "") or "").strip()
            if value:
                candidates.append(value)
    return list(dict.fromkeys(candidates))


def _pick_metric_helper(task_root: Path, metric_source_candidates: Sequence[str]) -> str:
    for candidate in metric_source_candidates:
        path = task_root / candidate
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "def compute_metrics" in text:
            return candidate
    return ""


def _prefer_existing_metric_source(
    metric_source_candidates: Sequence[str],
    *,
    preferred: Sequence[str],
) -> str:
    normalized = [str(item).replace("\\", "/") for item in metric_source_candidates]
    for candidate in preferred:
        if candidate in normalized:
            return candidate
    return normalized[0] if normalized else ""


def _collect_metric_source_candidates(
    task_root: Path,
    *,
    explicit_metric_sources: Sequence[str],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    candidates: list[str] = []
    for raw_candidate in explicit_metric_sources:
        candidate = str(raw_candidate).replace("\\", "/").strip()
        if not candidate:
            continue
        if not (task_root / candidate).exists():
            warnings.append(f"metric source path not found: {candidate}")
            continue
        candidates.append(candidate)

    auto_candidates: list[str] = []
    for relative_path in ["main.py", *sorted(path.relative_to(task_root).as_posix() for path in (task_root / "src").rglob("*.py") if path.is_file())] if (task_root / "src").exists() else ["main.py"]:
        path = task_root / relative_path
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(
            token in text
            for token in (
                "compute_metrics",
                "phase_ncc",
                "phase_nrmse",
                "ncc_boundary",
                "nrmse_boundary",
                "temperature_error",
                "metrics =",
            )
        ):
            auto_candidates.append(relative_path)
    candidates.extend(auto_candidates)
    return list(dict.fromkeys(candidates)), warnings


def _contains_placeholder_token(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return not stripped or "TODO" in stripped
    if isinstance(value, Mapping):
        if not value:
            return True
        return any(_contains_placeholder_token(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return True
        return any(_contains_placeholder_token(item) for item in value)
    return False


def _contract_missing_items(contract: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if _contains_placeholder_token(contract.get("family")):
        missing.append("family")
    if _contains_placeholder_token(contract.get("output_contract")):
        missing.append("output_contract")
    if _contains_placeholder_token(contract.get("judge_contract")):
        missing.append("judge_contract")
    if _contains_placeholder_token(contract.get("execution_conventions")):
        missing.append("execution_conventions")
    return sorted(dict.fromkeys(missing))
