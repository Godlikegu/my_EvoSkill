"""Single CLI entry point for MyEvoSkill.

Subcommands
-----------

* ``register-task``    Build / refresh a registry manifest for one task.
* ``run-task``         Run one registered task end-to-end (one process,
                       one Claude session, multi-round with judge feedback).
* ``run-batch``        Run several tasks concurrently in isolated subprocesses,
                       deleting per-run claude history afterwards.
* ``setup-task-env``   Build the per-task venv consumed by registration.
* ``export-trajectory`` Write a distillation-clean trajectory JSONL.
* ``validate-skill``   Valid-split gate: run +skill on selected valid tasks
                       by default; optionally run baseline vs +skill with
                       ``--compare-baseline``.

Everything else (compilation, visualisation, legacy bootstrap, ...) lives in
its own module under ``myevoskill/`` and is invoked directly via
``python -m myevoskill.<module>`` for advanced users.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


from .concurrency import run_tasks_parallel
from .artifact_paths import model_slug
from .harness import HarnessConfig, run_task_once
from .model_provider import (
    ModelProviderError,
    default_llm_config_path,
    load_model_provider_registry,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- helpers


def _load_manifest(repo_root: Path, task_id: str) -> dict[str, Any]:
    path = repo_root / "registry" / "tasks" / f"{task_id}.json"
    if not path.exists():
        raise SystemExit(
            f"manifest not found: {path}\n"
            f"Run `python -m myevoskill.cli register-task --task-id {task_id}` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_default_model(cli_model: str | None) -> str | None:
    """Resolve the model name with precedence:

    1. ``--model`` flag (explicit override).
    2. ``MYEVOSKILL_MODEL`` environment variable.
    3. ``model`` field in ``~/.claude/settings.json`` (so the harness uses
       the same model the user already configured for the Claude CLI,
       e.g. ``Vendor2/Claude-4.6-opus`` on a 3rd-party gateway).
    4. None - let the SDK pick its own default.
    """

    import os as _os

    if cli_model:
        return cli_model
    env_model = _os.environ.get("MYEVOSKILL_MODEL")
    if env_model:
        return env_model
    home = Path(_os.environ.get("USERPROFILE") or _os.environ.get("HOME") or "")
    settings_path = home / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            m = data.get("model")
            if isinstance(m, str) and m.strip():
                return m.strip()
        except (OSError, json.JSONDecodeError):
            pass
    return None


def _resolve_model_provider_for_cli(
    *,
    repo_root: Path,
    model_id: str | None,
    llm_config: str | None,
) -> tuple[str | None, dict[str, str], dict[str, Any], str] | None:
    if not model_id:
        return None

    config_path = (
        Path(llm_config).resolve()
        if llm_config
        else default_llm_config_path(repo_root).resolve()
    )
    registry = load_model_provider_registry(config_path)
    runtime = registry.resolve_claude_gateway_runtime(model_id)
    summary = runtime.safe_log_config()
    if summary.get("api_key_source") == "inline":
        logger.warning(
            "model_id %s uses an inline api_key from %s; keep this file out of git",
            model_id,
            config_path,
        )
    logger.info(
        "using model provider: %s",
        json.dumps(summary, ensure_ascii=False, sort_keys=True),
    )
    return (
        runtime.model_config.model_name,
        dict(runtime.env),
        summary,
        model_slug(str(summary.get("model_id") or runtime.model_config.model_name)),
    )


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )


# --------------------------------------------------------------------- commands


def cmd_register_task(args: argparse.Namespace) -> int:
    """Run the deterministic v2 registration step.

    Reads ``tasks/<task_id>/evaluation/task_contract.json`` and writes
    ``registry/tasks/<task_id>.json``. No LLM calls, no agent involvement.
    """

    from .registration import RegistrationError, register_task

    repo_root = Path(args.repo_root).resolve()
    try:
        result = register_task(
            repo_root=repo_root,
            task_id=args.task_id,
            tasks_root=Path(args.tasks_root) if args.tasks_root else None,
            force=bool(args.force),
            require_task_env=bool(getattr(args, "require_task_env", False)),
        )
    except RegistrationError as exc:
        print(f"registration failed: {exc}", file=sys.stderr)
        return 2

    print(f"registered: {result.manifest_path}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0


def cmd_setup_task_env(args: argparse.Namespace) -> int:
    from .task_env import TaskEnvSetupError, setup_task_env

    try:
        result = setup_task_env(
            repo_root=Path(args.repo_root),
            task_id=args.task_id,
            tasks_root=Path(args.tasks_root) if args.tasks_root else None,
            force=bool(args.force),
            base_python=Path(args.python) if args.python else None,
            shared_torch_env=Path(args.shared_torch_env) if args.shared_torch_env else None,
            torch_cuda_index_url=args.torch_cuda_index_url,
            torch_version=args.torch_version,
            require_gpu_torch=bool(args.require_gpu_torch),
            skip_notebook_packages=not bool(args.install_notebook_packages),
        )
    except TaskEnvSetupError as exc:
        print(f"setup-task-env failed: {exc}", file=sys.stderr)
        return 2

    print(f"setup-task-env: {result.task_id}")
    print(f"  state:  {result.state_path}")
    print(f"  python: {result.python_executable}")
    if result.shared_torch_env:
        print(f"  shared torch: {result.shared_torch_env}")
    return 0


def cmd_setup_shared_torch_env(args: argparse.Namespace) -> int:
    from .task_env import TaskEnvSetupError, setup_shared_torch_env

    try:
        info = setup_shared_torch_env(
            repo_root=Path(args.repo_root),
            force=bool(args.force),
            base_python=Path(args.python) if args.python else None,
            shared_torch_env=Path(args.shared_torch_env) if args.shared_torch_env else None,
            torch_cuda_index_url=args.torch_cuda_index_url,
            torch_version=args.torch_version,
            require_gpu_torch=bool(args.require_gpu_torch),
        )
    except TaskEnvSetupError as exc:
        print(f"setup-shared-torch-env failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(info))
    else:
        print(f"setup-shared-torch-env: {info['shared_torch_env']}")
        print(f"  python: {info['python_executable']}")
        print(f"  torch:  {info['torch_version']} cuda={info['torch_cuda']}")
        print(f"  gpu:    {info['device_name']}")
    return 0


def cmd_export_trajectory(args: argparse.Namespace) -> int:
    from .harness.trajectory import write_clean_events

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        print(f"trajectory input not found: {input_path}", file=sys.stderr)
        return 2
    count = write_clean_events(input_path, output_path)
    if args.json:
        print(json.dumps({"input": str(input_path), "output": str(output_path), "events": count}))
    else:
        print(f"clean trajectory: {output_path}")
        print(f"events: {count}")
    return 0


def _discover_task_ids(repo_root: Path, tasks_root_arg: str | None) -> list[str]:
    tasks_root = Path(tasks_root_arg).resolve() if tasks_root_arg else repo_root.parent / "tasks"
    return sorted(p.name for p in tasks_root.iterdir() if p.is_dir())


def cmd_prepare_tasks(args: argparse.Namespace) -> int:
    import csv
    import time

    from .registration import RegistrationError, register_task
    from .task_env import TaskEnvSetupError, setup_task_env, setup_shared_torch_env

    repo_root = Path(args.repo_root).resolve()
    tasks_root = Path(args.tasks_root).resolve() if args.tasks_root else repo_root.parent / "tasks"
    task_ids = list(args.task_ids) if args.task_ids else _discover_task_ids(repo_root, args.tasks_root)

    if args.setup_shared_torch:
        try:
            setup_shared_torch_env(
                repo_root=repo_root,
                force=bool(args.force_shared_torch),
                base_python=Path(args.python) if args.python else None,
                shared_torch_env=Path(args.shared_torch_env) if args.shared_torch_env else None,
                torch_cuda_index_url=args.torch_cuda_index_url,
                torch_version=args.torch_version,
                require_gpu_torch=bool(args.require_gpu_torch),
            )
        except TaskEnvSetupError as exc:
            print(f"shared torch setup failed: {exc}", file=sys.stderr)
            return 2

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        row: dict[str, Any] = {
            "task_id": task_id,
            "setup": "PENDING",
            "register": "PENDING",
            "python_executable": "",
            "manifest_path": "",
            "error": "",
        }
        try:
            env = setup_task_env(
                repo_root=repo_root,
                task_id=task_id,
                tasks_root=tasks_root,
                force=bool(args.force_env),
                base_python=Path(args.python) if args.python else None,
                shared_torch_env=Path(args.shared_torch_env) if args.shared_torch_env else None,
                torch_cuda_index_url=args.torch_cuda_index_url,
                torch_version=args.torch_version,
                require_gpu_torch=bool(args.require_gpu_torch),
                skip_notebook_packages=not bool(args.install_notebook_packages),
            )
            row["setup"] = "READY"
            row["python_executable"] = str(env.python_executable)
            if env.shared_torch_env:
                row["shared_torch_env"] = str(env.shared_torch_env)
            if env.torch_info:
                row["torch_cuda"] = env.torch_info.get("torch_cuda")
                row["torch_gpu"] = env.torch_info.get("device_name")
        except TaskEnvSetupError as exc:
            row["setup"] = "FAILED"
            row["register"] = "SKIPPED"
            row["error"] = str(exc)
            rows.append(row)
            print(f"[prepare] {task_id}: setup FAILED: {exc}", file=sys.stderr)
            continue

        try:
            reg = register_task(
                repo_root=repo_root,
                task_id=task_id,
                tasks_root=tasks_root,
                force=True,
                require_task_env=True,
            )
            row["register"] = "READY"
            row["manifest_path"] = str(reg.manifest_path)
        except RegistrationError as exc:
            row["register"] = "FAILED"
            row["error"] = str(exc)
            print(f"[prepare] {task_id}: register FAILED: {exc}", file=sys.stderr)

        rows.append(row)
        print(f"[prepare] {task_id}: setup={row['setup']} register={row['register']}")

    out_dir = repo_root / "artifacts" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    json_path = out_dir / f"prepare_{stamp}.json"
    csv_path = out_dir / f"prepare_{stamp}.csv"
    payload = {"task_ids": task_ids, "rows": rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    failed = [r for r in rows if r["setup"] != "READY" or r["register"] != "READY"]
    print(f"prepare complete: {len(rows)} task(s), {len(failed)} failed.")
    print(f"summary: {json_path}")
    print(f"csv:     {csv_path}")
    return 0 if not failed else 1


def cmd_run_task(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    manifest = _load_manifest(repo_root, args.task_id)

    model_provider_env: dict[str, str] = {}
    model_provider_summary: dict[str, Any] = {}
    artifact_model_slug = None
    try:
        provider_runtime = _resolve_model_provider_for_cli(
            repo_root=repo_root,
            model_id=getattr(args, "model_id", None),
            llm_config=getattr(args, "llm_config", None),
        )
    except (FileNotFoundError, ModelProviderError) as exc:
        print(f"model provider error: {exc}", file=sys.stderr)
        return 2

    if provider_runtime is not None:
        (
            resolved_model,
            model_provider_env,
            model_provider_summary,
            artifact_model_slug,
        ) = provider_runtime
    else:
        resolved_model = _resolve_default_model(args.model)
        artifact_model_slug = model_slug(resolved_model)
    if getattr(args, "artifact_model_slug", None):
        artifact_model_slug = model_slug(args.artifact_model_slug)
    if resolved_model:
        logger.info("using model: %s", resolved_model)
    config = HarnessConfig(
        repo_root=repo_root,
        manifest=manifest,
        max_rounds=args.max_rounds,
        budget_seconds=args.budget_seconds,
        max_turns_per_round=args.max_turns_per_round,
        model=resolved_model,
        model_provider_env=model_provider_env,
        model_provider_summary=model_provider_summary,
        skill_pack_dir=(
            Path(args.skill_pack_dir).resolve()
            if getattr(args, "skill_pack_dir", None)
            else None
        ),
        artifact_model_slug=artifact_model_slug,
        judge_python=args.judge_python,
        show_metric_status=bool(args.show_metric_status),
        keep_workspace_on_success=bool(args.keep_workspace),
        sandbox_root=Path(args.sandbox_root) if args.sandbox_root else None,
        keep_sandbox=bool(args.keep_sandbox),
        record_thinking=bool(args.record_thinking),
    )

    outcome = run_task_once(config)

    payload = {
        "task_id": outcome.task_id,
        "run_id": outcome.run_id,
        "verdict": outcome.verdict,
        "rounds_used": outcome.rounds_used,
        "runtime_seconds": outcome.runtime_seconds,
        "summary_path": str(outcome.summary_path),
        "trajectory_path": str(outcome.trajectory_path),
        "log_root": str(outcome.log_root),
        "workspace_root": str(outcome.workspace_root),
        "error": outcome.error,
    }

    if args.json:
        # print exactly one JSON line at the very end so the parent process
        # (concurrency pool) can find it deterministically.
        print(json.dumps(payload))
    else:
        print(f"verdict: {payload['verdict']}")
        print(f"rounds:  {payload['rounds_used']}")
        print(f"summary: {payload['summary_path']}")
        print(f"traj:    {payload['trajectory_path']}")

    return 0 if outcome.verdict == "PASS" else 1


def cmd_run_batch(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    task_ids = args.task_ids
    extra: dict[str, Any] = {
        "max-rounds": args.max_rounds,
        "budget-seconds": args.budget_seconds,
        "max-turns-per-round": args.max_turns_per_round,
    }
    if getattr(args, "model_id", None):
        extra["model-id"] = args.model_id
        if getattr(args, "llm_config", None):
            extra["llm-config"] = args.llm_config
    else:
        resolved_model = _resolve_default_model(args.model)
        if resolved_model:
            logger.info("using model: %s", resolved_model)
            extra["model"] = resolved_model
    if args.judge_python:
        extra["judge-python"] = args.judge_python
    if not bool(getattr(args, "show_metric_status", True)):
        extra["hide-metric-status"] = True
    if args.keep_sandbox:
        extra["keep-sandbox"] = True
    if args.record_thinking:
        extra["record-thinking"] = True
    if not args.keep_workspace:
        extra["delete-workspace-on-success"] = True

    outcomes = run_tasks_parallel(
        repo_root=repo_root,
        task_ids=task_ids,
        max_workers=args.max_workers,
        extra_run_args=extra,
        timeout_seconds=args.budget_seconds + 600,
    )

    # Persist a batch-level summary.
    summary_path = repo_root / "artifacts" / "logs" / f"batch_{int(__import__('time').time())}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_ids": list(task_ids),
        "outcomes": [asdict(o) for o in outcomes],
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = [o for o in outcomes if not o.success]
    print(f"\nbatch complete: {len(outcomes)} task(s), {len(failures)} failed.")
    print(f"summary: {summary_path}")
    for o in outcomes:
        print(f"  {o.task_id:30s} {o.verdict:8s} ({o.runtime_seconds:6.1f}s)")
    return 0 if not failures else 1


def _build_anthropic_polish(
    *, repo_root: Path, model_id: str | None, llm_config: str | None
):
    """Construct an LLMPolishFn backed by the project's llm.yaml gateway.

    Returns ``None`` if ``model_id`` is falsy or the SDK / config is
    missing; in that case the synthesizer falls back to the deterministic
    playbook draft (no LLM calls, still produces a valid SKILL.md).
    """

    if not model_id:
        return None
    try:
        config_path = (
            Path(llm_config).resolve()
            if llm_config
            else default_llm_config_path(repo_root).resolve()
        )
        registry = load_model_provider_registry(config_path)
        runtime = registry.resolve_claude_gateway_runtime(model_id)
    except (FileNotFoundError, ModelProviderError) as exc:
        logger.warning("distill-skill: cannot load model_id=%s (%s); "
                       "falling back to deterministic draft", model_id, exc)
        return None

    try:
        # The Anthropic SDK is an optional dependency. Import lazily so
        # that running `distill-skill` without `--model-id` works on
        # machines that never installed it.
        import anthropic  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time only
        logger.warning("distill-skill: anthropic SDK not importable (%s); "
                       "falling back to deterministic draft", exc)
        return None

    api_key = runtime.env.get("ANTHROPIC_API_KEY") or runtime.model_config.api_key
    base_url = runtime.env.get("ANTHROPIC_BASE_URL") or runtime.model_config.base_url
    model_name = runtime.model_config.model_name

    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    SYSTEM_PROMPT = (
        "You are an expert curriculum author distilling a *transferable* "
        "skill from successful agent trajectories AND the train-task "
        "source code that produced them.\n"
        "\n"
        "INPUT you receive: (a) a deterministic Markdown draft we already "
        "wrote, and (b) JSON evidence with two parts:\n"
        "  - `episodes`: per-train-task tool-use signatures and judge "
        "feedback tags;\n"
        "  - `source_evidence`: per-train-task scrubbed snippets of the "
        "actual reference source files (README, agent_task_spec.json, "
        "and files the agent opened during the run);\n"
        "  - `train_gap_evidence`: train-only failed/timeout attempts "
        "compared against train source, used for generic anti-patterns.\n"
        "\n"
        "GOAL: rewrite the draft into a tight, *non-hardcoded* Markdown "
        "playbook that helps a future agent on a *different* but related "
        "task in the same domain family. Cross-check the draft against "
        "what the source actually requires (forward model shapes, primary "
        "output schema, common pitfalls). Use train_gap_evidence to add "
        "missing algorithm families and timeout-avoidance tactics. "
        "Make the first screen actionable: tell the future agent to load the "
        "skill before writing the solver, run bundled helper scripts for "
        "npz inspection, schema-shaped guard outputs from public arrays, "
        "FFT-grid/Stolt mapping checks, SSNP/ODT sampling checks, CFL checks, "
        "and simple tomography baselines, then choose a physics route. "
        "Generalise across tasks; do "
        "not paste literal code or task_ids; do not invent thresholds. "
        "The polished skill MUST include `## Routes`, `## Metric Diagnostic`, "
        "and `## Anti-Patterns` sections. `Routes` must map public README/data "
        "signals to algorithm routes and required checks. `Metric Diagnostic` "
        "must map failed metric patterns to first diagnostic checks, next "
        "actions, and give-up signals. `Anti-Patterns` must come from "
        "train-only failures/timeouts and must not mention task ids. "
        "hard anti-timeout lessons from failed train attempts: for large 3D "
        "diffraction/tomography volumes, write an intensity/projection guard "
        "once and avoid full-volume autograd through all slices, angles, and "
        "iterations unless cropped/downsampled timing probes prove the exact "
        "loop is cheap. If the guard fails, do not resubmit it unchanged; "
        "switch routes. For f-k/Stolt "
        "migration, emphasize axis/FFT/interpolation/Jacobian checks plus delay/TOF "
        "alignment before cropping and round-trip or virtual-wave speed. For "
        "waveform inversion, emphasize saving any same-shaped public initial "
        "or smoothed model as a one-time schema guard before solver work; "
        "only refine after a tiny timing probe proves the full loop is cheap. "
        "If a timing probe estimates the documented optimization will exceed "
        "budget, do not implement a shortened CPML/FWI solver just to fit "
        "the clock. If CFL substeps, memory, or single-shot runtime look risky, "
        "stop the long path and revise the solver rather than repeating the guard. Also "
        "emphasize adjoint-state gradients and sparse checkpointing instead "
        "of black-box autograd through every timestep. "
        "Make this visible in the Outline, before any solver-family details, "
        "so a future agent writes the first `plan.md` around probes and cheap "
        "guards rather than copying an expensive README hint.\n"
        "\n"
        "HARD CONSTRAINTS:\n"
        "- Output ONLY Markdown body content (no YAML frontmatter, no "
        "code fences around the whole reply).\n"
        "- Do NOT mention any train-task id or valid-task id by name.\n"
        "- Do NOT include absolute filesystem paths or API keys.\n"
        "- Keep total length under ~6000 characters.\n"
        "- Preserve `## When to use`, `## Outline`, `## Helper scripts`, "
        "`## Routes`, `## Metric Diagnostic`, `## Anti-Patterns`, "
        "and `## Self-check` sections; you may add others.\n"
        "- Do NOT write that a baseline or guard is a final/default final "
        "answer. It is a one-time schema guard; if it fails, route to the "
        "algorithmic solver instead of repeating it.\n"
    )

    def polish(draft: str, evidence: Mapping[str, object]) -> str:
        # Compact the evidence so we don't blow the context window.
        # Truncate snippets aggressively; the LLM only needs the gist.
        compact_episodes: list[dict[str, object]] = []
        for ep in evidence.get("episodes", []) or []:  # type: ignore[union-attr]
            if not isinstance(ep, Mapping):
                continue
            tools = list(ep.get("tools", []) or [])[:8]
            failures = list(ep.get("failures", []) or [])[:6]
            compact_episodes.append({
                "task_tag": "train-task-#" + str(len(compact_episodes) + 1),
                "rounds_used": ep.get("rounds_used"),
                "metrics_actual": ep.get("metrics_actual"),
                "metric_status": ep.get("metric_status"),
                "plan_summary": ep.get("plan_summary"),
                "main_py_digest": ep.get("main_py_digest"),
                "tools": tools,
                "failures": failures,
            })
        compact_sources: list[dict[str, object]] = []
        for item in evidence.get("source_evidence", []) or []:  # type: ignore[union-attr]
            if not isinstance(item, Mapping):
                continue
            snips = []
            for s in item.get("snippets", []) or []:
                if not isinstance(s, Mapping):
                    continue
                txt = str(s.get("snippet") or "")
                snips.append({
                    "rel_path": s.get("rel_path"),
                    "kind": s.get("kind"),
                    "bytes": s.get("bytes"),
                    "snippet": txt[:1500],  # extra cap on top of the
                                            # 4 KB cap from the collector
                })
            compact_sources.append({
                "task_tag": "train-task-#" + str(len(compact_sources) + 1),
                "primary_output_rel": item.get("primary_output_rel"),
                "snippets": snips,
            })
        compact_gaps: list[dict[str, object]] = []
        for item in evidence.get("train_gap_evidence", []) or []:  # type: ignore[union-attr]
            if not isinstance(item, Mapping):
                continue
            gap_sources = []
            for s in item.get("source_hint", []) or []:
                if not isinstance(s, Mapping):
                    continue
                gap_sources.append({
                    "rel_path": s.get("rel_path"),
                    "bytes": s.get("bytes"),
                    "snippet": str(s.get("snippet") or "")[:1200],
                })
            attempt = item.get("agent_attempt") or {}
            if not isinstance(attempt, Mapping):
                attempt = {}
            compact_gaps.append({
                "task_tag": "train-gap-#" + str(len(compact_gaps) + 1),
                "failure_mode": item.get("failure_mode"),
                "metric_statuses": item.get("metric_statuses"),
                "metrics_actual": item.get("metrics_actual"),
                "agent_attempt": {
                    "plan": str(attempt.get("plan") or "")[:1200],
                    "trajectory": str(attempt.get("trajectory") or "")[:1800],
                    "main_py_digest": attempt.get("main_py_digest") or {},
                },
                "source_hint": gap_sources[:6],
                "transferable_lesson": item.get("transferable_lesson"),
            })

        user_payload = (
            "## Deterministic draft\n\n" + draft + "\n\n"
            "## Evidence (JSON)\n\n```json\n"
            + json.dumps(
                {
                    "episodes": compact_episodes,
                    "source_evidence": compact_sources,
                    "train_gap_evidence": compact_gaps,
                },
                ensure_ascii=False,
                indent=2,
            )[:60_000]
            + "\n```\n"
        )
        msg = client.messages.create(
            model=model_name,
            max_tokens=4_096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        )
        # Concatenate any text blocks from the response.
        out_parts: list[str] = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                out_parts.append(text)
        return "".join(out_parts).strip()

    return polish


def cmd_distill_skill(args: argparse.Namespace) -> int:
    """Mine train-split episodes, synthesise a SKILL pack, sanitise, write.

    Does NOT run any harness; pure offline derivation from existing
    trajectories + train-task source. Output: a directory under
    ``--out-root`` named after ``--skill-id`` containing SKILL.md.

    Use ``--model-id`` to enable the LLM polish layer (writes a richer,
    cross-checked SKILL.md). Without it, you get the deterministic
    playbook draft (still a valid Anthropic-format skill, just terser).
    """

    from .distill import (
        DistillUniverse,
        SanitizationError,
        ValidationLeakError,
        mine_train_split,
        synthesize_skill,
        write_skill_pack,
    )

    repo_root = Path(args.repo_root).resolve()
    split_path = Path(args.split).resolve()
    out_root = Path(args.out_root).resolve()
    audit_path = (
        Path(args.audit_log).resolve()
        if args.audit_log
        else (out_root / "_audit" / f"{args.skill_id}.audit.jsonl")
    )

    if not split_path.exists():
        print(f"split file not found: {split_path}", file=sys.stderr)
        return 2

    universe = DistillUniverse.from_split_file(repo_root, split_path)
    universe.bind_audit_log(audit_path)

    episodes = mine_train_split(universe)
    pass_episodes = [e for e in episodes if e.final_verdict == "PASS"]
    gap_episodes = [e for e in episodes if e.final_verdict != "PASS"]
    if not pass_episodes:
        print(
            f"no passing train episodes found under "
            f"artifacts/logs/{universe.model_slug}/<train_task>/run-* "
            f"for split {split_path.name}",
            file=sys.stderr,
        )
        print("hint: run `myevoskill run-task` on at least one train task "
              "until it PASSES, then retry distill-skill", file=sys.stderr)
        return 1

    logger.info(
        "distill-skill: mined %d passing episode(s) and %d train-only gap episode(s) across %d train task(s)",
        len(pass_episodes), len(gap_episodes), len({e.task_id for e in episodes}),
    )

    polish_fn = _build_anthropic_polish(
        repo_root=repo_root,
        model_id=getattr(args, "model_id", None),
        llm_config=getattr(args, "llm_config", None),
    )
    if polish_fn is None:
        logger.info("distill-skill: deterministic draft only (no LLM polish)")
    else:
        logger.info("distill-skill: LLM polish enabled (model_id=%s)", args.model_id)

    try:
        spec = synthesize_skill(
            skill_id=args.skill_id,
            episodes=episodes,
            universe=universe,
            llm_polish=polish_fn,
        )
    except (PermissionError, ValueError) as exc:
        print(f"distill-skill failed: {exc}", file=sys.stderr)
        return 2

    try:
        pack_dir = write_skill_pack(
            spec,
            out_root,
            valid_task_ids=universe.valid_task_ids,
            train_task_ids=universe.train_task_ids,
        )
    except SanitizationError as exc:
        print(f"distill-skill: sanitizer rejected the synthesised pack: {exc}",
              file=sys.stderr)
        print("inspect the *.rejected/ dir alongside the target out-root for "
              "the offending bytes.", file=sys.stderr)
        return 3

    # Defence-in-depth tail check: confirm the audit recorded zero allowed
    # accesses to valid-split tasks.
    try:
        universe.assert_no_valid_access()
    except ValidationLeakError as exc:
        print(f"distill-skill: AUDIT LEAK: {exc}", file=sys.stderr)
        return 4

    payload = {
        "skill_id": spec.skill_id,
        "pack_dir": str(pack_dir),
        "skill_md": str(pack_dir / "SKILL.md"),
        "train_task_count": len(spec.train_task_ids),
        "primary_output_rel": spec.primary_output_rel,
        "audit_log": str(audit_path),
        "polish_used": polish_fn is not None,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"skill_id:        {payload['skill_id']}")
        print(f"pack_dir:        {payload['pack_dir']}")
        print(f"SKILL.md:        {payload['skill_md']}")
        print(f"trained_on:      {payload['train_task_count']} train task(s)")
        print(f"primary_output:  {payload['primary_output_rel']}")
        print(f"audit_log:       {payload['audit_log']}")
        print(f"llm_polish:      {payload['polish_used']}")
        print()
        print("next: validate with")
        print(f"  python -m myevoskill.cli validate-skill \\")
        print(f"    --skill-pack-dir {pack_dir} \\")
        print(f"    --split {split_path}")
    return 0


def cmd_validate_skill(args: argparse.Namespace) -> int:

    """Validation gate for a freshly distilled skill pack.

    By default this runs each selected valid-split task once with
    ``--skill-pack-dir`` injected. ``--compare-baseline`` restores the
    original baseline-vs-skill transfer comparison.
    """

    from .distill.transfer_validator import (
        VERDICT_PROMOTE,
        stamp_promotion,
        validate_skill,
        write_transfer_report,
    )
    from .distill.universe import DistillUniverse

    repo_root = Path(args.repo_root).resolve()
    pack_dir = Path(args.skill_pack_dir).resolve()
    split_path = Path(args.split).resolve()

    if not pack_dir.exists():
        print(f"skill pack not found: {pack_dir}", file=sys.stderr)
        return 2
    if not split_path.exists():
        print(f"split file not found: {split_path}", file=sys.stderr)
        return 2

    # Build the universe (we only consult `is_train` / `valid_task_ids`
    # here; no source-code reads happen).
    universe = DistillUniverse.from_split_file(repo_root, split_path)

    # Resolve the model once so both baseline and +skill runs use the
    # same routing / artifact slug.
    try:
        provider_runtime = _resolve_model_provider_for_cli(
            repo_root=repo_root,
            model_id=getattr(args, "model_id", None),
            llm_config=getattr(args, "llm_config", None),
        )
    except (FileNotFoundError, ModelProviderError) as exc:
        print(f"model provider error: {exc}", file=sys.stderr)
        return 2

    if provider_runtime is not None:
        (
            resolved_model,
            model_provider_env,
            model_provider_summary,
            artifact_model_slug,
        ) = provider_runtime
    else:
        resolved_model = _resolve_default_model(args.model)
        model_provider_env = {}
        model_provider_summary = {}
        artifact_model_slug = model_slug(resolved_model)
    if getattr(args, "artifact_model_slug", None):
        artifact_model_slug = model_slug(args.artifact_model_slug)

    if resolved_model:
        logger.info("validate-skill using model: %s", resolved_model)

    valid_ids = (
        list(args.valid_task_ids) if args.valid_task_ids else list(universe.valid_task_ids)
    )

    # Build the per-task runner closure. It calls ``run_task_once`` with
    # exactly the same options run-task uses, plus skill_pack_dir when
    # ``with_skill`` is True. Verdict strings come straight from the
    # HarnessOutcome.
    def _runner(task_id: str, with_skill: bool) -> str:
        manifest = _load_manifest(repo_root, task_id)
        config = HarnessConfig(
            repo_root=repo_root,
            manifest=manifest,
            max_rounds=args.max_rounds,
            budget_seconds=args.budget_seconds,
            max_turns_per_round=args.max_turns_per_round,
            model=resolved_model,
            model_provider_env=model_provider_env,
            model_provider_summary=model_provider_summary,
            skill_pack_dir=pack_dir if with_skill else None,
            artifact_model_slug=artifact_model_slug,
            judge_python=args.judge_python,
            show_metric_status=bool(args.show_metric_status),
            keep_workspace_on_success=bool(args.keep_workspace),
            keep_sandbox=False,
            record_thinking=False,
        )
        outcome = run_task_once(config)
        logger.info(
            "[validate] %s with_skill=%s -> %s (run_id=%s, log=%s)",
            task_id, with_skill, outcome.verdict, outcome.run_id, outcome.log_root,
        )
        return outcome.verdict

    try:
        report = validate_skill(
            universe=universe,
            skill_pack_dir=pack_dir,
            runner=_runner,
            valid_task_ids=valid_ids,
            compare_baseline=bool(args.compare_baseline),
        )
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        print(f"validate-skill failed: {exc}", file=sys.stderr)
        return 2

    out_path = (
        Path(args.report_path).resolve()
        if args.report_path
        else None
    )
    written = write_transfer_report(report, out_path=out_path)
    promo_path = stamp_promotion(report, model_slug=artifact_model_slug or "unknown")

    if args.json:
        print(json.dumps({
            "verdict": report.verdict,
            "summary": report.summary(),
            "report_path": str(written),
            "promotion_path": str(promo_path) if promo_path else None,
        }))
    else:
        s = report.summary()
        print(f"verdict:    {report.verdict}")
        print(f"mode:       {report.mode}")
        print(f"valid:      {s['n_valid']}")
        if report.mode == "compare":
            print(f"baseline:   {s['n_baseline_pass']} PASS")
        else:
            print("baseline:   SKIPPED")
        print(f"+skill:     {s['n_plus_skill_pass']} PASS")
        if report.mode == "compare":
            print(f"new_pass:   {s['n_new_pass']}")
            print(f"regression: {s['n_regression']}")
        print(f"report:     {written}")
        if promo_path is not None:
            print(f"promotion:  {promo_path}")
        for reason in report.rejection_reasons:
            print(f"reject:     {reason}")

    return 0 if report.verdict == VERDICT_PROMOTE else 1


# --------------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myevoskill",
        description="MyEvoSkill harness CLI.",
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # setup-task-env
    p_env = sub.add_parser(
        "setup-task-env",
        help="create/update the per-task venv used by register-task",
    )
    p_env.add_argument("--task-id", required=True)
    p_env.add_argument("--repo-root", default=".")
    p_env.add_argument("--tasks-root", default=None,
                       help="Override tasks/ root (defaults to <repo_root>/../tasks)")
    p_env.add_argument("--force", action="store_true")
    p_env.add_argument("--python", default=None,
                       help="base Python executable used to create the venv")
    p_env.add_argument("--shared-torch-env", default=None,
                       help="shared CUDA Torch venv reused by Torch tasks")
    p_env.add_argument("--torch-cuda-index-url",
                       default="https://download.pytorch.org/whl/cu118")
    p_env.add_argument("--torch-version", default="2.5.1")
    p_env.add_argument("--require-gpu-torch", action="store_true",
                       help="for Torch tasks, fail setup unless torch sees a CUDA GPU")
    p_env.add_argument("--install-notebook-packages", action="store_true",
                       help="install jupyter/notebook packages from task requirements instead of filtering them")
    p_env.set_defaults(func=cmd_setup_task_env)

    # setup-shared-torch-env
    p_torch = sub.add_parser(
        "setup-shared-torch-env",
        help="create/update the shared CUDA Torch runtime venv",
    )
    p_torch.add_argument("--repo-root", default=".")
    p_torch.add_argument("--force", action="store_true")
    p_torch.add_argument("--python", default=None,
                         help="base Python executable used to create the venv")
    p_torch.add_argument("--shared-torch-env", default=None,
                         help="default: .venvs/_torch-cu118-py310")
    p_torch.add_argument("--torch-cuda-index-url",
                         default="https://download.pytorch.org/whl/cu118")
    p_torch.add_argument("--torch-version", default="2.5.1")
    p_torch.add_argument("--require-gpu-torch", action="store_true", default=True)
    p_torch.add_argument("--json", action="store_true")
    p_torch.set_defaults(func=cmd_setup_shared_torch_env)

    # export-trajectory
    p_export = sub.add_parser(
        "export-trajectory",
        help="write a distillation-clean trajectory JSONL",
    )
    p_export.add_argument("--input", required=True, help="raw trajectory JSONL path")
    p_export.add_argument("--output", required=True, help="clean trajectory JSONL path")
    p_export.add_argument("--json", action="store_true", help="emit one JSON summary line")
    p_export.set_defaults(func=cmd_export_trajectory)

    # prepare-tasks
    p_prepare = sub.add_parser(
        "prepare-tasks",
        help="setup per-task envs and register manifests, writing JSON/CSV summary",
    )
    p_prepare.add_argument("--repo-root", default=".")
    p_prepare.add_argument("--tasks-root", default=None)
    p_prepare.add_argument("--task-ids", nargs="+", default=None)
    p_prepare.add_argument("--python", default=None)
    p_prepare.add_argument("--force-env", action="store_true")
    p_prepare.add_argument("--setup-shared-torch", action="store_true", default=True)
    p_prepare.add_argument("--force-shared-torch", action="store_true")
    p_prepare.add_argument("--shared-torch-env", default=None)
    p_prepare.add_argument("--torch-cuda-index-url",
                           default="https://download.pytorch.org/whl/cu118")
    p_prepare.add_argument("--torch-version", default="2.5.1")
    p_prepare.add_argument("--require-gpu-torch", action="store_true")
    p_prepare.add_argument("--install-notebook-packages", action="store_true")
    p_prepare.set_defaults(func=cmd_prepare_tasks)

    # register-task
    p_reg = sub.add_parser("register-task", help="register / refresh a task manifest")
    p_reg.add_argument("--task-id", required=True)
    p_reg.add_argument("--repo-root", default=".")
    p_reg.add_argument("--tasks-root", default=None,
                       help="Override tasks/ root (defaults to <repo_root>/../tasks)")
    p_reg.add_argument("--force", action="store_true")
    p_reg.add_argument(
        "--require-task-env",
        action="store_true",
        help=(
            "Refuse to register unless runtime_logs/setup/<task_id>.json "
            "reports a ready per-task venv (produced by setup_task_env.sh)."
        ),
    )
    p_reg.set_defaults(func=cmd_register_task)

    # run-task
    p_run = sub.add_parser("run-task", help="run a single registered task")
    p_run.add_argument("--task-id", required=True)
    p_run.add_argument("--repo-root", default=".")
    p_run.add_argument("--max-rounds", type=int, default=4)
    p_run.add_argument("--budget-seconds", type=int, default=7200)
    p_run.add_argument("--max-turns-per-round", type=int, default=60)
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--model-id", default=None,
                       help="model id from config/llm.yaml; requires an Anthropic-compatible gateway")
    p_run.add_argument("--llm-config", default=None,
                       help="path to llm.yaml (default: <repo_root>/config/llm.yaml)")
    p_run.add_argument("--artifact-model-slug", default=None,
                       help="override artifacts/logs/<model_slug>/ and artifacts/workspaces/<model_slug>/")
    p_run.add_argument("--judge-python", default=None)
    p_run.add_argument(
        "--show-metric-status",
        dest="show_metric_status",
        action="store_true",
        default=True,
        help="show per-metric pass/fail feedback without numeric values (default)",
    )
    p_run.add_argument(
        "--hide-metric-status",
        dest="show_metric_status",
        action="store_false",
        help="hide per-metric pass/fail feedback from the agent",
    )
    p_run.add_argument(
        "--keep-workspace",
        dest="keep_workspace",
        action="store_true",
        default=True,
        help="keep the run workspace after PASS (default)",
    )
    p_run.add_argument(
        "--delete-workspace-on-success",
        dest="keep_workspace",
        action="store_false",
        help="delete the run workspace after PASS to save disk",
    )
    p_run.add_argument("--keep-sandbox", action="store_true",
                       help="do not wipe the per-run isolated $HOME on exit (debug only)")
    p_run.add_argument("--skill-pack-dir", default=None,
                       help="optional directory containing a sanitised .claude/skills/ pack to inject into the workspace")
    p_run.add_argument("--sandbox-root", default=None,
                       help="override sandbox dir (default: artifacts/sandboxes/<model>/<task>/<run>/home)")
    p_run.add_argument("--json", action="store_true", help="emit one JSON summary line at end")
    p_run.add_argument("--record-thinking", action="store_true",
                       help="debug only: keep SDK thinking blocks in raw trajectory")
    p_run.set_defaults(func=cmd_run_task)

    # run-batch
    p_batch = sub.add_parser("run-batch", help="run several tasks in parallel subprocesses")
    p_batch.add_argument("--repo-root", default=".")
    p_batch.add_argument("--task-ids", nargs="+", required=True)
    p_batch.add_argument("--max-workers", type=int, default=2)
    p_batch.add_argument("--max-rounds", type=int, default=4)
    p_batch.add_argument("--budget-seconds", type=int, default=7200)
    p_batch.add_argument("--max-turns-per-round", type=int, default=60)
    p_batch.add_argument("--model", default=None)
    p_batch.add_argument("--model-id", default=None,
                         help="model id from config/llm.yaml; requires an Anthropic-compatible gateway")
    p_batch.add_argument("--llm-config", default=None,
                         help="path to llm.yaml (default: <repo_root>/config/llm.yaml)")
    p_batch.add_argument("--judge-python", default=None)
    p_batch.add_argument(
        "--show-metric-status",
        dest="show_metric_status",
        action="store_true",
        default=True,
        help="show per-metric pass/fail feedback without numeric values (default)",
    )
    p_batch.add_argument(
        "--hide-metric-status",
        dest="show_metric_status",
        action="store_false",
        help="propagate --hide-metric-status to every child run-task",
    )
    p_batch.add_argument("--keep-sandbox", action="store_true",
                         help="propagate --keep-sandbox to every child run-task")
    p_batch.add_argument("--record-thinking", action="store_true",
                         help="propagate --record-thinking to every child run-task")
    p_batch.add_argument(
        "--keep-workspace",
        dest="keep_workspace",
        action="store_true",
        default=True,
        help="keep successful child workspaces after PASS (default)",
    )
    p_batch.add_argument(
        "--delete-workspace-on-success",
        dest="keep_workspace",
        action="store_false",
        help="propagate --delete-workspace-on-success to every child run-task",
    )
    p_batch.set_defaults(func=cmd_run_batch)

    # distill-skill
    p_dist = sub.add_parser(
        "distill-skill",
        help="mine train-split passing runs into a sanitised SKILL pack",
    )
    p_dist.add_argument("--repo-root", default=".")
    p_dist.add_argument("--split", required=True,
                        help="path to the train/valid split JSON")
    p_dist.add_argument("--skill-id", required=True,
                        help="slug for the skill pack directory (lowercase a-z 0-9 _)")
    p_dist.add_argument("--out-root", default="artifacts/skills",
                        help="parent dir for skill packs (default: artifacts/skills)")
    p_dist.add_argument("--audit-log", default=None,
                        help="audit JSONL path (default: <out_root>/_audit/<skill_id>.audit.jsonl)")
    p_dist.add_argument("--model-id", default=None,
                        help="optional llm.yaml model id for the polish pass; "
                             "if omitted, ships the deterministic playbook draft")
    p_dist.add_argument("--llm-config", default=None,
                        help="path to llm.yaml (default: <repo_root>/config/llm.yaml)")
    p_dist.add_argument("--json", action="store_true",
                        help="emit one JSON summary line")
    p_dist.set_defaults(func=cmd_distill_skill)

    # validate-skill
    p_val = sub.add_parser(
        "validate-skill",

        help="run the skill pack on selected valid tasks",
    )
    p_val.add_argument("--repo-root", default=".")
    p_val.add_argument("--skill-pack-dir", required=True,
                       help="distilled skill pack dir containing SKILL.md")
    p_val.add_argument("--split", required=True,
                       help="path to the train/valid split JSON (e.g. registry/splits/wave_optics_v1.json)")
    p_val.add_argument("--valid-task-ids", nargs="+", default=None,
                       help="optional subset of valid task_ids to evaluate (default: all)")
    p_val.add_argument("--compare-baseline", action="store_true",
                       help=("also run a no-skill baseline and apply the original "
                             "baseline-vs-skill promotion rule"))
    p_val.add_argument("--max-rounds", type=int, default=4)
    p_val.add_argument("--budget-seconds", type=int, default=7200)
    p_val.add_argument("--max-turns-per-round", type=int, default=60)
    p_val.add_argument("--model", default=None)
    p_val.add_argument("--model-id", default=None,
                       help="model id from config/llm.yaml; requires an Anthropic-compatible gateway")
    p_val.add_argument("--llm-config", default=None,
                       help="path to llm.yaml (default: <repo_root>/config/llm.yaml)")
    p_val.add_argument("--artifact-model-slug", default=None,
                       help="override artifacts/logs/<model_slug>/ and artifacts/workspaces/<model_slug>/ for validation runs")
    p_val.add_argument("--judge-python", default=None)
    p_val.add_argument(
        "--show-metric-status",
        dest="show_metric_status",
        action="store_true",
        default=True,
    )
    p_val.add_argument(
        "--hide-metric-status",
        dest="show_metric_status",
        action="store_false",
    )
    p_val.add_argument(
        "--keep-workspace",
        dest="keep_workspace",
        action="store_true",
        default=True,
    )
    p_val.add_argument(
        "--delete-workspace-on-success",
        dest="keep_workspace",
        action="store_false",
    )
    p_val.add_argument("--report-path", default=None,
                       help="output path for the TransferReport JSON "
                            "(default: <skill_pack_dir>/transfer_report.json)")
    p_val.add_argument("--json", action="store_true", help="emit one JSON summary line")
    p_val.set_defaults(func=cmd_validate_skill)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
