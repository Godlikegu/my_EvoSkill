# Executor Adapter Specification

## Goal

Abstract task execution so MyEvoSkill can orchestrate scientific tasks without
binding the system to a single vendor SDK.

## Required Interfaces

Each executor implements:

`run(task_bundle, session_config, active_skills) -> RunRecord`

Inputs:

- `task_bundle`
  Provides the compiled `public_bundle`, task metadata, and manifest-derived
  runtime spec.
- `session_config`
  Contains run id, env hash, runtime budget, workspace, runtime command, and
  optional model configuration.
- `active_skills`
  List of skill ids or paths active for the run.

## Built-in Executors

- `LocalRunnerAdapter`
  Local subprocess runner and the current guaranteed fallback path.
- `ClaudeWorkspaceAdapter`
  Claude SDK constrained workspace agent and the current reference
  model-backed execution path for `cars_spectroscopy`.
- `InspectBridgeAdapter`
  Optional adapter for Inspect AI bridge integrations.
- `ClaudeAdapter`
  Compatibility wrapper around `ClaudeWorkspaceAdapter`.
- `OpenHandsAdapter`
  Optional adapter for OpenHands runtime.

## Inspect Bridge State Machine

`InspectBridgeAdapter` must be testable without depending on whether
`inspect_ai` happens to be installed in the active machine or conda
environment. Its dependency detection therefore needs an explicit control point
for tests.

Required behavior:

- `inspect_ai` unavailable and `allow_fallback=True`
  Return a `RunRecord` with:
  - `provider="inspect_unavailable_fallback"`
  - `metadata["fallback_reason"] == "inspect_ai_missing"`
- `inspect_ai` unavailable and `allow_fallback=False`
  Raise `RuntimeError`
- `inspect_ai` available but bridge execution still placeholder-only
  Return a `RunRecord` with:
  - `provider="inspect_bridge"`
  - `metadata["bridge_mode"] == "placeholder"`

For legacy model-backed runs, `InspectBridgeAdapter` must additionally:

- require `ModelConfig`
- resolve the API key through `api_key_env`
- record only `api_key_env`, never the resolved key itself
- write generated code and transcripts into the run workspace
- keep execution constrained to the compiled `public_bundle`
- stage public files into a standardized runtime root and run with
  `cwd=runtime_root`
- resolve effective timeouts from manifest/session defaults before request or
  execution starts
- prefer a structured JSON response contract with:
  - `python_code`
  - `declared_outputs`
  - `assumptions`
  - `solver_summary`
- execute only the `python_code` field when structured JSON is returned
- fall back to plain-text Python extraction only when structured JSON parsing
  fails
- sanitize real model responses into executable Python code before writing
  `agent_solution.py`
- reject empty or non-code model responses with a clear runtime error
- try structured JSON candidates in a stable order:
  - full text
  - fenced `json` block
  - fenced untyped block
  - first balanced JSON object slice from raw text

The current implementation is allowed to reuse fallback-style execution as the
placeholder bridge path, but the returned metadata must make that fact visible
for debugging.

## Claude Workspace Agent

`ClaudeWorkspaceAdapter` is the primary reference path for live scientific task
runs. Its main path uses `provider_name="claude-sdk"` and a Claude Code style
tool harness rather than a single chat-completions response.

Required behavior:

- the SDK-backed agent reads public files through explicit tools such as:
  - `Read`
  - `Write`
  - `Bash`
  - `Glob`
  - `Grep`
- the adapter stages public inputs read-only and preserves a read-only
  `public_bundle/` mirror
- only `work/`, `output/`, and `checkpoints/` are writable during execution
- the adapter runs `python work/main.py` from the runtime root
- the adapter generates a read-only public evaluator at:
  - `evaluation/self_eval.py`
  - `evaluation/self_eval_spec.json`
- the adapter uses `python evaluation/self_eval.py` as the default public stop oracle
- the adapter does not perform outer repair rounds or output-based interrupts
- hidden judge signals are never fed back into the workspace agent
- the final agent response is a structured summary containing:
  - `solver_summary`
  - `declared_outputs`
  - `assumptions`
  - `files_written`
  - `commands_run`
- metadata must additionally include:
  - `sdk_backend = "claude_sdk"`
  - `allowed_tools`
  - `tool_policy_summary`
  - `stop_oracle = "public_self_eval"`
  - `agent_stop_policy = "run_self_eval_then_summary"`
  - `public_self_eval_seen_in_trace`
  - `public_self_eval_passed_post_run`

Prompt contract metadata for this path should use
`prompt_contract_version = "v7_claude_sdk_public_self_eval"`.

The formal live/reference executor is Claude SDK only. Legacy OpenAI-compatible
workspace JSON code paths may still exist for compatibility tests, but they are
not part of the formal execution pipeline.

Prompt requirements for the Claude path are intentionally lightweight:

- tell the agent the resolved `workspace_root` and `cwd`
- state that `README_public.md` is authoritative
- prefer relative paths over guessed absolute paths
- restrict writes to `work/`, `output/`, and `checkpoints/`
- instruct the agent to run `python evaluation/self_eval.py` after `python work/main.py`
- require the agent to stop immediately after `python evaluation/self_eval.py` passes
- render a manifest-driven task template with:
  - `## Problem Description` from `README_public.md`
  - `## Data Specification` from public metadata and `requirements.txt`
  - `## IMPORTANT_RULES` from the manifest output contract and harness rules
  - `## Recommended Workflow` without hidden-judge optimization hints
- fail explicitly when manifest `output_contract.required_outputs` is missing instead of guessing output fields

Default Claude SDK option requirements:

- `allowed_tools = [Read, Write, Bash, Glob, Grep]`
- `disallowed_tools` includes `WebFetch`, `WebSearch`, `TodoWrite`
- `max_turns = 50` unless explicitly overridden
- `ResultMessage` remains the SDK-native completion boundary
- if `python evaluation/self_eval.py` passes but no `ResultMessage` arrives, the adapter still records a protocol failure

Optional compatibility behavior:

- `workspace_stop_oracle = "submit_tool"` may re-enable the legacy
  `check_ready()` / `submit_result(...)` path for compatibility experiments
- the default formal execution path is `public_self_eval`, not `submit_tool`

## Runtime Layout

Executors should consume a manifest-driven runtime layout with defaults:

- `data_dir = "data"`
- `work_dir = "work"`
- `output_dir = "output"`
- `checkpoints_dir = "checkpoints"`
- `public_bundle_dir = "public_bundle"`

Expected environment variables for generated code:

- `MYEVOSKILL_RUNTIME_ROOT`
- `MYEVOSKILL_PUBLIC_BUNDLE`
- `MYEVOSKILL_WORK_DIR`
- `MYEVOSKILL_OUTPUT_DIR`
- `MYEVOSKILL_CHECKPOINT_DIR`
- `MYEVOSKILL_WORKSPACE` for backward compatibility

The public bundle must not be mutated in place. Executors may stage a fresh
runtime copy for each run, but the original compiled bundle stays immutable.

For workspace-agent runs, the staged runtime root should expose public content
in two read-only views:

- normal runtime-relative input paths such as `data/...`
- a debug mirror at `public_bundle/`

Default persistent run roots for live or manual execution are:

- `artifacts/workspaces/<task_id>/<run_id>/`
- `artifacts/logs/<task_id>/<run_id>/`

## Runtime Policy

Task specs may declare:

- `runtime_policy.model_timeout_seconds`
- `runtime_policy.execution_budget_seconds`

Effective timeout precedence is:

1. explicit session/model override
2. manifest `runtime_policy`
3. global safe defaults

Current safe defaults:

- `model_timeout_seconds = 240`
- `execution_budget_seconds = 900`

Timeout behavior requirements:

- `0` must not mean infinite runtime
- solver timeouts must terminate the whole spawned process group
- timeout records must keep partial stdout/stderr when available
- timeout metadata must include:
  - `timed_out`
  - `timeout_scope`
  - `effective_model_timeout_seconds`
  - `effective_execution_budget_seconds`

## Session And Model Config

`ExecutorSessionConfig` includes:

- runtime command / args
- workspace path
- timeout
- env vars
- tool policy
- provider-specific extras
- optional `ModelConfig`

`ModelConfig` includes:

- `provider_name`
- `base_url`
- `api_key_env`
- `model_name`
- `timeout`
- `max_tokens`
- `temperature`
- `extra_headers`

For the current live Claude SDK path:

- `provider_name = "claude-sdk"`
- `api_key_env = "MYEVOSKILL_CLAUDE_API_KEY"` by default
- if `MYEVOSKILL_CLAUDE_API_KEY` is unset, the resolver falls back to
  `CLAUDE_API_KEY`, then `ANTHROPIC_API_KEY`
- `model_name` may be supplied directly or through `MYEVOSKILL_CLAUDE_MODEL`
- the host should also export `ANTHROPIC_API_KEY` for Claude Code CLI

## Output Contract

Every adapter returns a `RunRecord` with:

- `run_id`
- `task_id`
- `provider`
- `provider_session_id`
- `model_provider`
- `model_name`
- `env_hash`
- `skills_active`
- `workspace_root`
- `artifacts_uri`
- `transcript_uri`
- `stdout`
- `stderr`
- `runtime_seconds`
- optional `proxy_feedback`
- optional `judge_result`
- metadata including provider-specific diagnostics such as:
  - `model_provider_kind`
  - `api_key_env`
  - `prompt_contract_version`
  - `command_history_summary`
  - `response_format`
  - `response_candidate_count`
  - `response_selected_source`
  - `parsed_response`
  - optional `response_parse_error`

## Constraints

- adapters only receive `public_bundle`
- adapters must not read `hidden_bundle`
- adapters should write results into a run workspace and return references, not
  mutate task assets in place
- if an optional provider dependency is unavailable, the adapter must fail
  clearly or degrade in a controlled way, depending on configuration
- `InspectBridgeAdapter` depends on `inspect-ai`, which requires Python >= 3.10
  in the dev environment
- `ClaudeWorkspaceAdapter` live SDK mode depends on both the Python
  `claude-agent-sdk` package and the external `claude` CLI
- executor tests must not rely on the host environment accidentally missing an
  optional dependency
