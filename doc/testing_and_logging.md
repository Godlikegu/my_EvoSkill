# Testing And Logging

## Required Test Coverage

Unit tests must cover:

- split overlap detection
- no validation leakage into distillation
- all-metrics-pass success semantics
- transfer validation subset enforcement
- rejection on regression
- registry promotion and draft behavior
- task compiler leakage controls
- environment hash reuse and workspace reset
- provider-neutral executor adapter behavior
- optional executor dependency state machines without relying on ambient package
  installation state
- proxy verifier low-leakage output
- doc-sync policy coverage
- dev environment config files
- model provider config behavior
- environment-variable based API key resolution
- compile-audit suggestions that do not override hard rule barriers
- structured executor response parsing for JSON and plain-text fallbacks
- semantic output schema validation for hidden-judge task adapters
- workspace-agent path validation for read-only public files and writable
  `work/` / `output/` / `checkpoints/`
- harness-generated `evaluation/self_eval.py` creation and execution
- Claude SDK protocol handling when `self_eval.py` passes but no final
  `ResultMessage` arrives
- manifest-driven workspace prompt rendering without hidden-judge optimization text

Integration tests must cover:

- a candidate that improves validation and becomes `validated`
- a candidate that regresses and becomes `rejected`
- a candidate that ties and remains `draft`
- first-task compile -> local run -> proxy -> hidden judge -> validation path
- inspect-bridge fallback participation in a minimal pipeline when dependency
  availability is forced off
- real `cars_spectroscopy` manifest compile and model-backed workspace-agent flow
- failure when any required metric is not passed
- `cars_spectroscopy` semantic reconstruction judged against hidden metrics
- a live smoke path marked `external_network` for explicit real-API execution
- a structured JSON mock response that drives the workspace-agent loop
- runtime-root staging where relative `data/...` access succeeds
- rerun cleanup where stale files in `work/` and `output/` do not leak into the
  next run
- at least two manifests sharing the same executor main flow without executor
  branching on task name

## Logging

Each run writes a dedicated log directory containing:

- `summary.json`
- `stdout.log`
- `stderr.log`
- `executor_config.json`
- `model_config.json`
- `raw_response.txt`
- `parsed_response.json` or a parse-failure artifact for model-backed runs
- round-level workspace artifacts such as:
  - `agent_prompt_round_N.txt`
  - `agent_summary_round_N.json`
  - `files_written_round_N.json`
  - `commands_run_round_N.json`
  - `public_self_eval_round_N.json`
  - `public_self_eval_stdout_round_N.log`
  - `public_self_eval_stderr_round_N.log`
  - `sdk_messages_round_N.json`
  - `stdout_round_N.log`
  - `stderr_round_N.log`
  - `claude_sdk_diagnostics_round_N.json` when SDK summary parsing or timeout diagnostics are relevant
  - runtime evaluator assets:
    - `evaluation/self_eval.py`
    - `evaluation/self_eval_spec.json`
- trajectory artifacts for every run result, including success, timeout, `max_turns`,
  permission failures, and judge failures:
  - `trajectory_native.jsonl`
  - `trajectory_normalized.json`
  - `trajectory_summary.json`
  - `trajectory_redaction_report.json`
  - `vendor_session_ref.json`
- optional `artifacts_manifest.json`
- optional `proxy_feedback.json`
- optional `judge_result.json`

Persistent live and manual runs should write under:

- `artifacts/workspaces/<task_id>/<run_id>/`
- `artifacts/logs/<task_id>/<run_id>/`

Validation summaries must preserve:

- baseline success set
- with-skill success set
- newly successful tasks
- regressions
- final decision reason

Timeout failures must additionally preserve:

- `summary.json` with timeout scope
- `stderr.log` with timeout message
- effective timeout metadata in executor artifacts
- Claude SDK protocol diagnostics, including whether `python evaluation/self_eval.py`
  appeared in trace before the failure

## Testing Policy Notes

- Executor tests must be deterministic across developer machines and CI.
- Do not encode assumptions such as "`inspect_ai` is not installed here".
- For optional dependencies, tests should force the desired branch through
  constructor injection or monkeypatching.
- Model-backed tests should use mock LLM responses unless they are explicitly
  marked as external-network integrations.
- External-network tests must not run in the default `pytest -q tests` path.
- External-network tests should `skip` when required credentials are absent.
- The live Claude SDK smoke test additionally depends on host availability of
  Claude Code CLI, so a skipped live test on Linux does not imply a broken
  default regression environment.
- When Linux cannot provide Claude Code CLI, the supported fallback is to run
  the live Claude SDK smoke test on Windows with the same repository and
  project-local `artifacts/` layout.

## Live Claude SDK Notes

Current live workspace-agent smoke prerequisites:

- `claude-agent-sdk` installed in the active Python environment
- Claude Code CLI installed and available as `claude`
- one of:
  - `MYEVOSKILL_CLAUDE_API_KEY`
  - `CLAUDE_API_KEY`
  - `ANTHROPIC_API_KEY`
- optional `MYEVOSKILL_CLAUDE_MODEL`

Recommended command:

```bash
conda activate myevoskill
./scripts/run_tests.sh -q -m external_network tests/test_integration_pipeline.py -k real_manifest_live
```
