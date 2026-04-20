# Architecture

## Summary

MyEvoSkill is a doc-first scaffold for secure skill distillation and transfer
validation on scientific coding tasks. The system is split into three planes:

- control plane
  Owns task manifests, dataset splits, validation policy, skill registry,
  logging, and documentation contracts.
- execution plane
  Runs provider-neutral executors through adapters.
- evaluation plane
  Separates low-leakage proxy verification from the hidden judge.

The end-to-end pipeline is:

`Task Registration -> TaskBundle -> Env Reuse -> Executor Run -> Proxy Verify -> Hidden Judge -> Skill Distillation -> Transfer Validation -> Skill Registry`

The first real end-to-end target is `cars_spectroscopy`.

## Control Plane

The control plane owns:

- task registration from raw `tasks/<task_id>/` directories into
  contract-driven registry artifacts
- fixed split manifests for `distill_train`, `transfer_val`, and optional
  `final_test`
- task compilation from raw assets into `public_bundle` and `hidden_bundle`
- rule-first compile auditing where optional LLM-style audit adapters can add
  warnings and public-contract suggestions without weakening rule barriers
- environment caching metadata and run workspace policies
- registry state transitions and versioned skill promotion
- logging and doc-sync enforcement

The control plane does not solve tasks directly. It orchestrates the inputs and
validates outputs from external executors.

## Task Registration

Task onboarding is contract-driven. Before a task can be compiled or scheduled
by batch runners, it should first produce a draft registration contract and
then register the confirmed contract.

Draft stage entrypoints:

- installed CLI: `myevoskill-task-contract-draft`
- module CLI: `python -m myevoskill.task_registration_draft`
- Python API: `draft_task_contract(...)`

Confirmed registration entrypoints:

- installed CLI: `myevoskill-task-register`
- module CLI: `python -m myevoskill.task_register`
- Python API: `register_task(...)`

Live-run entrypoints:

- installed CLI: `myevoskill-task-live`
- module CLI: `python -m myevoskill.task_live`

The draft stage is responsible for:

- reading `tasks/<task_id>/evaluation/registration_input.json`
- running the mandatory Claude registration agent to draft the contract
- retrying once with validation feedback when the first draft is structurally invalid
- generating `tasks/<task_id>/evaluation/registration_contract.draft.json`
- generating `tasks/<task_id>/evaluation/contract_generation.notes.json`

The confirmed registration stage is responsible for:

- validating `tasks/<task_id>/evaluation/registration_contract.json`
- validating task-local file references used by resources and judge metrics
- building or reusing one task runtime environment from `requirements.txt`
- generating `registry/tasks/<task_id>.json`
- generating `registry/tasks/<task_id>.notes.json`
- generating `tasks/<task_id>/evaluation/judge_adapter.py`

The confirmed
`tasks/<task_id>/evaluation/registration_contract.json` is the frozen
task-local source of truth. Metric pass/fail conditions are normalized into
`judge_contract.metrics[*].pass_condition = { operator, threshold }`. The
manifest, runtime-env metadata, and ready judge are derived artifacts.

## Execution Plane

Execution is provider-neutral and model-provider-neutral.

Key interfaces:

- `ExecutorAdapter`
- `ExecutorSessionConfig`
- `RunRecord`
- `ModelProviderAdapter`
- `ModelConfig`

Available executors:

- `LocalRunnerAdapter`
  Local subprocess runner and current guaranteed fallback path.
- `ClaudeWorkspaceAdapter`
  Primary Claude SDK workspace executor. It runs a Claude Code style agent
  inside the staged runtime root, lets the agent read public inputs with
  explicit tools, runs `python work/main.py`, and retries only from public
  self-check feedback.
- `InspectBridgeAdapter`
  Inspect-oriented bridge that can run placeholder mode without a model or
  single-shot model-backed code generation with a user API.
- `ClaudeAdapter`
  Compatibility wrapper over `ClaudeWorkspaceAdapter`.
- `OpenHandsAdapter`
  Placeholder for OpenHands runtime.

The preferred manual/live entrypoint is manifest-driven:

- installed CLI: `myevoskill-task-live`
- module CLI: `python -m myevoskill.task_live`

This keeps live execution task-agnostic. Adding a new task should not require a
new task-specific live script inside the framework.

Available model provider placeholders:

- `ClaudeSDKAdapter`
  Current formal execution provider.
- `OpenAICompatibleAdapter`
  Legacy compatibility surface only.
- `AnthropicCompatibleAdapter`
  Legacy compatibility surface only.
- `CustomHTTPAdapter`
  Legacy compatibility surface only.

The formal model-backed workspace path uses `provider_name="claude-sdk"`.
Raw OpenAI-compatible HTTP execution is no longer treated as a formal execution
pipeline for reference tasks. Future additions such as OpenHands or Inspect must
arrive as harness backends rather than bare chat-completions adapters.

The execution plane only receives `public_bundle` content. It must never see:

- washed implementations
- hidden judges
- `tasks/*/src`
- `tasks/*/main.py`
- `tasks/*/notebooks`
- raw hidden assets

README content provided by the user is treated as public specification by
default. Implementation notes or reference-output explanations stay public
unless they expose private file paths or hidden evaluator internals.

Execution remains split from evaluation:

- the execution agent generates and runs code
- the evaluation side stays deterministic through `ProxyVerifier` and
  `HiddenJudge`

For many-task support, executors must be manifest-driven rather than
task-name-driven. Task-specific differences should stay confined to:

- `public_policy`
- `output_contract`
- `proxy_spec`
- `judge_spec`

The executor main loop, workspace layout, logging format, and transfer
validation logic should remain shared across tasks.

Live execution is gated by manifest readiness:

- `judge_spec.ready == true`
- `runtime_env.ready == true`
- the confirmed registration contract exists and validates

The live runner may rebuild the registered task runtime environment when the
local cache is missing, but it must verify that the rebuilt `env_hash` matches
the manifest before execution continues.

Each run uses a standardized runtime workspace rooted at `workspace_root/`:

- `data/`
- `work/`
- `output/`
- `checkpoints/`
- a staged `public_bundle/` mirror

The workspace-agent path treats public inputs as read-only and only permits
writes under `work/`, `output/`, and `checkpoints/`.

The executor runs generated code with `cwd=workspace_root`, so agents can use
stable relative paths such as `data/raw_data.npz` and `output/result.npz`
across different scientific tasks.

For live and manual runs, the default persistent project-local layout is:

- `artifacts/workspaces/<task_id>/<run_id>/`
- `artifacts/logs/<task_id>/<run_id>/`

This keeps generated solvers, staged data, outputs, checkpoints, and logs
inside the repository tree without mixing different task/run instances.

Task-related subprocesses use the manifest-declared task runtime Python rather
than the control-plane interpreter. That shared task runtime is used for
workspace execution, public self-eval, post-run checks, and the task-local
judge subprocess.

The Claude SDK live path also depends on host tooling:

- `claude-agent-sdk` in the Python environment
- Claude Code CLI available as `claude`
- native Claude credentials exported in the shell

Because of that dependency, default Linux regression work and live Claude SDK
task execution may happen on different hosts. When the Linux server cannot
install Claude Code CLI, moving the live run workflow to Windows is an accepted
deployment pattern.

Timeouts are manifest-driven by default. Executors resolve effective values in
this order:

1. explicit session/model overrides
2. task manifest `runtime_policy`
3. global safe defaults

The current global safe defaults are:

- `model_timeout_seconds = 240`
- `execution_budget_seconds = 900`

`0` is not treated as "run forever". Missing or zero values fall back to the
resolved safe default.

## Evaluation Plane

The evaluation plane is split in two:

- `ProxyVerifier`
  Returns only low-leakage signals such as file existence, shape, dtype,
  runtime, NaN/Inf checks, and simple public-baseline comparisons.
- `HiddenJudge`
  Evaluates the full private benchmark and returns only:
  - `all_metrics_passed`
  - `metrics_actual`
  - `failed_metrics`
  - `failure_tags`

For `cars_spectroscopy`, success requires all of:

- `ncc_vs_ref >= ncc_boundary`
- `nrmse_vs_ref <= nrmse_boundary`
- `temperature_error_K <= temperature_error_K_boundary`

Those metrics are not public output fields. The execution agent writes only a
semantic reconstruction artifact, and the hidden judge computes the private
metrics deterministically from hidden assets and helper code.

The hidden judge is the only component that decides task success. A task
passes only when every user metric passes.

## Skill Lifecycle

Skills are not snapshots of one successful implementation. They are distilled
from multiple successful or repaired runs, with an emphasis on reusable
capabilities:

- workflow structure
- diagnostics
- solver-selection heuristics
- training and checkpoint routines
- scientific debugging strategies

Candidate sources must be:

- legal
- reusable
- non-cheating
- sourced from `distill_train`

Promotion on `transfer_val` uses:

- `S0 = success(no-skill)`
- `S1 = success(with-skill)`

Rules:

- if `S0 ⊄ S1`: reject
- if `S0 = S1`: draft
- if `S0 ⊂ S1`: validated

Permanent registry storage requires the `validated` case.
