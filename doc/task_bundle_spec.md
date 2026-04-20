# Task Bundle Specification

## Goal

Compile a raw task directory into:

- a public bundle for the Execution Agent
- a hidden bundle for judging and replay

The public bundle must be derived from the canonical public contract, not from
ad hoc filename heuristics.

## Canonical Inputs

Bundle compilation consumes:

- `registry/tasks/<task_id>.json`
- `tasks/<task_id>/evaluation/task_contract.public.json`
- task-level public policy used for README sanitization

`task_contract.public.json` is the authoritative source for which task files
are visible to the Execution Agent.

## Public Bundle

Typical contents:

- `README_public.md`
- `task_contract.public.json`
- declared public data such as `data/raw_data.npz`
- declared public metadata such as `data/meta_data.json`
- `requirements.txt`
- generated public self-eval files when live execution needs them

Public bundle rules:

- only files declared public in `task_contract.public.json` are copied
- files not declared public are excluded by default
- private implementation files such as `main.py` or `src/*.py` stay hidden
  unless the contract explicitly marks them public
- private references, hidden metric helpers, and hidden judge assets are never
  copied to the public bundle

## Hidden Bundle

The hidden bundle contains the full task directory needed for judging and
replay, including:

- private reference data
- hidden metric helper code
- ready judge adapter
- reference implementations and internal scripts

## README Policy

README handling is still policy-driven:

- preserve public problem statements and method hints by default
- strip references to hidden files or hidden paths
- allow task-level overrides through `READMEPolicy`

`READMEPolicy` controls:

- `preserve_sections`
- `remove_sections`
- `remove_path_patterns`
- `preserve_user_eval_notes`

## Compiler Output

Each compiled task writes:

- `task.yaml`
- `compile_report.json`
- `public_bundle/`
- `hidden_bundle/`

`task.yaml` is the runtime handoff file and should contain only runtime-facing
metadata, such as:

- `task_id`
- `family`
- `runtime_layout`
- `runtime_policy`
- `runtime_env`
- `primary_output_path`
- `task_contract_path`
- `task_contract_public_path`
- `judge_adapter_path`
- `ready`

It should not embed a duplicate full contract payload.

## Compile Report

The compile report records:

- `copied_public`
- `blocked_paths`
- `rule_blocked_paths`
- `preserved_sections`
- `sanitized_sections`
- `public_data_allowlist`
- `public_data_denylist`
- `llm_audit_warnings`
- `final_public_contract`
- `runtime_layout`
- `runtime_policy`

## Runtime Layout

The runtime workspace defaults to:

- `data/`
- `work/`
- `output/`
- `checkpoints/`
- `public_bundle/`

Executors stage the public bundle into the runtime root and also keep a
read-only `public_bundle/` mirror for inspection and debugging.

## Security Boundary

Compilation is rule-first:

- the contract decides visibility
- compiler rules enforce the visibility boundary
- compile-audit suggestions may add warnings, but they must not widen exposure
  beyond the contract

In short: if a file is not public in the contract, it should not appear in the
Execution Agent workspace.
