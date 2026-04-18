# Task Bundle Specification

## Goal

Compile raw task assets into a public bundle for executors and a hidden bundle
for judges.

## Public Bundle

Allowed contents:

- `README_public.md`
- declared public data such as `data/raw_data.npz`
- declared public metadata such as `data/meta_data.json`
- `requirements.txt`
- optional starter scaffold created by MyEvoSkill
- optional public proxy checks
- optional output contract declarations

The public README preserves user-provided task specification by default,
including:

- task description
- method hints
- user-supplied implementation notes
- user-supplied reference output notes
- public metric definitions

Forbidden contents:

- `src/`
- `main.py`
- `notebooks/`
- `plan/`
- hidden judge code
- ground truth and private references unless explicitly marked public
- private file paths for hidden assets

`evaluation/reference_outputs/` is not copied as data, but README references to
those outputs may remain if they are part of the public task specification.

## Hidden Bundle

Contains the complete private evaluation context, including hidden references,
reference implementations, and judge assets.

## README Policy

README handling is policy-driven:

- preserve text by default
- remove only sections or lines that expose:
  - hidden judge source
  - private GT paths
  - washed implementation file paths
  - hidden asset locations
- allow task-level overrides through `READMEPolicy`

`READMEPolicy` includes:

- `preserve_sections`
- `remove_sections`
- `remove_path_patterns`
- `preserve_user_eval_notes`

## Compile Audit Layer

Task compilation is rule-first.

- rules are the only hard security boundary
- forbidden paths and denied assets are always blocked by rules
- an optional compile-audit adapter may inspect `README.md`, manifest metadata,
  and compile context to suggest:
  - suspicious README lines
  - public method hints
  - public output-contract hints
  - extra sanitization warnings

Compile-audit suggestions must never override rule barriers.

## Compiler Output

Each compiled task writes:

- `task.yaml`
- `compile_report.json`
- `public_bundle/`
- `hidden_bundle/`

`task.yaml` is the manifest-driven runtime handoff and should include:

- `task_id`
- `family`
- `public_policy`
- `runtime_layout`
- `runtime_policy`
- `output_contract`
- `proxy_spec`
- `judge_spec`

The compile report records:

- `copied_public`
- `blocked_paths`
- `rule_blocked_paths`
- `preserved_sections`
- `sanitized_sections`
- `public_data_allowlist`
- `public_data_denylist`
- `llm_audit_warnings`
- `llm_suggested_public_contract`
- `final_public_contract`
- `runtime_policy`

`runtime_layout` defaults to a stable per-run workspace shape:

- `data/`
- `work/`
- `output/`
- `checkpoints/`
- `public_bundle/`

Executors stage `public_bundle` files into the runtime root while also keeping
an explicit `public_bundle/` mirror for read-only inspection and debugging.

`runtime_policy` holds task-level default budgets such as:

- `model_timeout_seconds`
- `execution_budget_seconds`

Task-level `output_contract` may describe semantic artifacts instead of direct
metric files. For `cars_spectroscopy`, `output/reconstruction.npz` is public
only as a semantic reconstruction container and must include:

- `estimated_temperature_K`
- `reconstructed_spectrum`
- `nu_axis`
