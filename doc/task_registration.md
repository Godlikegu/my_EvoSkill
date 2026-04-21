# Task Registration

## Summary

MyEvoSkill now uses one canonical task contract per task:

- `tasks/<task_id>/evaluation/task_contract.json`

This file is the only authoritative interface definition for:

- public and private task files plus their semantics
- execution-facing public inputs
- output path, field names, dtypes, and exact shapes
- hidden metrics, metric goals, thresholds, and metric input bindings

The register step reads this contract, validates it, derives the public
contract, generates the ready judge adapter, and writes the manifest used by
live runs.

Legacy draft-era files such as `registration_input.json`,
`registration_contract.draft.json`, and `registration_contract.json` are not
part of the canonical flow for new tasks.

## Canonical Files

User-authored source of truth:

- `tasks/<task_id>/evaluation/task_contract.json`

Generated during registration:

- `tasks/<task_id>/evaluation/task_contract.public.json`
- `tasks/<task_id>/evaluation/judge_adapter.py`
- `tasks/<task_id>/evaluation/contract_generation.notes.json`
- `registry/tasks/<task_id>.json`
- `registry/tasks/<task_id>.notes.json`

## CLI Usage

Register a task from its canonical contract:

```bash
myevoskill-task-register --task-root ../tasks/conventional_ptychography --output-root .
```

Equivalent module forms:

```bash
python -m myevoskill.task_registration --task-root ../tasks/conventional_ptychography --output-root .
PYTHONPATH=src python -m myevoskill.task_registration --task-root ../tasks/conventional_ptychography --output-root .
```

Run a live test after registration:

```bash
python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
PYTHONPATH=src python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
```

If a task explicitly needs online retrieval during solver generation, live runs
may opt in with:

```bash
python -m myevoskill.task_live --task-id conventional_ptychography --project-root . --allow-network
```

This opt-in enables both workspace-level network access and Claude SDK web
tools for the Execution Agent.

## Contract Responsibilities

`task_contract.json` should define only the durable runtime contract:

- `files`
  Public and private resources, their roles, and optional field-level shape and
  dtype declarations.
- `execution`
  What the Execution Agent should read first, which public files are readable,
  the public entrypoint, and writable roots.
- `output`
  The one public output artifact path and the exact output field schema.
- `metrics`
  Metric names, goals, thresholds, helper interface, and explicit input
  bindings from output or private references.

The contract should not rely on weak global shape conventions such as
`same_shape_fields`. Each output field and each metric input binding must carry
its own exact shape expectation.

## Registration Rules

Registration is contract-first:

- registration fails if `task_contract.json` is missing
- registration fails if required fields, shapes, or helper bindings are invalid
- registration fails if `file_id`, output field, or helper references do not
  resolve
- registration fails if task-local referenced paths do not exist
- registration fails if metric input shape derivation does not match the
  declared `expected_shape`
- registration builds or reuses one task runtime environment from the
  task-local `requirements.txt`

If registration succeeds, the generated manifest stores only runtime handoff
information plus the canonical contract path. It does not embed a second copy
of the full interface schema.

## Public Bundle Rules

The compiled public bundle is derived from `task_contract.public.json`, not
from heuristics about filenames alone.

That means:

- only files declared `visibility: "public"` are staged for the Execution Agent
- private helper code such as `main.py`, `src/*.py`, hidden references, and
  metric helpers remain hidden unless the contract explicitly marks them public
- `task_contract.public.json` becomes the interface truth source visible to the
  Execution Agent

## Metric Execution Model

The platform resolves metrics in one uniform pipeline:

1. Load the metric definition from `task_contract.json`.
2. Resolve each metric input from output fields, private files, or literal
   values.
3. Apply declared selectors and preprocess steps.
4. Enforce exact `expected_shape` checks before metric execution.
5. Invoke the declared helper interface.
6. Coerce the result to one scalar float.
7. Compare it against `goal` plus `threshold`.

This guarantees that:

- the Execution Agent sees one explicit output schema
- the judge consumes inputs derived from that same schema
- self-eval and hidden judge fail for the same shape mismatch reasons

## Live Run Gating

Live execution is allowed only when all of the following are true:

- `registry/tasks/<task_id>.json` exists
- `manifest.ready == true`
- `manifest.runtime_env.ready == true`
- `manifest.task_contract_path` resolves to a valid `task_contract.json`
- the generated judge adapter is present and ready

If the runtime environment cache is missing locally, the live runner may
rebuild it from the registered `requirements.txt` and `env_hash`.

## Authoring Checklist

For a new task, the user should normally prepare:

- `README.md`
- `requirements.txt`
- public input files such as `data/raw_data.npz`
- public metadata files such as `data/meta_data.json`
- private reference files needed by judging
- `tasks/<task_id>/evaluation/task_contract.json`

The user does not need to hand-author:

- `task_contract.public.json`
- `judge_adapter.py`
- manifest files under `registry/tasks/`

Those are derived artifacts created by registration.
