# MyEvoSkill

MyEvoSkill is a doc-first scaffold for secure skill distillation and transfer
validation on scientific coding tasks.

The repository enforces:

- explicit `distill_train` and `transfer_val` splits
- hidden judging where a task passes only if all user metrics pass
- monotonic transfer validation: `success(no-skill) <= success(with-skill)`
- promotion of permanent skills only when they improve validation coverage
  without regressions
- isolation between public task bundles and hidden reference implementations

See `doc/` for the design specifications that govern the implementation.

## Canonical Task Registration

New tasks now enter MyEvoSkill through one canonical contract file:

- `tasks/<task_id>/evaluation/task_contract.json`

This file is the single source of truth for:

- task file semantics
- public files visible to the Execution Agent
- output path, field names, dtypes, and exact shapes
- metric goals, thresholds, helpers, and input bindings

Register a task from its canonical contract:

```bash
myevoskill-task-register --task-root ../tasks/conventional_ptychography --output-root .
```

Equivalent module forms:

```bash
python -m myevoskill.task_registration --task-root ../tasks/conventional_ptychography --output-root .
PYTHONPATH=src python -m myevoskill.task_registration --task-root ../tasks/conventional_ptychography --output-root .
```

Registration writes:

- `tasks/<task_id>/evaluation/task_contract.public.json`
- `tasks/<task_id>/evaluation/judge_adapter.py`
- `tasks/<task_id>/evaluation/contract_generation.notes.json`
- `registry/tasks/<task_id>.json`
- `registry/tasks/<task_id>.notes.json`

Registration also builds or reuses one task runtime environment from the
task-local `requirements.txt`. If environment setup fails, the task is not
marked live-ready.

## Live Run

Run a registered task through the unified live runner:

```bash
myevoskill-task-live --task-id conventional_ptychography --project-root .
```

Equivalent module forms:

```bash
python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
PYTHONPATH=src python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
```

Live execution is gated by:

- a registered manifest under `registry/tasks/`
- `manifest.ready == true`
- `manifest.runtime_env.ready == true`
- a valid `task_contract.json`
- a ready generated judge adapter

## Public Bundle Rules

The public bundle is derived from `task_contract.public.json`, not from
filename heuristics alone.

That means:

- only files declared public in the contract are exposed to the Execution Agent
- private implementation files such as `main.py` and `src/*.py` stay hidden
  unless the contract explicitly marks them public
- hidden references and metric helpers remain private

## Documentation

- [doc/task_registration.md](doc/task_registration.md)
  Canonical registration flow and contract responsibilities.
- [doc/task_bundle_spec.md](doc/task_bundle_spec.md)
  Public-bundle derivation and visibility rules.
- [doc/runtime_and_env_cache.md](doc/runtime_and_env_cache.md)
  Runtime environment build and cache behavior.
- [doc/judge_and_security.md](doc/judge_and_security.md)
  Judge model, hidden metrics, and isolation rules.
