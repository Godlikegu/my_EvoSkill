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
./scripts/run_task_register.sh --task-root ../tasks/conventional_ptychography --output-root .
```

Compatibility forms:

```bash
PYTHONPATH=src python -m myevoskill.task_registration --task-root ../tasks/conventional_ptychography --output-root .
python -m myevoskill.task_registration --task-root ../tasks/conventional_ptychography --output-root .
myevoskill-task-register --task-root ../tasks/conventional_ptychography --output-root .
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
./scripts/run_task_live.sh --task-id conventional_ptychography --project-root .
```

Run the Inspect executor against an OpenAI-style endpoint:

```bash
./scripts/run_task_live.sh \
  --task-id conventional_ptychography \
  --project-root . \
  --executor inspect \
  --model-provider openai-compatible \
  --model-base-url https://your-endpoint.example/v1 \
  --model-api-key-env OPENAI_API_KEY \
  --model-name your-model
```

Compatibility forms:

```bash
PYTHONPATH=src python -m myevoskill.task_live --task-id conventional_ptychography --project-root .
myevoskill-task-live --task-id conventional_ptychography --project-root .
```

To explicitly allow the Execution Agent to use the network during a live run:

```bash
./scripts/run_task_live.sh --task-id conventional_ptychography --project-root . --allow-network
```

When this flag is enabled, MyEvoSkill allows both networked Bash/Python access
and Claude SDK web tools such as `WebSearch` and `WebFetch`.
The generated execution prompt also switches to a paper-first workflow: after
reading `README_public.md` and `task_contract.public.json`, the agent is told
to do one brief bounded external search before writing code, preferring papers,
project pages, and abstract pages first, and only then consulting official or
author implementations if more detail is needed.

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

## Dev Environment

Use a named system conda environment instead of the repo-local `.conda_env` workflow:

```bash
./scripts/create_dev_env.sh
conda activate myevoskill
./scripts/run_tests.sh -q
```

`pip install -e .` and installed console scripts are still supported for compatibility, but the
recommended workflow is `conda activate <env>` plus the bash wrappers under `scripts/`.
