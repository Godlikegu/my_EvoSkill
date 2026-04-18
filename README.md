# MyEvoSkill

MyEvoSkill is a doc-first scaffold for secure skill distillation and transfer
validation on scientific coding tasks.

The repository enforces:

- explicit `distill_train` and `transfer_val` splits
- hidden judging where a task passes only if all user metrics pass
- monotonic transfer validation: `success(no-skill) ⊆ success(with-skill)`
- promotion of permanent skills only when they improve validation coverage
  without regressions
- isolation between public task bundles and hidden reference implementations

See `doc/` for the design specifications that govern the implementation.

## Task Registration

New raw tasks now enter MyEvoSkill through a fixed two-step contract-driven
registration flow instead of hand-editing `registry/tasks/*.json`.

Step 1: provide `tasks/<task_id>/evaluation/registration_input.json`, then
generate the draft contract with the mandatory Claude registration agent

```bash
myevoskill-task-contract-draft --task-root ../tasks/conventional_ptychography --output-root .
```

If the first draft is structurally invalid, the harness feeds the validation
errors back to the registration agent and performs one automatic repair round.

Step 2: after the user confirms
`tasks/<task_id>/evaluation/registration_contract.json`, register it

```bash
myevoskill-task-register --task-root ../tasks/conventional_ptychography --output-root .
```

Module form after editable install:

```bash
python -m myevoskill.task_registration --task-root ../tasks/conventional_ptychography --output-root .
```

Repo-local module form without installation:

```bash
PYTHONPATH=src python -m myevoskill.task_registration --task-root ../tasks/conventional_ptychography --output-root .
```

The draft step writes:

- `tasks/<task_id>/evaluation/registration_input.json`
- `tasks/<task_id>/evaluation/registration_contract.draft.json`
- `tasks/<task_id>/evaluation/contract_generation.notes.json`

The register step writes:

- `registry/tasks/<task_id>.json`
- `registry/tasks/<task_id>.notes.json`
- `tasks/<task_id>/evaluation/judge_adapter.py`

The confirmed source of truth is always
`tasks/<task_id>/evaluation/registration_contract.json`.

`registration_input.json` is the registration-stage semantic input. It tells
the Claude registration agent which files are task description, public input,
public metadata, evaluation logic, hidden references, and which metrics plus
thresholds define success. User input keeps the simple `operator` plus
`threshold` form; the generated contract normalizes those into
`judge_contract.metrics[*].pass_condition = { operator, threshold }`. It does
not inline README or script bodies.

See [doc/task_registration.md](doc/task_registration.md) for the full
registration flow and manual follow-up checklist.

## Task Live Run

Registered tasks can be executed through one unified live-run entrypoint:

```bash
myevoskill-task-live --task-id conventional_ptychography --project-root .
```

Module form after editable install:

```bash
python -m myevoskill.live_runner --task-id conventional_ptychography --project-root .
```

Repo-local module form without installation:

```bash
PYTHONPATH=src python -m myevoskill.live_runner --task-id conventional_ptychography --project-root .
```

This interface is manifest-driven and now also gated by a confirmed contract
plus `judge_spec.ready=true`. New tasks should not require framework code
changes in order to run live; they should enter through registration and then
reuse the same live runner.
