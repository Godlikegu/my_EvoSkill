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

## Packaging Model

MyEvoSkill is intentionally **not** a pip-installable package: there is no
`pyproject.toml` and we never run `pip install -e .`. All entry points are
invoked as `python -m myevoskill.cli ...` with the source on `PYTHONPATH`.

Two-tier environment layout:

- **Harness env (conda, name `evoskill`)** — Python toolchain that runs the
  CLI, the SDK loop, the judge bridge, and the test suite. Provisioned
  from `environment.yml` by `scripts/setup_env.sh`.
- **Per-task venv (`MyEvoSkill/.venvs/<task_id>/`)** — isolated runtime
  used by the agent and the judge to execute the task's `main.py`,
  `judge_adapter.py`, etc. Provisioned by
  `python -m myevoskill.cli setup-task-env --task-id <task_id>` or the
  shell wrapper `scripts/setup_task_env.sh <task_id>` from each task's
  `tasks/<task_id>/requirements.txt`.

`pytest` is configured by `pytest.ini` (testpaths=`tests`,
pythonpath=`src`). The harness scripts export
`PYTHONPATH=$REPO_ROOT/src:$PYTHONPATH` automatically.

## Three-Step Pipeline

```bash
# 1. Build / refresh the harness env (once per checkout)
bash MyEvoSkill/scripts/setup_env.sh

# 2. Provision the task's runtime venv (once per task / per requirements change)
python -m myevoskill.cli setup-task-env --repo-root MyEvoSkill --task-id <task_id>
# Equivalent shell wrapper on POSIX/Git Bash:
bash MyEvoSkill/scripts/setup_task_env.sh <task_id>

# 3. Register the task manifest and judge adapter
bash MyEvoSkill/scripts/register_task.sh <task_id>

# 4. Run the task end-to-end
bash MyEvoSkill/scripts/run_task.sh <task_id>
```

The combined three-task smoke (`cars_spectroscopy`, `mri_grappa`,
`xray_tooth_gridrec`) is one command:

```bash
bash MyEvoSkill/scripts/run_smoke_three.sh
```

## Canonical Task Registration

New tasks enter MyEvoSkill through one canonical contract file:

- `tasks/<task_id>/evaluation/task_contract.json`

This file is the single source of truth for:

- task file semantics
- public files visible to the Execution Agent
- output path, field names, dtypes, and exact shapes
- metric goals, thresholds, helpers, and input bindings

Registration writes:

- `tasks/<task_id>/evaluation/task_contract.public.json`
- `tasks/<task_id>/evaluation/judge_adapter.py`
- `tasks/<task_id>/evaluation/contract_generation.notes.json`
- `registry/tasks/<task_id>.json`
- `registry/tasks/<task_id>.notes.json`

`register_task.sh` first runs `setup_task_env.sh`, then refuses to mark
the task live-ready unless the venv build succeeded
(`runtime_logs/setup/<task_id>.json: ready=true`). Pass `--no-task-env`
only for tasks with no extra dependencies, or when iterating on the
contract itself.

## Live Run

Once a task is registered, run it through the live runner:

```bash
bash MyEvoSkill/scripts/run_task.sh conventional_ptychography
```

Equivalent module form (with `PYTHONPATH` set, inside the `evoskill`
conda env):

```bash
python -m myevoskill.cli run-task --repo-root . --task-id conventional_ptychography
```

On Windows from the repository root, the direct module form is:

```powershell
$env:PYTHONPATH='src'
.\.conda_env\python.exe -m myevoskill.cli setup-task-env --repo-root . --task-id conventional_ptychography
.\.conda_env\python.exe -m myevoskill.cli register-task --repo-root . --task-id conventional_ptychography --force --require-task-env
.\.conda_env\python.exe -m myevoskill.cli run-task --repo-root . --task-id conventional_ptychography --max-rounds 5 --budget-seconds 7200
```

To explicitly allow the Execution Agent to use the network during a
live run, forward `--allow-network` through `run_task.sh`:

```bash
bash MyEvoSkill/scripts/run_task.sh conventional_ptychography --allow-network
```

When this flag is enabled, MyEvoSkill allows both networked Bash/Python
access and Claude SDK web tools such as `WebSearch` and `WebFetch`.
The generated execution prompt also switches to a paper-first workflow:
after reading `README_public.md` and `task_contract.public.json`, the
agent is told to do one brief bounded external search before writing
code, preferring papers, project pages, and abstract pages first, and
only then consulting official or author implementations if more detail
is needed.

Live execution is gated by:

- a registered manifest under `registry/tasks/`
- `manifest.ready == true`
- `manifest.runtime_env.ready == true`
- a valid `task_contract.json`
- a ready generated judge adapter

During each live run, the Claude SDK harness keeps a single multi-round
conversation, requires one authored `## Round N` plan before the first code
action in each judge round, feeds back only pass/fail by default, and cleans up
Claude Bash/Python child tasks before judging so stale background processes
cannot keep mutating `output/`.

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
- [doc/harness_design.md](doc/harness_design.md)
  Harness internals: workspace builder, hooks, runner, sandbox.
