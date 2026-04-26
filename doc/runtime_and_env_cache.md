# Runtime And Env Cache

## Goal

Avoid rebuilding task execution environments on every iteration while keeping
registration, live execution, post-run checks, and judging on the same task
runtime Python.

## Two Environment Layers

- `dev environment`
  A dedicated conda environment for MyEvoSkill itself.
- `task runtime environment`
  Derived from `EnvSpec -> env_hash` for each task and built with
  `venv + pip`.

The dev environment is separate from task environments and should not absorb
task-specific dependencies by default.

## Current Backend

The current live backend is `per_task_venv`.

For each task, MyEvoSkill builds or refreshes a task-local virtual environment
under `.venvs/<task_id>` by:

1. running the control-plane Python with `-m venv`
2. running task-env Python with `-m pip install -r requirements.txt` when the
   task has a requirements file
3. writing `runtime_logs/setup/<task_id>.json`

The environment is considered ready only when:

- the task-env Python executable exists
- `runtime_logs/setup/<task_id>.json` reports `ready == true`

## Setup State

The task-facing state records:

- `backend = "per_task_venv"` in the registry manifest
- resolved `python_executable`
- `requirements_path`
- `requirements_sha256`
- setup state path

The requirements hash is used as an audit/debug signal for now; registration
does not rebuild the environment and does not silently repair missing setup
state when `--require-task-env` is used.

## Cache Layout

Task runtime environments live under:

- `.venvs/<task_id>/`

Setup state and pip logs live under:

- `runtime_logs/setup/<task_id>.json`
- `runtime_logs/setup/<task_id>.pip.log`

Additional cache directories may still be maintained for dataset, artifact, and
checkpoint reuse, but the live task Python runtime is the explicit per-task
venv recorded in setup state.

## Workspace Policy

Each run has a runtime workspace containing:

- `data/`
- `work/`
- `output/`
- optional `checkpoints/`
- a staged `public_bundle/` mirror

The executor runs with `cwd` fixed to the runtime root. Public task inputs are
staged into stable relative paths so solver code can read `data/...` and write
`output/...` without per-task path guessing.

For persistent live and manual runs, the runtime root should be resolved under:

- `artifacts/workspaces/<task_id>/<run_id>/`

The matching log root should be:

- `artifacts/logs/<task_id>/<run_id>/`

On rerun:

- staged runtime files are refreshed from the compiled public bundle
- `work/` and `output/` are reset
- environment caches are preserved
- checkpoint caches may be restored into the run workspace

This ensures iteration does not reconfigure the environment from scratch.

## Register And Live Integration

Task environment setup is an explicit step, not a side effect of registration:

```bash
python -m myevoskill.cli setup-task-env --repo-root . --task-id <task_id>
python -m myevoskill.cli register-task --repo-root . --task-id <task_id> --require-task-env
```

Registration embeds the resulting metadata into the manifest as `runtime_env`,
including:

- `backend`
- `requirements_path`
- `python_executable`
- `ready`
- `requirements_sha256`
- `setup_state_path`

Live execution refuses to start unless:

- `judge_spec.ready == true`
- `runtime_env.ready == true`
- the confirmed registration contract is present and valid

If the registered environment is missing locally, rerun `setup-task-env` and
then `register-task --require-task-env`. The live runner does not silently
create the environment.

## Runtime Consistency

Task-related subprocesses use the manifest-declared task Python, not the
control-plane `sys.executable`.

That shared task runtime Python is used for:

- `python work/main.py`
- post-run audit and replay
- task-local judge subprocess execution

The executor also places the task environment Python directory at the front of
`PATH` and records the resolved `task_python_executable` in run metadata and
transcripts.

## Runtime Policy

Each task manifest may declare:

- `runtime_policy.model_timeout_seconds`
- `runtime_policy.execution_budget_seconds`

Resolved timeout priority is:

1. explicit override in `ExecutorSessionConfig` or `ModelConfig`
2. task manifest `runtime_policy`
3. global safe defaults

Long-running optimization or ML training tasks should raise their execution
budget through manifest policy or explicit session override rather than relying
on an implicit no-timeout behavior.

## Dev Environment Policy

The repository ships:

- `environment.yml`
- `scripts/create_dev_env.sh`
- `scripts/print_env_info.py`

The recommended dev environment name is `myevoskill`, but a repo-local prefix
is also acceptable for isolated setup.

For the current integration plan, the dev environment should use Python 3.10+
so that optional `inspect-ai` installation works.
