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

The current runtime backend is `venv_pip`.

For each task, MyEvoSkill builds or reuses a cached runtime environment by:

1. running the control-plane Python with `-m venv`
2. running task-env Python with `-m pip install --upgrade pip setuptools wheel`
3. running task-env Python with `-m pip install -r requirements.txt`
4. recording `pip freeze`

The environment is considered ready only when:

- the task-env Python executable exists
- `install_report.json.success == true`

## EnvSpec And Hashing

`EnvSpec` is hashed into `env_hash` using a stable serialized description.

The task-facing fields that currently drive the hash are:

- `backend = "venv_pip"`
- host Python major/minor version
- normalized `requirements.txt` lines
- `task_id`
- `task_family`

Requirement normalization removes blank lines and comments while preserving the
remaining requirement order.

## Cache Layout

Shared runtime environment caches live under:

- `artifacts/env_cache/task_envs/<env_hash>/`

Each task environment cache writes:

- `env_spec.json`
- `requirements.txt`
- `install_report.json`
- `pip_freeze.txt`
- `build.log`
- `venv/`

`install_report.json` records whether the build succeeded, the commands run,
the resolved Python executable, and the paths to the log and freeze outputs.

Additional cache directories are still maintained for dataset, artifact, and
checkpoint reuse, but task runtime reuse is keyed by `env_hash`.

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

Confirmed registration eagerly builds or reuses the task runtime environment
from `tasks/<task_id>/requirements.txt`.

Registration embeds the resulting metadata into the manifest as `runtime_env`,
including:

- `backend`
- `env_hash`
- `requirements_path`
- `python_executable`
- `ready`
- `build_log_path`
- `install_report_path`

Live execution refuses to start unless:

- `judge_spec.ready == true`
- `runtime_env.ready == true`
- the confirmed registration contract is present and valid

If the registered environment is missing locally, the live runner may rebuild
it from the task root and then verify that the rebuilt `env_hash` matches the
manifest.

## Runtime Consistency

Task-related subprocesses use the manifest-declared task Python, not the
control-plane `sys.executable`.

That shared task runtime Python is used for:

- `python work/main.py`
- `python evaluation/self_eval.py`
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
