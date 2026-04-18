# Runtime And Env Cache

## Goal

Avoid rebuilding execution environments on every iteration while supporting
both classical scientific solvers and ML training workloads.

## Two Environment Layers

- `dev environment`
  A dedicated conda environment for MyEvoSkill itself.
- `task runtime environment`
  Derived from `EnvSpec -> env_hash` for each task.

The dev environment is separate from task environments and should not absorb
task-specific dependencies by default.

## Cache Layers

- `base image cache`
  Reused across families or identical runtime images.
- `task env cache`
  Reused for the exact `EnvSpec`.
- `dataset cache`
  Shared public input cache.
- `artifact cache`
  Shared non-secret output artifacts.
- `checkpoint cache`
  Persisted training checkpoints for long jobs.

## Hashing

`EnvSpec` is hashed into `env_hash` using the full serialized environment
description:

- Python version
- requirements
- system packages
- CUDA / GPU profile
- container image
- task family
- extra runtime metadata

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
