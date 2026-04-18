# Doc Sync Policy

## Core Rule

Design docs are the source of truth. Before implementing a behavior-changing
change:

1. read the corresponding spec
2. update the spec if the design changes
3. update code
4. update tests

## Required Coverage

At minimum, these module groups must map to docs:

- compiler -> `task_bundle_spec.md`
- envs -> `runtime_and_env_cache.md`
- executor / proxy -> `executor_adapter_spec.md`
- judging / validation -> `judge_and_security.md`,
  `dataset_split_and_validation.md`
- registry / feedback -> `skill_registry_spec.md`,
  `distillation_and_feedback.md`
- logging / test policy -> `testing_and_logging.md`
- model provider -> `model_provider_spec.md`
- dev environment -> `dev_environment.md`
- first real task -> `first_task_cars_spectroscopy.md`

## Enforcement

The first implementation uses a `DocSyncChecker` that reports:

- missing required docs
- missing module-to-doc mapping

It is acceptable for the checker to be conservative and file-presence-based in
v1. Later it can inspect diffs.
