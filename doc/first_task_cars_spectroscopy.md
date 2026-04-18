# First Task: cars_spectroscopy

## Why This Task

`cars_spectroscopy` is the first real end-to-end target because it is small,
dependency-light, and has clear metrics.

## Public Bundle Policy

Keep public:

- `README_public.md`
- `data/raw_data.npz`
- `data/meta_data.json`
- `requirements.txt`
- starter scaffold

Preserve README notes on:

- method hints
- implementation guidance
- reference-output explanations
- public metric definitions

Remove from public bundle as files:

- `src/`
- `main.py`
- `notebooks/`
- `plan/`
- `data/ground_truth.npz`
- hidden evaluation assets

## Hidden Judge Metrics

The public output contract is semantic reconstruction. Executors write
`output/reconstruction.npz` with:

- `estimated_temperature_K`
- `reconstructed_spectrum`
- `nu_axis`

The hidden judge then computes:

- `ncc_vs_ref >= ncc_boundary`
- `nrmse_vs_ref <= nrmse_boundary`
- `temperature_error_K <= temperature_error_K_boundary`

## Implementation Order

1. compile policy
2. task manifest
3. compile-audit report fields
4. Claude SDK workspace executor plumbing
5. manifest-driven runtime layout with stable `data/`, `work/`, `output/`,
   `checkpoints/`
6. model-backed workspace agent run from runtime root while keeping a read-only
   `public_bundle/` mirror
7. proxy verifier
8. hidden judge
9. validation wiring
10. registry wiring

## Runtime Contract

The first task already follows the shared runtime convention:

- inputs staged under `data/`
- generated multi-file code written under `work/`
- evaluated outputs written under `output/reconstruction.npz`
- optional training state written under `checkpoints/`

This task remains the reference implementation for later task manifests, but it
should not require task-name checks inside the executor main loop.

## Persistent Run Defaults

`cars_spectroscopy` live and manual runs should default to:

- workspace:
  `artifacts/workspaces/cars_spectroscopy/<run_id>/`
- logs:
  `artifacts/logs/cars_spectroscopy/<run_id>/`

The task manifest currently fixes:

- `runtime_policy.model_timeout_seconds = 600`
- `runtime_policy.execution_budget_seconds = 240`

## Live Host Notes

The current reference live path for `cars_spectroscopy` is the Claude SDK
workspace executor.

Live prerequisites:

- `claude-agent-sdk`
- Claude Code CLI (`claude`)
- `MYEVOSKILL_CLAUDE_API_KEY`
- `ANTHROPIC_API_KEY`
- optional `MYEVOSKILL_CLAUDE_MODEL`

If the Linux server cannot install Claude Code CLI, it is acceptable to move
the live `cars_spectroscopy` run workflow to Windows while keeping the same
manifest, workspace layout, proxy flow, hidden judge, and log layout.
