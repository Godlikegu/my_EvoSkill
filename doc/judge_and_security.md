# Judge And Security

## Hidden Judge Output Contract

The hidden judge returns only:

- `all_metrics_passed`
- `metrics_actual`
- `failed_metrics`
- `failure_tags`

It must not return:

- ground-truth paths
- hidden evaluator source
- reference arrays
- stack traces that reveal private paths

## Proxy Verifier Contract

The proxy verifier is separate from the hidden judge. It may report:

- file existence
- shape and dtype mismatches
- NaN / Inf detections
- runtime budget and elapsed time
- warnings about suspicious or non-physical outputs

It must not report:

- hidden thresholds not already public
- private references
- secret file paths
- washed implementation details

## Metric Evaluation

Each metric has:

- `name`
- `threshold`
- `operator`: `>=` or `<=`

`all_metrics_passed` is true only if every metric satisfies its operator.

For `cars_spectroscopy`, the executor writes a semantic artifact at
`output/reconstruction.npz` with:

- `estimated_temperature_K`
- `reconstructed_spectrum`
- `nu_axis`

The hidden judge then deterministically computes private metrics from that
semantic output plus hidden assets such as `data/ground_truth.npz`,
`data/raw_data.npz`, and private helper code under `src/`. The agent must not
output hidden metrics directly.

## Leakage Controls

- public bundles exclude washed implementations as files
- public bundles exclude task-author planning material such as `plan/`
- hidden bundles are not mounted into executor workspaces
- compile reports are safe to expose, but they must not print hidden file paths
  beyond relative task-local paths
- public README sanitization removes private paths and hidden evaluator internals
  but preserves user-provided implementation/reference-output explanations when
  they are part of the public task spec
