# Dataset Split And Validation

## Split Policy

Each task family is assigned a fixed split manifest:

- `distill_train`
- `transfer_val`
- optional `final_test`

Rules:

- no task may appear in more than one split for the same family
- `distill_train` may be used for candidate generation and debugging
- `transfer_val` may be used only for validation/evaluation
- `transfer_val` must never be used to author or edit a skill

## Success Definition

A task is successful only when **all user metrics pass**.

No proxy score, mean score, or aggregate score may override this rule.

## Transfer Validation

The current default validation mode is skill-only:

1. Run each selected validation task once with the skill pack injected.
2. Require every selected validation task to PASS.
3. Preserve the validation report as evaluation evidence only; do not feed
   validation source, trajectory, data, or failures back into distillation.

If a candidate still gets `0/N` PASS on the selected valid set, it remains a
`draft` even when the train split has already reached full PASS coverage.

The older paired transfer comparison is still available explicitly through
`validate-skill --compare-baseline`. In comparison mode:

1. Run `baseline` on the validation set without the skill.
2. Run `treatment` on the same validation set with the skill.
3. Compute:
   - baseline success set `S0`
   - treatment success set `S1`

Comparison-mode decision policy:

- if `S0` is not a subset of `S1`: reject
- if `S0 = S1`: draft
- if `S0` is a strict subset of `S1`: promotable

The preferred comparison-mode case is at least one validation task moving
from failure to success with no regressions.
