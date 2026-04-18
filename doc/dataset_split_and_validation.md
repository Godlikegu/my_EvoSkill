# Dataset Split And Validation

## Split Policy

Each task family is assigned a fixed split manifest:

- `distill_train`
- `transfer_val`
- optional `final_test`

Rules:

- no task may appear in more than one split for the same family
- `distill_train` may be used for candidate generation and debugging
- `transfer_val` may only be used for paired baseline vs with-skill evaluation
- `transfer_val` must never be used to author or edit a skill

## Success Definition

A task is successful only when **all user metrics pass**.

No proxy score, mean score, or aggregate score may override this rule.

## Transfer Validation

For each candidate skill:

1. Run `baseline` on the validation set without the skill.
2. Run `treatment` on the same validation set with the skill.
3. Compute:
   - baseline success set `S0`
   - treatment success set `S1`

Decision policy:

- if `S0 ⊄ S1`: reject
- if `S0 = S1`: draft
- if `S0 ⊂ S1`: promotable

The preferred case is at least one validation task moving from failure to
success with no regressions.

