# Skill Registry Specification

## Candidate Sources

Candidates may only be distilled from train-split evidence.

PASS train runs become success episodes. FAIL/TIMEOUT train runs may
contribute gap evidence for missing algorithms and failure anti-patterns, but
they do not count as PASS episodes.

Skill synthesis may read selected train source through the audited
`DistillUniverse` boundary. It must never read valid-split source,
trajectories, workspaces, or data.

Candidates should capture:

- known failure modes,
- known bad triggers,
- helper scripts or references when they are reusable and task-agnostic.

## Registry States

- `draft`
  Legal, reusable, and train-derived, but not yet validated.
- `validated`
  Legal, reusable, and passed the selected validation gate.
- `rejected`
  Regressive, illegal, leaky, hardcoded, or not reusable.

## Permanent Storage Rule

A skill may be permanently stored only when:

- the source is legal,
- the skill is reusable,
- the sanitizer accepts the whole skill pack,
- the distillation audit contains no valid-split reads,
- the configured validation gate passes.

The default validation gate is skill-only: each selected valid task runs once
with the skill pack injected, and every selected task must PASS.

A train-complete but valid-failing candidate remains `draft`. For example, a
wave-optics skill that reaches train `6/6` but valid `0/4` is preserved as a
legal train-derived artifact, yet it is not a validated transfer result.

The older baseline-vs-skill promotion rule is still available as an explicit
comparison mode. In comparison mode, a candidate must avoid regressions and
rescue at least one previously failing valid task.

If a candidate causes any validation regression, it is rejected immediately.

## Upgrade Policy

- prefer editing an existing domain skill over creating a duplicate skill,
- upgrades must satisfy the same sanitizer and isolation rules,
- comparison-mode upgrades must also satisfy non-regression.
