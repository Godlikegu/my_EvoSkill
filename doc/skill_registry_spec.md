# Skill Registry Specification

## Candidate Sources

Candidates may only be distilled from `distill_train` runs that are:

- hidden-judge passing
- multi-task reusable
- free of evidence of leakage or hardcoding

Candidates should also capture:

- `known_failure_modes`
- `known_bad_triggers`
- optional `parent_skill_id` if the candidate edits an existing skill

## Registry States

- `draft`
  Non-regressive but no new transfer wins.
- `validated`
  Legal, reusable, and promotable with strict improvement.
- `rejected`
  Regressive, illegal, or not reusable.

## Permanent Storage Rule

A skill may be permanently stored only when:

- the source is legal
- the skill is reusable
- transfer validation satisfies `S0 ⊂ S1`
- no regression exists

If a candidate causes any validation regression, it is rejected immediately.

## Upgrade Policy

- prefer editing an existing skill over creating a duplicate skill
- upgrades must also satisfy non-regression
- if an upgraded version regresses any validation task that previously
  succeeded, the upgrade is rejected

