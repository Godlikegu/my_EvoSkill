# Distillation And Feedback

## Distillation Inputs

Skill distillation only uses `distill_train` runs. Valid source runs include:

- hidden-judge passing runs
- repaired runs that eventually pass
- multiple runs with shared failure modes and reusable fixes

Invalid source runs include:

- task-specific hardcodes
- leakage-dependent traces
- runs that rely on hidden evaluators or washed implementations

## Surrogate Feedback

MyEvoSkill supports a surrogate feedback builder that summarizes:

- proxy verifier findings
- hidden judge failure tags
- repeated error patterns across tasks

The builder must not reveal hidden file paths or private references. Its role
is diagnostic, not authoritative.

## Distillation Output

Candidate skills should describe:

- applicability
- trigger hints
- reusable workflow steps
- known failure modes fixed
- known bad triggers

These become inputs to `SkillRegistry`.

