# Domain Skill Distillation Pipeline

This pipeline turns train-split trajectories and train-only source evidence
into one reusable Anthropic skill pack for a domain. It is designed for the
wave-optics reconstruction workflow, but the split isolation and skill-pack
contracts are domain-agnostic.

## Current invariants

- One split/domain produces one domain skill, for example
  `wave-optics-recon-v1`. The trainer does not create per-task skills.
- Distillation may read train trajectories, train summaries, train workspaces,
  and selected train public/source files through `DistillUniverse`.
- Distillation must not read valid trajectories, valid source, valid
  workspaces, valid data, or valid task ids.
- The skill pack uses Anthropic's native format:
  `artifacts/skills/<skill-name>/SKILL.md`, with optional `scripts/`,
  `references/`, and `meta.json`.
- Runtime injection is native Claude skill injection. The harness copies the
  pack into `.claude/skills/<skill-name>/...`, allows the `Skill` tool, and
  pins an SDK agent with `skills=[<skill-name>]`. The skill body is not copied
  into the task prompt.
- Valid evaluation runs only after every train task in the split has at least
  one PASS run. By default validation is skill-only; baseline comparison is an
  explicit `--compare-baseline` mode.

## End-to-end flow

```text
scripts/train_domain_skill.py
  |
  |-- scan artifacts/logs/<model_slug>/<train_task>/run-*/summary
  |
  |-- epoch 1:
  |     run every train task once before first distillation
  |
  |-- later epochs:
  |     run only train tasks that still lack a PASS
  |
  |-- after each epoch:
  |     myevoskill.cli distill-skill
  |       - mine all current PASS train episodes
  |       - collect train-only gap evidence for FAIL/TIMEOUT tasks
  |       - read selected train source through DistillUniverse
  |       - synthesize one domain SKILL.md
  |       - write reusable helper scripts
  |       - sanitize the whole pack
  |       - verify the audit log has no valid reads
  |
  |-- when all train tasks PASS:
  |     re-distill the final all-train skill
  |
  |-- then, and only then:
        myevoskill.cli validate-skill --valid-task-ids <one-valid-task>
```

For strict training gates, pass `--require-all-train-pass`. If any train task
still lacks a PASS when the epoch budget is exhausted, the script exits nonzero
and skips valid evaluation.

## Evidence sources

`distill/episode_miner.py` parses only train runs. It accepts both
`trajectory.jsonl` and `trajectory.json`, and it accepts `run_summary.json`
with a `summary.json` fallback. Runs without a readable summary are skipped
instead of crashing the distillation job.

PASS train runs become `TaskEpisode` records. Each episode contains:

- scrubbed tool-use signatures,
- judge feedback tags,
- relevant file references observed in the trajectory,
- run metadata such as rounds and runtime,
- the public primary output path.

FAIL and TIMEOUT train runs do not become PASS episodes. They are collected as
train-only gap evidence by `collect_train_gap_evidence`, with:

- `failure_mode`,
- `agent_attempt` from recent trajectory and `plan.md` content,
- `source_hint` from selected train files such as README, specs, and solver
  source,
- `transferable_lesson`, a prompt-side instruction to extract generic
  algorithm gaps and anti-patterns.

This lets failed train tasks teach the skill about missing domain algorithms
without letting them masquerade as successful demonstrations.

## Split isolation

`distill/universe.py` is the only allowed read boundary for distillation. It
knows the split file's train and valid ids, confines paths to each train task
or train run directory, and writes an audit JSONL record for each allowed or
denied access.

At the end of `distill-skill`, `DistillUniverse.assert_no_valid_accesses()` is
called. Any audited valid access fails the command. The sanitizer also scans
the finished pack for literal valid task ids.

## Skill synthesis and assets

`distill/skill_synthesizer.py` builds the skill in two layers:

- Deterministic draft: merges all current PASS train episodes and train gap
  evidence into an Anthropic-format `SKILL.md`.
- Optional LLM polish: if `--model-id` is supplied and the configured
  Anthropic gateway is available, the same model can rewrite the draft using
  compacted train-only evidence. The polished body is sanitized again before
  it is written.

The synthesizer writes helper scripts into `scripts/`. These helpers are
domain-general assets, not task answers. Current wave-optics helpers include
array inspection, schema-shaped public-array baselines, FFT grid checks,
Stolt/f-k migration checks, SSNP/ODT grid checks, acoustic/wave CFL checks,
FWI budget guards, a CPML FWI helper, and small tomography baselines.

`distill/skill_sanitizer.py` scans `SKILL.md` and all text files in the pack.
It rejects:

- train or valid task id literals,
- metric thresholds,
- specific data file paths,
- hidden/judge file mentions,
- reference source path leaks,
- hardcoded array shapes,
- overlong `SKILL.md` or oversized packs.

## Current implementation status

The current implementation is already enforcing the intended domain-skill
training rhythm:

- one split produces one domain skill
- epoch 1 runs the full train list before the first distillation
- later epochs rerun only train tasks that still lack a PASS
- distillation uses current PASS train episodes plus train-only gap evidence
- valid evaluation does not run until every train task has at least one PASS

On the validation side:

- `validate-skill` defaults to skill-only validation
- `scripts/diagnose_valid_failures.py` can summarize valid outcomes into a
  diagnosis report
- valid source, valid trajectories, valid data, and valid failures remain
  evaluation-only evidence and do not flow back into distillation

## Runtime injection

`workspace/builder.py` copies a skill pack like:

```text
artifacts/skills/wave-optics-recon-v1/
  SKILL.md
  scripts/
  meta.json
```

into the run workspace as:

```text
<workspace>/.claude/skills/wave-optics-recon-v1/
  SKILL.md
  scripts/
  meta.json
```

`harness/runner.py` then:

- adds `Skill` to the allowed Claude tools,
- uses project setting sources so Claude can discover project skills,
- creates an SDK `AgentDefinition` named `myevoskill-domain`,
- pins `skills=["wave-optics-recon-v1"]`,
- passes `agent=myevoskill-domain` through SDK extra args.

`harness/prompts.py` only adds a short nudge telling the agent that a native
domain skill is pinned and should be loaded. It does not paste the skill body
or helper contents into the prompt.

## Main commands

Distill one domain skill from the current train evidence:

```powershell
$env:PYTHONPATH='src'
conda run -n evoskill python -m myevoskill.cli distill-skill `
  --repo-root . `
  --split registry\splits\wave_optics_v1.json `
  --skill-id wave_optics_recon_v1 `
  --out-root artifacts\skills `
  --model-id "Vendor2/Claude-4.6-opus"
```

Run the strict domain train gate:

```powershell
$env:PYTHONPATH='src'
conda run -n evoskill python scripts\train_domain_skill.py `
  --repo-root . `
  --split registry\splits\wave_optics_v1.json `
  --skill-id wave_optics_recon_v1 `
  --model-id "Vendor2/Claude-4.6-opus" `
  --max-epochs 8 `
  --max-rounds 5 `
  --budget-seconds 5400 `
  --valid-task-id reflection_ODT `
  --require-all-train-pass
```

Validate one valid task with the final skill, skill-only mode:

```powershell
conda run -n evoskill python -m myevoskill.cli validate-skill `
  --repo-root . `
  --split registry\splits\wave_optics_v1.json `
  --skill-pack-dir artifacts\skills\wave-optics-recon-v1 `
  --valid-task-ids reflection_ODT `
  --model-id "Vendor2/Claude-4.6-opus"
```

Run the older baseline-vs-skill transfer comparison explicitly:

```powershell
conda run -n evoskill python -m myevoskill.cli validate-skill `
  --repo-root . `
  --split registry\splits\wave_optics_v1.json `
  --skill-pack-dir artifacts\skills\wave-optics-recon-v1 `
  --valid-task-ids reflection_ODT `
  --model-id "Vendor2/Claude-4.6-opus" `
  --compare-baseline
```

## Tests

```powershell
$env:PYTHONPATH='src'
conda run -n evoskill python -m pytest `
  tests\distill `
  tests\test_train_domain_skill.py `
  tests\test_workspace_builder_paths.py `
  -q
```

The tests cover missing-summary skips, JSON/JSONL trajectories, gap evidence,
valid isolation, sanitizer rejection rules, helper script packaging,
Anthropic skill injection paths, one-valid-task validation, and the train gate
that skips valid until all train tasks PASS.

Additional operational scripts:

- `python scripts/check_train_status.py`
- `python scripts/diagnose_valid_failures.py`
- `python scripts/reproduce_wave_optics_train_gate.py`
- `python scripts/visualize_train_pass_results.py`
- `python scripts/visualize_valid_results.py`

## Current wave-optics status

The current local wave-optics training gate has produced a final skill pack at
`artifacts/skills/wave-optics-recon-v1/` and all six train tasks have PASS
evidence. Recent end-to-end evaluation still has no valid PASS, so the current
skill should be treated as a `draft` candidate rather than validated transfer.
In particular:

- train `6/6` is currently reproducible
- `n_plus_skill_pass >= 1` is still an unmet end-to-end transfer goal
- `artifacts/logs/_valid_runs/.../diagnosis.md` is a diagnosis artifact, not
  distillation evidence

Valid results should be treated as evaluation feedback only. They must not be
fed back into distillation evidence.
