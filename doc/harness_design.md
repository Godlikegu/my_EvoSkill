# Harness design (clean rewrite)

This document describes the design of the **clean** Claude-Code harness that
lives under `MyEvoSkill/src/myevoskill/`.  It supersedes the legacy `executor.py`
and ad-hoc scripts that were quarantined on the `01_feedback_iteration` branch.

## 0. Current module layout

```
src/myevoskill/
├── __init__.py             re-exports the live API only
├── cli.py                  register-task | run-task | run-batch
├── registration.py         deterministic v2-only manifest builder (no LLM)
├── concurrency/pool.py     run_tasks_parallel (uses harness.sandbox)
├── task_env.py             cross-platform per-task venv setup
├── harness/                runner.py + hooks.py + plan_guard.py + prompts.py
│                           + trajectory.py + sandbox.py + process_reaper.py
├── judge/bridge.py         JudgeRunner (subprocess) + JudgeFeedback
├── workspace/              policy.py + bash_parser.py + builder.py + agent_spec.py
└── (judge-adapter API, frozen for tasks/*/evaluation/judge_adapter.py)
    models.py · judging.py · task_contract.py · task_runtime.py
    resource_probe.py
```

### `harness/sandbox.py` — per-run isolated `$HOME`

The Claude CLI persists conversation history, plan sidecars, MCP cache and
other state under the operator's real `~/.claude/`.  We never want that:

* it leaks state between tasks (and between concurrent runs of the same task);
* it bloats the user's disk with hundreds of MB per long task;
* it makes deterministic cleanup hard.

`sandbox.py` provides three primitives shared by both `harness/runner.py`
(in-process SDK) and `concurrency/pool.py` (subprocess child):

* `make_isolated_home(repo_root, task_id, run_id, sandbox_root=None)` —
  creates `artifacts/sandboxes/<task_id>/<run_id>/home`, wipes any
  pre-existing contents, then **whitelist-seeds** only
  `~/.claude/{settings.json, config.json}` from the real HOME so that
  3rd-party gateway URLs and the user's chosen model are preserved.
  Critically, `~/.claude/projects/` and `~/.claude/sessions/` are
  **never** copied — that's the leak vector for prior conversation
  history, and is covered by `tests/test_sandbox_isolation.py`.
* `env_overrides_for(home)` — returns the `HOME` / `USERPROFILE` /
  `PYTHONIOENCODING` env-var dict to hand to `ClaudeAgentOptions(env=...)`
  or `subprocess.run(env=...)`.
* `cleanup_isolated_home(home, keep=False)` — invoked from a `finally`
  block so a crashed agent or SDK exception still releases the sandbox;
  `--keep-sandbox` opts out for debugging.

The five "judge-adapter API" modules at the bottom of the tree are **frozen**:
they are imported by every `tasks/*/evaluation/judge_adapter.py` and may only
change in lock-step with all task adapters.  Everything else above is fair game
for refactoring.

## 0.1 Registration (`myevoskill.registration`)

`register-task --task-id <id>` is now a pure, deterministic transform; it does
not call any LLM and does not touch any task workspace.  Inputs are **only**:

* `tasks/<id>/evaluation/task_contract.json` (v2 schema; required)
* `tasks/<id>/evaluation/judge_adapter.py`   (must exist; not opened)

It produces `registry/tasks/<id>.json`, with:

* `public_policy.allow` / `public_policy.denylist` derived from
  `files[].visibility` in the contract — anything marked `private` (e.g.
  `data/ground_truth.npz`, `src/visualization.py`) lands on the denylist
  automatically, so the agent **cannot** even reference those paths.
* `runtime_layout.writable_paths` derived from `execution.writable_paths`.
* `primary_output_path` lifted from `output.path`.

If the contract is missing, the schema is wrong, or `judge_adapter.py` is
absent, registration **refuses** (`RegistrationError`).  This is the only way a
task gets registered, and it is what guarantees the harness's filesystem
boundary is consistent with the contract the task author signed.

The design directly answers the eight problems the user reported:

| # | Reported problem | Where it is fixed |
|---|------------------|-------------------|
| 1 | Agent breaks out of workspace via `cd` / `bash` | `workspace/policy.py` + `harness/hooks.py` (`make_pre_tool_use_hook`) |
| 2 | Judge ↔ agent feedback loop never really iterates | `harness/runner.py` round-loop, `judge/bridge.py` produces structured `JudgeFeedback` injected back as a user turn |
| 3 | System prompt does not adapt across rounds | `harness/prompts.py` (`first_round_user_prompt`, `iteration_user_prompt`) |
| 4 | Logs are noisy, hard to debug, can't be replayed | `harness/trajectory.py` -- single JSONL of `{role, kind, ...}` events |
| 5 | Code base is bloated and confusing | Single package layout under `myevoskill/{workspace,harness,judge,concurrency}` and one CLI entry point (`cli.py`) |
| 6 | Branch hygiene | `master` is the trunk; `03_clean_harness` is the active feature branch; legacy state archived as tags |
| 7 | Iteration context is lost | `harness/runner.py` reuses one `ClaudeSDKClient` for the whole task, with the **same conversation** across rounds |
| 8 | `plan.md` looks identical every round | `harness/plan_guard.py` (`PlanGuard`) requires a top-level `## Round N` plan before the first code action in each judge round |

## 1. Layered architecture

```
                              ┌───────────────────────────────────┐
        register-task ───────►│ workspace/builder.py              │
                              │   builds run/<task>/<ts>/work/... │
                              │   from registry/tasks/<id>.json   │
                              └────────────────┬──────────────────┘
                                               │ (manifest + agent_root)
                                               ▼
                              ┌───────────────────────────────────┐
                              │ workspace/policy.py               │
                              │   WorkspacePolicy: agent_root,    │
                              │   denylist subs, dangerous bash   │
                              └────────────────┬──────────────────┘
                                               ▼
              ┌───────────────────────────────────────────────────┐
              │ harness/runner.py                                 │
              │   one ClaudeSDKClient, multi-round                │
              │                                                   │
              │   round k:                                        │
              │     ┌─ inject prompt (first vs iteration) ──┐     │
              │     │  hooks.PreToolUse intercepts every    │     │
              │     │  tool call, blocks policy violations  │     │
              │     │  PlanGuard blocks code if this round  │     │
              │     │  has no authored plan                 │     │
              │     │  trajectory.py records every event    │     │
              │     └─► judge/bridge.py runs judge_adapter  │     │
              │         on agent_root + ground_truth, hidden│     │
              │         from the agent.  Returns Feedback.  │     │
              └───────────────────────────────────────────────────┘
                                               │
                          PASS / FAIL / INVALID + per-metric mask
                                               ▼
                                concurrency/pool.py for batch
```

## 2. Workspace policy (problem 1)

`WorkspacePolicy` is a frozen dataclass derived from the task manifest:

* `agent_root` – the only directory the agent may read or write.  The harness
  pins `cwd` here, and **every path-shaped string** in any tool call is
  resolved against it.
* `forbidden_substrings` – a denylist (`ground_truth`, `evaluation/`,
  `task_contract.json`, `judge_adapter.py`, …).  Importantly we **do not**
  block `/main.py` or `/src/`: the agent legitimately writes those inside its
  own workspace.  The workspace builder is responsible for **never copying**
  source-task implementation files into the workspace.
* `dangerous_bash_patterns` – regexes matched against `Bash` commands:
  `sudo`, `curl`, `wget`, `pip install`, `cd /`, `cd C:\`, `cat /etc/passwd`, …

These rules are evaluated by `harness/hooks.make_pre_tool_use_hook`, which is
registered as a SDK `PreToolUse` hook.  When a violation is found:

1. the hook returns `permissionDecision="deny"` with a **specific reason** so
   the agent sees *why* and can self-correct;
2. the violation is recorded as an `env_feedback` event in `trajectory.jsonl`
   so we can audit attempted breakouts later.

This intercepts the action **before** Claude executes it -- the hack never
actually runs.  It also means the agent cannot fail the task post-hoc just for
*trying* a forbidden command; the next turn it sees the deny reason and adapts.

## 3. Plan-before-code (problems 8 + 1)

`PlanGuard` (in `harness/plan_guard.py`) implements the rule:

> At the start of judge round N, before the first code-writing or code-running
> tool call, the agent must author a top-level `## Round N` section in
> `plan.md`.

The seed file is only:

```md
# Plan

```

There is deliberately no templated `## Round 1`, so seed-only content never
counts as a real plan.  Once the current round heading exists, the agent may
write, edit, and run code multiple times inside that same round.  A failed
judge result advances the harness to round N+1, where the first code action is
blocked until `## Round N+1` exists.  Indented examples such as
`    ## Round 1` do not count. Editing `plan.md` itself is always allowed.

## 4. Iterative judge feedback (problems 2, 3, 7)

* The harness keeps **one** `ClaudeSDKClient` instance for the whole task.
  Across rounds, the conversation is preserved -- the agent remembers what it
  tried, what failed, and what the judge said.
* Round 1 prompt (`prompts.first_round_user_prompt`):
  *"Read README.md, meta_data.json, and agent_task_spec.json. Write
  `## Round 1` in plan.md before the first code action. Run from the workspace
  root with `python work/main.py`. Do not access ground truth or evaluation/."*
* Round k>1 prompt (`prompts.iteration_user_prompt`) is composed from the
  judge's `JudgeFeedback`:
    * If `feedback_mode == "pass_fail"` (default), the agent is told only
      `"verdict: FAIL"` plus a generic instruction to revise plan and retry.
    * If `feedback_mode == "metric_status"`, the agent is told **which**
      metrics passed and which did not, but **never** the actual numbers or
      thresholds (so it can't grid-search the threshold).
* The judge runs on the **hidden** task source (`tasks/<id>/evaluation/`),
  in a separate Python process, with `agent_root` and `ground_truth_path`
  passed via CLI.  Its stdout/stderr never reach the agent.
* Agent-fixable output problems (`missing_output`, schema errors, NaN/Inf,
  etc.) are reported as `INVALID` and may enter another round. Harness or judge
  infrastructure failures (`judge_runtime_error`, `judge_unparsable`,
  `judge_timeout`, `missing_judge_adapter`) are classified as harness `ERROR`
  and stop the run instead of being fed back as an agent task.

This is the loop that the user described: "Judge as environment signal,
agent in same conversation iterates until success or budget".

## 5. Trajectory + minimal logs (problem 4)

`harness/trajectory.py` writes a single `trajectory.jsonl` per run with one
event per line, of kinds:

```
assistant_text   – model natural-language reply
assistant_think  – thinking blocks (only kept if agent emits any)
tool_call        – {tool_name, tool_input, tool_use_id}
tool_result      – {tool_use_id, output_truncated, is_error}
env_feedback     – policy denials, plan_guard denials, judge verdict
task_started     – Claude Code background task started
task_progress    – Claude Code background task progress/notification
task_completed   – Claude Code background task completed
task_failed      – Claude Code background task failed
task_stopped     – Claude Code background task stopped by cleanup
round_marker     – round started / judged / aborted
```

PreToolUse hook events are the authoritative source for `tool_call`, so the
runner does not duplicate `ToolUseBlock` events from assistant messages.  Raw
`assistant_think` is retained for operator debugging; distillation export should
filter thinking blocks and keep the clean sequence of assistant text, tool
calls/results, environment feedback, and task lifecycle events.

A short human-readable `run_summary.json` is also written (including
`plan_history` and `process_cleanup_errors`), plus per-round
`judge_round_NN.json` and a fresh per-round `plan_round_NN.md` snapshot of
`plan.md` (with an index in
`plan_history.jsonl`: `{round, timestamp, sha256, size_bytes, diff_lines,
note}`) so we can audit what the agent claimed to be trying each round even
after the workspace is wiped on PASS. Snapshots are taken right after the
agent finishes the round and before the judge runs, ensuring we capture the
plan state the judge actually evaluated against. Kept on disk for debugging
but never shown to
the agent).  Workspaces are deleted on success unless `--keep-workspace`.

## 6. Concurrency + cleanup

`concurrency/pool.py` runs each task as a **separate subprocess** of
`python -m myevoskill.cli run-task --json`.  Each child:

* gets its own workspace directory under `artifacts/workspaces/<task>/<ts>/`;
* gets its own `~/.claude` and `~/.config/claude` directories under the
  workspace, redirected via `CLAUDE_HOME` / `XDG_CONFIG_HOME` env vars, so
  conversation history is **never written to the user's real home** and is
  deleted with the workspace at end of run;
* prints exactly one JSON line at the very end which the parent collects.

This satisfies the "many tasks in parallel + no claude history left on the
user's machine" requirement.

Within a live Claude SDK run, `runner.py` also treats background Bash/Python
tasks as harness-owned resources. It tracks SDK task lifecycle messages, calls
`client.stop_task(task_id)` for active tasks at the end of each agent round,
and then runs a best-effort process reaper scoped to the Claude CLI process
tree. The reaper only targets Bash/Python descendants whose command line or
lineage mentions the current `run_id`, workspace path, or sandbox path, and it
kills leaf-first so child processes are not orphaned. If a matching process is
still alive before judge, the run is marked harness `ERROR` and judge is not
called, because output may still be changing.

## 7. Branch hygiene (problem 6)

* `master` -- trunk, contains the clean rewrite once it lands.
* `03_clean_harness` -- the active feature branch (this work).
* `01_feedback_iteration` -- previous dirty work, archived as the tag
  `archive/feedback-iteration-2026-04-25` and removed from active branches.
* All commits use Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`,
  `chore:`).

## 8. What the agent can and cannot do

| Action | Status | Enforced by |
|--------|--------|-------------|
| Read `README.md`, `meta_data.json`, `data/*` | allow | policy.is_inside |
| Write `work/*.py`, `output/*` | allow | policy.is_writable |
| Write `plan.md` | allow (always) | plan_guard exception |
| Read `tasks/<id>/main.py` (source solution) | DENY | path is outside agent_root |
| Read `tasks/<id>/evaluation/` | DENY | substring + outside agent_root |
| Read any file matching `ground_truth` | DENY | substring |
| `cd /`, `cd ..\..`, absolute paths in cat/grep/find | DENY | dangerous_bash regex |
| `curl`, `wget`, `pip install`, `sudo` | DENY | dangerous_bash regex |
| First code action in round N without `## Round N` in `plan.md` | DENY | PlanGuard |

All denials are visible to the agent as the SDK `permissionDecisionReason`
string, **without** revealing thresholds, ground truth contents, or judge
internals.

## 9. Three-layer defence against reading the reference solution

The agent is expected to discover its own method. The reference solution
(`tasks/<id>/{src/, notebooks/, plan/, main.py, *.ipynb}`) must therefore be
unreachable from inside the workspace.  Three independent layers guarantee
this:

1. **Registration-time hard rejection** (`registration.py`).
   Author-provided contracts cannot mark `src/`, `notebooks/`, `plan/`,
   `main.py` or any `*.ipynb` file as `public`. `_is_always_hidden` is
   consulted *before* the per-file `visibility` flag, so even an honest
   mistake in the contract is corrected automatically (and a warning is
   emitted to `runtime_logs/`).
2. **Build-time copy filter** (`workspace/builder.py`).
   The workspace builder only copies files that are still on the public
   allowlist after step (1). Hidden files are never materialised inside the
   workspace tree.
3. **Run-time substring guard** (`workspace/policy.py` →
   `harness/hooks.py`).
   `GLOBAL_FORBIDDEN_SUBSTRINGS` rejects any tool input whose path contains
   `/src/`, `/notebooks/`, `/plan/`, `.ipynb`, `ground_truth`,
   `evaluation/`, `judge_adapter.py`, etc., regardless of how the path was
   constructed.  This catches paths the agent fabricates (e.g.
   `../../tasks/<id>/src/foo.py`).

`main.py` is intentionally **not** in the run-time substring denylist:
the contract's own entrypoint is `work/main.py`, which the agent must be
free to create.  Reference `tasks/<id>/main.py` is unreachable anyway
because it sits outside the agent's workspace, and `policy.is_inside`
denies any path that escapes `agent_root`.

## 10. Agent-visible task spec (`workspace/agent_spec.py`)

The harness ships a **machine-readable** IO contract to the agent as
`agent_task_spec.json`, written into the workspace by `builder.py`. This
gives the agent enough structural information to produce a file the judge
will accept, **without** revealing any evaluation-side detail.

What the spec exposes:

* `inputs.files[]` -- public input files only (`input_data`, `metadata`,
  `runtime_dependencies`), with `dtype` + `shape` per array key when the
  contract declares them.
* `output.path`, `output.format`, and `output.required_keys[].(name|dtype|shape)`
  -- so the agent knows it must write
  `output/reconstruction.npz` containing e.g.
  `reconstruction (float32, shape 1x128x128)`.
* `runtime.entrypoint`, `runtime.writable_paths`, `runtime.wall_clock_seconds`.
* `evaluation_protocol`: only the *channel* (`pass_fail` or
  `metric_status`), explicitly noting that names/numbers/thresholds are
  hidden.
* `forbidden.paths` and `forbidden.actions`, mirrored from the policy.

What the spec **never** exposes (enforced by `assert_spec_has_no_leaks`):

* metric names (`ncc`, `nrmse`, …) or threshold numbers,
* `ground_truth*`, `judge_adapter*`, `evaluation/*`, `metric_helper*`,
* paths that point at the reference solution
  (`task_contract.json`, `registration_contract.json`,
  `reference_outputs*`).

`builder.build_workspace` calls `derive_agent_task_spec` then
`assert_spec_has_no_leaks` on every workspace build, so a future
contract refactor cannot accidentally smuggle a metric name into the
agent's view -- the build will fail loudly first. A one-paragraph
`render_summary(spec)` is also returned via `BuildResult.agent_task_spec_summary`,
which `harness/runner.py` inlines into the first-round user prompt
verbatim (the agent then has both the inline summary and the JSON file
on disk to consult).

Tests live in `tests/test_agent_spec.py`:

* the spec contains the expected output schema,
* the spec lists *only* `visibility=public` inputs,
* the spec contains no leak words (parametrised over metric names and
  numeric thresholds),
* the leak list does not over-match friendly tokens such as
  `judge_verdict` or `evaluation_protocol` (which appear in the spec by
  design).

## 11. First-round plan.md authoring

`PLAN_SEED` deliberately ships **without a templated `## Round 1` block**.
The agent is told (via the system prompt and the first-round user prompt)
that it must author a fresh `## Round 1` entry containing Hypothesis /
Change / Verification before the first code action in Round 1. After a
failed judge attempt, the feedback prompt starts the next round and requires
`## Round N` before that round's first code action. The guard is round-based,
not mtime-based: repeated edits/runs inside one judge round do not require
re-writing the plan.


## 9. Sandbox hardening (commit A: fix(policy))

The pre-tool-use hooks now refuse a strictly larger set of attempts. The
guarantees are layered so each tool the agent can call has its own check:

* **`is_inside`** uses ``Path.relative_to``, not ``str.startswith``. A
  sibling directory like ``<agent_root>_evil/foo`` is correctly rejected;
  ``<agent_root>/work/../../etc/passwd`` is rejected after non-strict
  resolution.
* **`is_writable_for_write`** is the policy half of the write
  permission. Any tool that mutates a file (``Write``, ``Edit``,
  ``MultiEdit``, ``NotebookEdit``, plus ``Bash`` write targets) must
  land under ``writable_subdirs`` (default: ``work/``, ``output/``).
  ``plan.md`` is the *one* writable file outside those subdirs and is
  gated by the plan-guard layer, not by the policy.
* **`workspace.bash_parser.parse_bash_writes`** statically extracts every
  read/write target out of a ``Bash`` command before it is allowed to
  run. It handles shell redirections (`>`, `>>`, `tee`, `tee -a`),
  `cp`/`mv`/`rm`/`mkdir`/`touch`/`chmod`/`ln` with their last positional,
  and `python -c "..."` / `python <<EOF ... EOF` heredocs which it
  AST-walks for ``open(<lit>, 'w')``, ``Path(<lit>).write_text``,
  ``os.remove/rename/replace``, ``shutil.copy*/move/rmtree``, and the
  ``np.save*`` / ``json.dump`` family. Any non-literal path, eval/exec,
  command substitution or process substitution is tagged ``dynamic`` and
  the entire ``Bash`` call is denied with an explanation telling the
  agent to use the explicit ``Read``/``Write`` tools instead.

The regression suite ``tests/test_sandbox_hardening.py`` covers five
concrete hack vectors (H1..H5): ground-truth substring, root-write,
``python -c`` smuggle, prefix-confusion sibling directory, command
substitution / eval. All denials are routed through
``trajectory.env_feedback`` so the agent observes them as a normal turn
(no silent kill).

## 12. Deployment / packaging

MyEvoSkill is intentionally **not** a pip-installable package. The
repository has no `pyproject.toml`; pytest is configured via
`pytest.ini` (`testpaths=tests`, `pythonpath=src`).

Two-tier environment:

- **Harness env (conda, name `evoskill`)** built from
  `environment.yml` by `scripts/setup_env.sh`. This env hosts the CLI,
  the SDK loop, the judge bridge, and the test suite. The CLI is run as
  `python -m myevoskill.cli ...` with
  `PYTHONPATH=$REPO_ROOT/src:$PYTHONPATH` exported; the
  `setup_env.sh`, `register_task.sh`, `run_task.sh`, and
  `run_smoke_three.sh` helpers all do this automatically.
- **Per-task venv** at `MyEvoSkill/.venvs/<task_id>/`, built by either
  `python -m myevoskill.cli setup-task-env --task-id <task_id>` or
  `scripts/setup_task_env.sh <task_id>` from
  `tasks/<task_id>/requirements.txt`. The build status is recorded in
  `runtime_logs/setup/<task_id>.json`. `register-task` reads that file
  and refuses to mark the manifest live-ready unless `ready=true` (you
  can override with `--no-task-env` for tasks with no extra deps).
  The runner pre-pends this venv's `bin/` (POSIX) or `Scripts/`
  (Windows) to `PATH` so the agent's `python` and the judge bridge's
  default `python_executable` both resolve to the task venv.

Three-step pipeline per task:

```bash
bash MyEvoSkill/scripts/setup_env.sh                 # once per checkout
python -m myevoskill.cli setup-task-env --task-id <task_id>
bash MyEvoSkill/scripts/setup_task_env.sh  <task_id> # once per task
bash MyEvoSkill/scripts/register_task.sh   <task_id>
bash MyEvoSkill/scripts/run_task.sh        <task_id>
```

Conda is **only** used for the `evoskill` harness env; tasks never see
conda. There are no editable installs of MyEvoSkill itself.
