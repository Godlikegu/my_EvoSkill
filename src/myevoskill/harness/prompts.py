"""System & user prompt assembly for the harness.

Two contexts:

    * Round 1   - first attempt; no feedback yet.
    * Round N+1 - the agent has tried before; we splice the previous judge
                  result into a follow-up user message *in the same session*
                  so its previous reasoning stays in context.

We never tell the agent the metric *thresholds* nor the actual metric values.
The judge feedback is reduced to one of:

    * "PASS"      - all metrics satisfied; we stop iterating.
    * "FAIL"      - one or more metrics still below threshold; we ask it to
                    revise and try again.
    * "INVALID"   - the output file is missing / wrong schema / NaN; we ask it
                    to fix the output format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..judge.bridge import JudgeFeedback


SYSTEM_PROMPT = """You are an autonomous research engineer solving a computational-imaging task inside a sandboxed workspace.

Workspace contract
==================
* Your *only* working directory is the workspace root (the cwd the harness gives you).
* Read `agent_task_spec.json` first -- it is the *machine-readable* IO
  contract: input file list, primary output path, the exact keys / dtypes /
  shapes the judge expects, and your wall-clock budget. Then read `README.md`
  for the human-readable task description and `meta_data.json` for any
  task-specific physical parameters. Finally explore `data/`.
* Write code into `work/` (scratch) and produce the final result at the
  output path declared in `agent_task_spec.json:output.path` (also exposed
  as `meta_data.json:primary_output_path`). The `.npz` archive must contain
  every key listed under `output.required_keys` with the declared dtype
  and shape, otherwise the judge will return INVALID.
* The workspace is sandboxed: **anything outside the workspace is off-limits**.
  Filenames containing `ground_truth`, `evaluation/`, `task_contract.json`,
  `judge_adapter.py`, `src/`, `notebooks/`, or `plan/` are blocked
  by the harness regardless of where you encounter them. (You may freely
  create your own ``work/main.py`` -- only *reference* ``main.py`` files
  under ``tasks/<id>/`` are hidden from you.) Do not try to
  enumerate or guess at hidden files; they are deliberately removed.

Iteration discipline
====================
* Tool results, policy denials, plan-guard denials, and judge feedback are
  environment signals from the harness, not new user requests. The task request
  is already complete; do not wait for user clarification because an
  environment signal looks conversational or diagnostic.
* Maintain `plan.md` at the workspace root. At the start of each judge round,
  before that round's first code edit or code run, add a top-level
  `## Round <N>` block stating your Hypothesis, Change, and Verification.
  Once the current round's block exists, you may freely iterate on code and
  run checks within that same round.
* The harness will tell you when an attempt failed; you will *not* be given
  the actual metric values or thresholds. Your only feedback is PASS / FAIL
  (or INVALID, when your output cannot be parsed).
* On a FAIL or INVALID, reflect inside a new `## Round <N+1>` block before
  making the next round's code changes.

Coding style
============
* Use the Python interpreter the harness pins via `PATH`. Do not install
  packages (`pip install` / `conda install` are blocked).
* Run commands from the workspace root. Prefer `python work/main.py`; do not
  `cd work` and then reach back into `../data` or `../output`.
* Be deterministic: set seeds; write a single primary output file at the
  required path; print short progress lines so the trajectory is useful.
* Failing fast (an early `assert`) is preferred over silently wrong output.
"""


def initial_user_prompt(
    *,
    task_id: str,
    primary_output_rel: str,
    workspace_root: Path,
    budget_seconds: int,
    task_spec_summary: str = "",
) -> str:
    """First-round user message.

    ``task_spec_summary`` is the rendered short summary of
    ``agent_task_spec.json`` produced by
    :func:`myevoskill.workspace.agent_spec.render_summary`. We inline it
    so the agent has the full output schema in-context immediately,
    without having to read the spec file (it can still read the file for
    the authoritative version).
    """

    summary_block = ""
    if task_spec_summary:
        summary_block = "\n" + task_spec_summary.rstrip() + "\n"

    return (
        f"Task: **{task_id}**\n\n"
        f"Workspace: `{workspace_root}` (your cwd)\n"
        f"Primary output: `{primary_output_rel}`\n"
        f"Wall-clock budget: {budget_seconds} seconds total across all rounds.\n"
        + summary_block
        + "\nStep 1: read `agent_task_spec.json` for the machine-readable IO\n"
        "        contract, then `README.md` and `meta_data.json` for the\n"
        "        human description and task parameters. Then list `data/`\n"
        "        to confirm the inputs you actually have.\n"
        "Step 2: open `plan.md` and author a top-level `## Round 1` block\n"
        "        before your first code edit or code run. Include a one-line\n"
        "        summary, your Hypothesis about what the task needs, the concrete\n"
        "        Change you will make this round, and the Verification signal you\n"
        "        will inspect. The harness will refuse code edits and `python ...`\n"
        "        runs until `## Round 1` exists.\n"
        "Step 3: implement in `work/`, run, and produce the primary output\n"
        "        file at the path above with every required key. Always run\n"
        "        from the workspace root, e.g. `python work/main.py`; do not\n"
        "        `cd work`.\n"
        "When the output exists, reply with the single word `READY` so the\n"
        "judge can evaluate it."
    )


def feedback_user_prompt(
    *,
    round_index: int,
    feedback: JudgeFeedback,
    primary_output_rel: str,
    show_metric_status: bool = False,
) -> str:
    """Build the follow-up user message after a failed round.

    ``show_metric_status=False`` is the default and matches the
    "pass/fail only" mode the user asked for. We tell the agent *which*
    failure bucket it landed in (FAIL vs INVALID) and let it reflect.
    """

    head = (
        f"Previous round judgement: **{feedback.verdict}**.\n"
        f"You are now starting Round {round_index}.\n\n"
    )

    if feedback.verdict == "INVALID":
        body = (
            "The judge could not score your output. The most common causes are:\n"
            "  * the primary output file is missing or at the wrong path\n"
            f"    (expected `{primary_output_rel}`),\n"
            "  * the file is not a valid `.npz` archive,\n"
            "  * a required array key is missing or has the wrong shape/dtype,\n"
            "  * the array contains NaN/Inf values.\n\n"
            "Re-read `agent_task_spec.json:output` for the exact required keys"
            f" and shapes, then update `plan.md` with a new top-level"
            f" `## Round {round_index}` block, fix the output, and try again."
        )
    else:  # FAIL
        body = (
            "Your output was scored, but the judge returned FAIL. You will"
            " not be told the internal scoring details. Update"
            f" `plan.md` with a new top-level `## Round {round_index}` block,"
            " change your code accordingly, run it, and reply with `READY`"
            " when the new output is on disk."
        )

    if show_metric_status and feedback.metric_status:
        body += "\n\nPer-metric status (pass/fail only, no numbers):\n"
        for name, ok in feedback.metric_status.items():
            body += f"  * `{name}`: {'PASS' if ok else 'FAIL'}\n"

    return head + body
