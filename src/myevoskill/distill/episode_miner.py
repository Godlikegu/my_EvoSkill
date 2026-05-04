"""Mine reusable "episodes" from passing trajectories of train tasks.

An *episode* is a self-contained, deterministic record describing one
problem-solving move that the agent made successfully. Episodes are the
input to ``skill_synthesizer.synthesize_skill``; they are *not* shown to
the runtime agent directly (skills are).

Design constraints
------------------

1. **No valid leakage.** All file reads go through ``DistillUniverse``.
   The miner never opens a path on its own.
2. **Read-only.** The miner produces dataclasses; it does not mutate the
   trajectory or workspace.
3. **Content-grounded, not hallucinated.** Each episode field is either
   copied verbatim from a trajectory event or computed from a small
   deterministic feature extractor (regex / counter). Nothing here calls
   an LLM. The LLM step lives in ``skill_synthesizer``.
4. **Privacy.** We deliberately drop:
     * absolute paths beneath the run sandbox,
     * model-provider tokens / api keys,
     * anything that mentions a valid-split task id.
   The :func:`scrub_text` helper enforces this.

Trajectory contract
-------------------

We accept the *raw* harness JSONL trajectory (``trajectory.jsonl``) and
the per-run summary (``run_summary.json``). The structure produced by
:mod:`myevoskill.harness.trajectory` is what we parse here; we tolerate
unknown event kinds by ignoring them, so adding new event types in the
harness will not silently break the miner.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Sequence

from .universe import DistillUniverse, ValidationLeakError


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class ToolUseEpisode:
    """One tool call inside a passing run.

    Attributes are intentionally narrow: we want the synthesizer to see
    *what was done* and *what came back*, not the model's free-form
    chain-of-thought.
    """

    round_index: int
    tool: str  # "Bash" / "Read" / "Edit" / "Write" / ...
    input_signature: str  # short scrub of the tool input (e.g. "python -c '...'")
    success: bool  # judged by the post-tool-use hook
    bytes_written: int = 0  # for Write/Edit: size of the patch
    referenced_files: tuple[str, ...] = ()  # workspace-relative paths only
    result_tail: str = ""  # scrubbed tail from the matching tool_result


@dataclass(frozen=True)
class FailureSignal:
    """One judge feedback flip from FAIL/INVALID -> something else.

    Used by the synthesizer to write the *Failure Modes* section of a
    skill: "if you see metric X failing, do Y".
    """

    round_index: int
    verdict: str
    failure_tags: tuple[str, ...]
    metric_status: Mapping[str, object] = field(default_factory=dict)
    metrics_actual: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MainPyDigest:
    """Small, domain-agnostic AST/text digest of the agent's final solver."""

    helper_calls: tuple[str, ...] = ()
    has_timing_probe: bool = False
    has_metadata_epoch_loop: bool = False
    hardcoded_constants: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskEpisode:
    """All evidence mined from one train run.

    ``final_verdict == "PASS"`` records positive evidence. Non-PASS
    episodes are train-only counterexamples: they must never be treated as
    successful demonstrations, but they are valuable anti-pattern material
    for distillation.
    """

    task_id: str
    run_id: str
    family: str
    final_verdict: str
    rounds_used: int
    runtime_seconds: float
    tool_uses: tuple[ToolUseEpisode, ...]
    failure_signals: tuple[FailureSignal, ...]
    primary_output_rel: str
    metrics_actual: Mapping[str, object] = field(default_factory=dict)
    metric_status: Mapping[str, object] = field(default_factory=dict)
    plan_summary: tuple[str, ...] = ()
    main_py_digest: MainPyDigest = field(default_factory=MainPyDigest)
    # Hash-keyed pointers back to the train-side reference solution. The
    # synthesizer is allowed to *read* these via the universe; the miner
    # only records *which* files the agent actually used so we don't pull
    # in the entire reference dump for every skill draft.
    relevant_reference_files: tuple[str, ...] = ()


# --------------------------------------------------------------------------- scrubbers


# Substrings that, if seen in any text payload, mean we drop the whole event
# rather than risk leaking. Conservative on purpose.
_HARD_DROP_SUBSTRINGS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "Bearer ",
)

# Regexes used to strip per-run / per-host noise.
_ABS_RUN_PATH_RE = re.compile(
    r"(?ix)"
    r"(?:[A-Z]:)?[\\/](?:Users|home|root|tmp|var|opt|workspaces|sandboxes)"
    r"[\\/][^\s\"']+",
)
_HEX_RUN_ID_RE = re.compile(r"run-\d{10}-[0-9a-f]{6}")
_TOKEN_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")


def scrub_text(text: str, *, valid_task_ids: Sequence[str]) -> str | None:
    """Return a scrubbed copy of ``text`` or ``None`` if it must be dropped.

    The dropping behaviour is mandatory for hard-secret matches; for path
    / token noise we just rewrite to placeholder strings.
    """

    if any(needle in text for needle in _HARD_DROP_SUBSTRINGS):
        return None
    for tid in valid_task_ids:
        # If the trajectory event quotes a valid-split task id (impossible
        # if the universe boundary is honoured upstream, but defence in
        # depth) we drop the event outright.
        if tid and tid in text:
            return None
    cleaned = _ABS_RUN_PATH_RE.sub("<RUN_PATH>", text)
    cleaned = _HEX_RUN_ID_RE.sub("<RUN_ID>", cleaned)
    cleaned = _TOKEN_RE.sub("<TOKEN>", cleaned)
    return cleaned


# --------------------------------------------------------------------------- miner


def _iter_trajectory(text: str) -> Iterator[Mapping[str, object]]:
    """Yield JSON objects from a JSONL or JSON trajectory string."""

    stripped = text.strip()
    if not stripped:
        return
    if stripped.startswith("["):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, Mapping):
                    yield item
            return
    if stripped.startswith("{") and "\n" not in stripped:
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, Mapping):
            yield obj
            return

    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping):
            yield obj


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _short_signature(tool: str, tool_input: object, *, limit: int = 240) -> str:
    """Return a short, scrubbed string describing a tool input.

    We deliberately do *not* preserve the full tool input (which can be
    pages of source). Only the first ``limit`` characters of the most
    salient field are kept, which is enough for the synthesizer to
    cluster similar moves.
    """

    if not isinstance(tool_input, Mapping):
        return ""
    salient_keys = {
        "Bash": ("command", "description"),
        "Read": ("file_path",),
        "Write": ("file_path",),
        "Edit": ("file_path",),
        "MultiEdit": ("file_path",),
        "Glob": ("pattern",),
        "Grep": ("pattern",),
    }.get(tool, ())
    parts: list[str] = []
    for key in salient_keys:
        v = tool_input.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    snippet = " | ".join(parts) or json.dumps(tool_input, ensure_ascii=False)
    return snippet[:limit]


def _tail_text(text: object, *, limit: int = 800) -> str:
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]


_FILE_PATH_RE = re.compile(r"(?:[A-Za-z0-9_\-./\\]+)\.(?:py|ipynb|json|md|npy|npz|yaml|yml|txt)\b")


def _referenced_files(payload: object) -> tuple[str, ...]:
    """Best-effort extraction of workspace-relative file references."""

    if not isinstance(payload, str):
        return ()
    refs: list[str] = []
    for m in _FILE_PATH_RE.finditer(payload):
        candidate = m.group(0).replace("\\", "/")
        # Drop absolute paths and anything that escaped scrubbing.
        if candidate.startswith(("<RUN_PATH>", "/", "C:", "D:")):
            continue
        if ".." in candidate.split("/"):
            continue
        refs.append(candidate)
    # Stable, dedup, capped.
    seen: list[str] = []
    for r in refs:
        if r not in seen:
            seen.append(r)
        if len(seen) >= 12:
            break
    return tuple(seen)


def _normalise_metrics(obj: object) -> dict[str, object]:
    if not isinstance(obj, Mapping):
        return {}
    out: dict[str, object] = {}
    for key, value in obj.items():
        if isinstance(key, str) and (
            isinstance(value, (str, int, float, bool)) or value is None
        ):
            out[str(key)] = value
    return out


def _judge_metrics_for_round(
    *,
    universe: DistillUniverse,
    task_id: str,
    run_dir: Path,
    round_index: int,
) -> dict[str, object]:
    if round_index <= 0:
        return {}
    try:
        text = universe.read_log_file(task_id, run_dir, f"judge_round_{round_index:02d}.json")
    except (FileNotFoundError, PermissionError, OSError, ValidationLeakError):
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, Mapping):
        return {}
    judge = data.get("judge_result") or {}
    if isinstance(judge, Mapping):
        return _normalise_metrics(judge.get("metrics_actual"))
    return {}


def _plan_summary_from_history(summary: Mapping[str, object]) -> tuple[str, ...]:
    out: list[str] = []
    history = summary.get("plan_history") or []
    if isinstance(history, list):
        for item in history[-5:]:
            if not isinstance(item, Mapping):
                continue
            round_idx = _coerce_int(item.get("round"))
            diff = _coerce_int(item.get("diff_lines"))
            note = str(item.get("note") or "")
            out.append(f"round={round_idx} diff_lines={diff} note={note}"[:160])
    return tuple(out)


def _constant_repr(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str) and len(value) <= 80:
        return repr(value)
    return None


def _main_py_digest(text: str) -> MainPyDigest:
    lower = text.lower()
    has_timing_probe = any(
        needle in lower
        for needle in ("time.time", "perf_counter", "timing", "elapsed", "probe", "budget")
    )
    has_metadata_epoch_loop = bool(
        re.search(r"(epoch|n_epochs|epochs).{0,80}(meta|metadata|config)", lower)
        or re.search(r"(meta|metadata|config).{0,80}(epoch|n_epochs|epochs)", lower)
    )
    helper_calls: list[str] = []
    constants: list[str] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        scripts = sorted(set(re.findall(r"[\w./\\-]+\.py\b", text)))[:12]
        return MainPyDigest(
            helper_calls=tuple(scripts),
            has_timing_probe=has_timing_probe,
            has_metadata_epoch_loop=has_metadata_epoch_loop,
            hardcoded_constants=(),
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            string_args: list[str] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if ".py" in child.value or child.value.startswith("--"):
                        string_args.append(child.value.replace("\\", "/")[:120])
            if string_args:
                helper_calls.append(" ".join(string_args)[:240])
        elif isinstance(node, ast.Assign):
            names: list[str] = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
            value = node.value
            if names and isinstance(value, ast.Constant):
                rendered = _constant_repr(value.value)
                if rendered is not None:
                    name = names[0]
                    if re.search(r"(lr|scale|epoch|iter|vmin|vmax|clip|sigma|step|budget)", name, re.I):
                        constants.append(f"{name}={rendered}"[:120])

    def dedupe(items: list[str], limit: int) -> tuple[str, ...]:
        seen: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.append(item)
            if len(seen) >= limit:
                break
        return tuple(seen)

    return MainPyDigest(
        helper_calls=dedupe(helper_calls, 12),
        has_timing_probe=has_timing_probe,
        has_metadata_epoch_loop=has_metadata_epoch_loop,
        hardcoded_constants=dedupe(constants, 16),
    )


def _read_main_py_digest(
    universe: DistillUniverse,
    task_id: str,
    run_id: str,
) -> MainPyDigest:
    for rel in ("work/main.py", "main.py"):
        try:
            text = universe.read_workspace_file(task_id, run_id, rel)
        except (FileNotFoundError, PermissionError, OSError, ValidationLeakError):
            continue
        return _main_py_digest(text)
    return MainPyDigest()


def mine_run(
    *,
    universe: DistillUniverse,
    task_id: str,
    run_dir: Path,
    require_pass: bool = True,
) -> TaskEpisode | None:
    """Mine one train run.

    When ``require_pass`` is true, returns ``None`` unless the run passed.
    When false, non-PASS runs are returned as train-only counterexamples.
    Reads are routed through ``universe`` (which raises on valid leaks).
    """

    universe.assert_train(task_id, "log", run_dir.name)

    summary_text = None
    for rel in ("run_summary.json", "summary.json"):
        try:
            summary_text = universe.read_log_file(task_id, run_dir, rel)
            break
        except FileNotFoundError:
            continue
    if summary_text is None:
        return None
    try:
        summary = json.loads(summary_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(summary, Mapping):
        return None
    final_verdict = str(summary.get("verdict") or "")
    if require_pass and final_verdict != "PASS":
        return None

    family = ""
    policy = summary.get("policy") or {}
    if isinstance(policy, Mapping):
        primary_output_rel = str(policy.get("primary_output_rel") or "output/reconstruction.npz")
    else:
        primary_output_rel = "output/reconstruction.npz"
    feedback_history = summary.get("feedback_history") or []
    if isinstance(feedback_history, list):
        signals: list[FailureSignal] = []
        final_metric_status: dict[str, object] = {}
        final_metrics_actual: dict[str, object] = {}
        for entry in feedback_history:
            if not isinstance(entry, Mapping):
                continue
            fb = entry.get("feedback") or {}
            if not isinstance(fb, Mapping):
                continue
            verdict = str(fb.get("verdict") or "")
            round_index = _coerce_int(entry.get("round"))
            metric_status = _normalise_metrics(fb.get("metric_status"))
            metrics_actual = _judge_metrics_for_round(
                universe=universe,
                task_id=task_id,
                run_dir=run_dir,
                round_index=round_index,
            )
            if metric_status:
                final_metric_status = metric_status
            if metrics_actual:
                final_metrics_actual = metrics_actual
            if verdict in ("PASS", ""):
                continue
            tags = fb.get("failure_tags") or []
            tags_t = tuple(str(t) for t in tags if isinstance(t, str))
            signals.append(
                FailureSignal(
                    round_index=round_index,
                    verdict=verdict,
                    failure_tags=tags_t,
                    metric_status=metric_status,
                    metrics_actual=metrics_actual,
                )
            )
        failure_signals = tuple(signals)
    else:
        failure_signals = ()
        final_metric_status = {}
        final_metrics_actual = {}

    trajectory_text = None
    for rel in ("trajectory.jsonl", "trajectory.json"):
        try:
            trajectory_text = universe.read_log_file(task_id, run_dir, rel)
            break
        except FileNotFoundError:
            continue
    if trajectory_text is None:
        return None
    valid_ids = list(universe.valid_task_ids)
    events = list(_iter_trajectory(trajectory_text))
    result_by_id: dict[str, tuple[str, bool]] = {}
    for event in events:
        kind = str(event.get("kind") or event.get("event") or "")
        if kind != "tool_result":
            continue
        scrubbed = scrub_text(json.dumps(event, ensure_ascii=False), valid_task_ids=valid_ids)
        if scrubbed is None:
            continue
        tool_use_id = str(event.get("tool_use_id") or "")
        if tool_use_id:
            result_by_id[tool_use_id] = (
                _tail_text(event.get("text")),
                not bool(event.get("is_error")),
            )

    tool_uses: list[ToolUseEpisode] = []
    refs_aggregate: list[str] = []
    for event in events:
        kind = str(event.get("kind") or event.get("event") or "")
        if kind not in {"pre_tool_use", "post_tool_use", "tool_call"}:
            continue
        scrubbed = scrub_text(json.dumps(event, ensure_ascii=False), valid_task_ids=valid_ids)
        if scrubbed is None:
            continue
        round_index = _coerce_int(event.get("round_index") or event.get("round"))
        tool = str(event.get("tool_name") or event.get("tool") or "")
        if not tool:
            continue
        tool_input = event.get("tool_input") or event.get("input")
        signature = _short_signature(tool, tool_input)
        signature_clean = scrub_text(signature, valid_task_ids=valid_ids) or ""
        tool_use_id = str(event.get("tool_use_id") or "")
        result_tail = ""
        result_success = None
        if tool_use_id and tool_use_id in result_by_id:
            result_tail, result_success = result_by_id[tool_use_id]
            result_tail = scrub_text(result_tail, valid_task_ids=valid_ids) or ""
        success_field = event.get("success")
        if isinstance(success_field, bool):
            success = success_field
        elif isinstance(result_success, bool):
            success = result_success
        else:
            # PreToolUse events have no outcome yet; treat as success
            # placeholder (the synthesizer pairs them with the run-level
            # PASS verdict, so individual pre events aren't ground truth).
            success = kind != "post_tool_use" or bool(event.get("ok", True))
        bytes_written = _coerce_int(event.get("bytes_written"))
        ref_files = _referenced_files(signature_clean)
        for rf in ref_files:
            if rf not in refs_aggregate:
                refs_aggregate.append(rf)
        tool_uses.append(
            ToolUseEpisode(
                round_index=round_index,
                tool=tool,
                input_signature=signature_clean,
                success=success,
                bytes_written=bytes_written,
                referenced_files=ref_files,
                result_tail=result_tail,
            )
        )

    return TaskEpisode(
        task_id=str(task_id),
        run_id=str(summary.get("run_id") or run_dir.name),
        family=family,
        final_verdict=final_verdict or "UNKNOWN",
        rounds_used=_coerce_int(summary.get("rounds_used")),
        runtime_seconds=_coerce_float(summary.get("runtime_seconds")),
        tool_uses=tuple(tool_uses),
        failure_signals=failure_signals,
        primary_output_rel=primary_output_rel,
        metrics_actual=final_metrics_actual,
        metric_status=final_metric_status,
        plan_summary=_plan_summary_from_history(summary),
        main_py_digest=_read_main_py_digest(
            universe,
            task_id,
            str(summary.get("run_id") or run_dir.name),
        ),
        relevant_reference_files=tuple(refs_aggregate[:24]),
    )


def mine_train_split(universe: DistillUniverse) -> List[TaskEpisode]:
    """Mine recent train evidence for every train task.

    We keep the most recent PASS as positive evidence and the most recent
    FAIL/TIMEOUT as train-only counterexamples. If a train task has no
    passing run we still return its latest non-PASS evidence when present,
    but the synthesizer will refuse to build a skill unless at least one
    PASS exists.
    Valid-split tasks are *not* visited, by construction of the universe.
    """

    out: list[TaskEpisode] = []
    for task_id in universe.train_task_ids:
        try:
            runs = universe.list_runs(task_id)
        except ValidationLeakError:
            continue
        latest_pass: TaskEpisode | None = None
        nonpassing_by_verdict: dict[str, TaskEpisode] = {}
        for run_dir in reversed(runs):
            if latest_pass is None:
                episode = mine_run(
                    universe=universe,
                    task_id=task_id,
                    run_dir=run_dir,
                    require_pass=True,
                )
                if episode is not None:
                    latest_pass = episode
                    continue
            if not {"FAIL", "TIMEOUT"} - set(nonpassing_by_verdict):
                continue
            episode = mine_run(
                universe=universe,
                task_id=task_id,
                run_dir=run_dir,
                require_pass=False,
            )
            if episode is None or episode.final_verdict == "PASS":
                continue
            if episode.final_verdict in {"FAIL", "TIMEOUT"}:
                nonpassing_by_verdict.setdefault(episode.final_verdict, episode)
        if latest_pass is not None:
            out.append(latest_pass)
        out.extend(nonpassing_by_verdict[v] for v in ("FAIL", "TIMEOUT") if v in nonpassing_by_verdict)
    return out


__all__ = [
    "FailureSignal",
    "MainPyDigest",
    "TaskEpisode",
    "ToolUseEpisode",
    "mine_run",
    "mine_train_split",
    "scrub_text",
]
