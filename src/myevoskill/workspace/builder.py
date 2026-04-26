"""Build a per-run agent workspace from a registered task.

The builder is *deterministic* and does not require any LLM. It reads the
manifest produced by ``myevoskill register-task`` and produces:

    artifacts/workspaces/<task>/<run_id>/
        README.md            # sanitised public README
        meta_data.json       # task metadata visible to the agent
        data/                # symlinked or copied public data files
        work/                # empty, agent scratch
        output/              # empty, primary output target

We intentionally do NOT copy:
    * ground_truth.npz, reference_outputs/      (judge-only)
    * evaluation/ (task_contract.json, judge_adapter.py, ...)
    * src/, main.py, notebooks/, plan/          (reference solution)

Even if the public_policy.public_data_allowlist mentions one of these by
mistake, the WorkspacePolicy denylist will reject it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .agent_spec import (
    assert_spec_has_no_leaks,
    derive_agent_task_spec,
    render_summary,
)
from .policy import GLOBAL_FORBIDDEN_SUBSTRINGS, WorkspacePolicy

_DEFAULT_README_NAMES = ("README.md", "README.MD", "Readme.md", "readme.md")
_AGENT_TASK_SPEC_FILENAME = "agent_task_spec.json"
_TASK_CONTRACT_REL = "evaluation/task_contract.json"


@dataclass(frozen=True)
class WorkspaceBuild:
    """Result of a successful workspace build."""

    task_id: str
    run_id: str
    agent_root: Path
    policy: WorkspacePolicy
    primary_output_path: Path
    copied_files: tuple[str, ...]
    skipped_files: tuple[str, ...]
    readme_path: Path
    meta_path: Path
    agent_task_spec_path: Path | None = None
    agent_task_spec_summary: str = ""


# --------------------------------------------------------------------- helpers


def _resolve_task_root(repo_root: Path, manifest: Mapping[str, object]) -> Path:
    """Resolve the on-disk task directory pointed to by the manifest.

    Manifests written by ``register-task`` use ``source_task_dir`` paths that
    are relative to the *repository root* (``MyEvoSkill/``), e.g.
    ``"../tasks/cars_spectroscopy"`` to mean ``<repo_root>/../tasks/...``.
    We try a few candidate base directories so manifests authored in older
    layouts still resolve correctly.
    """

    raw = str(manifest.get("source_task_dir") or "")
    if not raw:
        raise ValueError("manifest is missing source_task_dir")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()

    bases = [
        repo_root,                          # new layout: relative to MyEvoSkill/
        repo_root / "registry" / "tasks",   # legacy layout
        repo_root.parent,                   # in case manifest used "tasks/<id>"
    ]
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    # Fall back to the new-layout interpretation so the missing-dir error
    # message points at the most likely intended location.
    return (repo_root / candidate).resolve()


def _read_readme(task_root: Path) -> str:
    for name in _DEFAULT_README_NAMES:
        path = task_root / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


# --- ground-truth heading filter (commit F) --------------------------------
#
# We deliberately strip *only* ground-truth-referencing headings. Sections
# that describe the metric, evaluation channel, or reference/solution
# **methodology** are kept: leaking the *method* is the agent's job to
# discover (and cribbing it back is fine - the task contract explicitly
# permits the agent to study `meta_data.json` + README), but the actual
# numerical ground truth (and any cell pointing at the ``ground_truth*``
# artefact) must not appear.
#
# The headings below are matched case-insensitively, with optional leading
# emoji / decorative punctuation tolerated, against H1..H6 markdown
# headings only. Anything inside a matched heading up to (but not
# including) the next heading at the same or shallower level is removed.

_GROUND_TRUTH_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^ground[\s\-_]*truth\b.*$", re.IGNORECASE),
    re.compile(r"^reference\s+(answer|target|values?)\b.*$", re.IGNORECASE),
    re.compile(r"^expected\s+(output|values?)\b.*$", re.IGNORECASE),
)

# Cells / rows whose presence in a markdown table flips us into
# "drop the whole table" mode. We only drop tables that *literally*
# point at the ground-truth artefact -- generic metric or threshold
# tables are preserved.
_GROUND_TRUTH_TABLE_NEEDLES: tuple[re.Pattern[str], ...] = (
    re.compile(r"ground[\s\-_]*truth", re.IGNORECASE),
    re.compile(r"reference[_\s]+(?:answer|target|values?|outputs?)", re.IGNORECASE),
    re.compile(r"\bexpected[_\s]+(?:output|values?)\b", re.IGNORECASE),
)


def _heading_is_ground_truth(title: str) -> bool:
    """``True`` iff ``title`` looks like a ground-truth section heading.

    Tolerates a leading emoji / punctuation prefix (e.g.
    ``"🟥 Ground-Truth (private)"``).
    """

    cleaned = title.strip()
    # strip any leading run of non-letter/non-digit characters (emoji,
    # decorative bullets, brackets, ...). We keep going until we hit the
    # first letter or end-of-string.
    i = 0
    while i < len(cleaned) and not (cleaned[i].isalnum()):
        i += 1
    cleaned = cleaned[i:]
    return any(p.match(cleaned) for p in _GROUND_TRUTH_HEADING_PATTERNS)


def _sanitise_readme(text: str, readme_policy: Mapping[str, object]) -> str:
    """Strip ground-truth references from the README.

    Always removes (commit F default):

    * any markdown heading whose title matches a ground-truth pattern,
      together with the entire section it heads (down to the next heading
      at the same or shallower level);
    * any markdown table that contains at least one cell mentioning the
      ground-truth artefact (header row or body row, case-insensitive).

    Additionally honours the legacy contract-driven knobs:

    * ``remove_path_patterns``  -- per-line regex denylist;
    * ``remove_sections``       -- explicit extra section titles to drop;
    * ``preserve_sections``     -- whitelist that wins over both above.
    """

    if not text:
        return ""

    remove_patterns = list(readme_policy.get("remove_path_patterns") or [])
    remove_sections = [s.strip().lower() for s in (readme_policy.get("remove_sections") or [])]
    preserve_sections = [s.strip().lower() for s in (readme_policy.get("preserve_sections") or [])]

    # Always strip global forbidden substrings (case-insensitive).
    extra = list(remove_patterns)
    # Substrings that, while forbidden as *paths*, would clobber too much
    # legitimate prose if applied line-wise to a README. The README sanitiser
    # skips them; the runtime PreToolUse hook still enforces them on actual
    # tool inputs, which is what matters.
    _README_SKIP = {
        "/src/", "\\src\\",
        "/notebooks/", "\\notebooks\\",
        "/plan/", "\\plan\\",
        "/main.py", "\\main.py",
        ".ipynb",
    }
    for sub in GLOBAL_FORBIDDEN_SUBSTRINGS:
        if sub in _README_SKIP:
            continue
        extra.append(re.escape(sub))

    # Order matters:
    #   1. drop GT-referencing tables (whole-block) first, otherwise the
    #      per-line filter below would replace the offending row with a
    #      comment placeholder and break our table detection regex,
    #      leaving the surrounding rows orphaned.
    #   2. drop GT-headed sections.
    #   3. apply the per-line denylist (paths, etc.) on what remains.
    #   4. apply contract-level extra/preserve sections.
    cleaned = _strip_ground_truth_tables(text)
    cleaned = _strip_ground_truth_sections(cleaned, preserve_sections)

    cleaned_lines: list[str] = []
    for line in cleaned.splitlines():
        drop = False
        for pat in extra:
            try:
                if re.search(pat, line):
                    drop = True
                    break
            except re.error:
                continue
        if drop:
            cleaned_lines.append("<!-- line removed by workspace sanitiser -->")
        else:
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    if remove_sections:
        cleaned = _strip_sections(cleaned, remove_sections, preserve_sections)
    return cleaned


_SECTION_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _strip_ground_truth_sections(text: str, preserve_sections: Sequence[str]) -> str:
    """Remove any heading recognised by ``_heading_is_ground_truth``."""

    preserve = {s.strip().lower() for s in preserve_sections}
    out: list[str] = []
    skipping = False
    skip_level = 0
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if skipping and level <= skip_level:
                skipping = False
            if (
                not skipping
                and _heading_is_ground_truth(title)
                and title.lower() not in preserve
            ):
                skipping = True
                skip_level = level
                continue
        if not skipping:
            out.append(line)
    return "\n".join(out)


def _strip_sections(
    text: str,
    remove_sections: Sequence[str],
    preserve_sections: Sequence[str],
) -> str:
    out: list[str] = []
    skipping = False
    skip_level = 0
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip().lower()
            if skipping and level <= skip_level:
                skipping = False
            if not skipping and title in remove_sections and title not in preserve_sections:
                skipping = True
                skip_level = level
                continue
        if not skipping:
            out.append(line)
    return "\n".join(out)


_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def _line_mentions_ground_truth(line: str) -> bool:
    return any(p.search(line) for p in _GROUND_TRUTH_TABLE_NEEDLES)


def _strip_ground_truth_tables(text: str) -> str:
    """Remove markdown tables that reference ground-truth artefacts.

    A markdown table is detected as a contiguous run of lines matching
    ``^\\s*\\|.*\\|\\s*$``. If *any* row of that block (header, separator,
    or body) contains a ground-truth needle we drop the whole block.
    Plain prose in between tables is untouched -- callers must pair this
    with ``_strip_ground_truth_sections`` to handle non-table mentions.
    """

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _TABLE_ROW_RE.match(line):
            out.append(line)
            i += 1
            continue

        # Collect the contiguous table block.
        j = i
        while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
            j += 1
        block = lines[i:j]
        block_has_gt = any(_line_mentions_ground_truth(b) for b in block)
        if block_has_gt:
            out.append(
                "<!-- ground-truth-referencing table removed by workspace sanitiser -->"
            )
        else:
            out.extend(block)
        i = j
    return "\n".join(out)


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _build_meta(manifest: Mapping[str, object], task_root: Path) -> dict[str, object]:
    """Build the meta_data.json visible to the agent."""

    out: dict[str, object] = {
        "task_id": manifest.get("task_id"),
        "family": manifest.get("family"),
        "primary_output_path": manifest.get("primary_output_path"),
        "runtime_layout": manifest.get("runtime_layout"),
    }
    # Copy any meta_data.json the task already publishes (it is *task-author*
    # facing, not judge-facing).
    candidate = task_root / "data" / "meta_data.json"
    if candidate.exists():
        try:
            out["task_meta"] = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            out["task_meta_raw"] = candidate.read_text(encoding="utf-8", errors="replace")
    return out


# ---------------------------------------------------------------- main builder


def build_workspace(
    *,
    repo_root: Path,
    manifest: Mapping[str, object],
    run_id: str,
) -> WorkspaceBuild:
    """Materialise a clean per-run workspace and return its policy."""

    task_id = str(manifest["task_id"])
    primary_output_rel = str(manifest.get("primary_output_path") or "output/reconstruction.npz")

    repo_root = Path(repo_root).resolve()
    task_root = _resolve_task_root(repo_root, manifest)

    # 1. Decide layout.
    workspace_root = repo_root / "artifacts" / "workspaces" / task_id / run_id
    if workspace_root.exists():
        shutil.rmtree(workspace_root, ignore_errors=True)
    workspace_root.mkdir(parents=True, exist_ok=True)

    runtime_layout = dict(manifest.get("runtime_layout") or {})
    data_dir_name = str(runtime_layout.get("data_dir") or "data")
    work_dir_name = str(runtime_layout.get("work_dir") or "work")
    output_dir_name = str(runtime_layout.get("output_dir") or "output")

    (workspace_root / data_dir_name).mkdir(parents=True, exist_ok=True)
    (workspace_root / work_dir_name).mkdir(parents=True, exist_ok=True)
    (workspace_root / output_dir_name).mkdir(parents=True, exist_ok=True)

    # 2. Build the policy first; we use it to filter the data allowlist.
    policy = WorkspacePolicy.from_manifest(
        agent_root=workspace_root,
        manifest=manifest,
        primary_output_rel=primary_output_rel,
    )

    # 3. Copy public data files.
    public_policy = dict(manifest.get("public_policy") or {})
    allowlist: Iterable[str] = list(public_policy.get("public_data_allowlist") or [])
    copied: list[str] = []
    skipped: list[str] = []
    for rel in allowlist:
        rel = str(rel).strip()
        if not rel:
            continue
        # The workspace policy denylist applies even to the allowlist - belt &
        # braces.
        if policy.find_forbidden(rel):
            skipped.append(f"{rel} (denylisted)")
            continue
        src = (task_root / rel).resolve()
        if not str(src).startswith(str(task_root)):
            skipped.append(f"{rel} (outside task_root)")
            continue
        if not src.exists():
            skipped.append(f"{rel} (missing)")
            continue
        dst = workspace_root / rel
        _safe_copy(src, dst)
        copied.append(rel)

    # 4. README.
    readme_text = _sanitise_readme(
        _read_readme(task_root),
        public_policy.get("readme_policy") or {},
    )
    readme_path = workspace_root / "README.md"
    if not readme_text.strip():
        readme_text = (
            f"# {task_id}\n\nNo public README is provided for this task. Read"
            " `meta_data.json` and the files under `data/` to understand the"
            " problem, then write your solution into `output/`.\n"
        )
    readme_path.write_text(readme_text, encoding="utf-8")

    # 5. Meta data.
    meta_path = workspace_root / "meta_data.json"
    meta_path.write_text(
        json.dumps(_build_meta(manifest, task_root), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    primary_output_path = (workspace_root / primary_output_rel).resolve()

    # 6. Agent task spec (machine-readable IO contract).
    #
    # We derive a *strictly redacted* view of the task contract -- no
    # threshold, no metric name, no ground_truth path, no judge helper --
    # so the agent has a deterministic source of truth for the output
    # schema without being able to inspect or hack the judge.
    agent_spec_path: Path | None = None
    agent_spec_summary: str = ""
    contract_path = task_root / _TASK_CONTRACT_REL
    if contract_path.exists():
        try:
            contract_dict = json.loads(contract_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            contract_dict = None
        if isinstance(contract_dict, Mapping):
            spec = derive_agent_task_spec(contract=contract_dict, manifest=manifest)
            # Defence-in-depth: refuse to write a spec that leaks.
            assert_spec_has_no_leaks(spec)
            agent_spec_path = workspace_root / _AGENT_TASK_SPEC_FILENAME
            agent_spec_path.write_text(
                json.dumps(spec, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            agent_spec_summary = render_summary(spec)

    return WorkspaceBuild(
        task_id=task_id,
        run_id=run_id,
        agent_root=workspace_root,
        policy=policy,
        primary_output_path=primary_output_path,
        copied_files=tuple(copied),
        skipped_files=tuple(skipped),
        readme_path=readme_path,
        meta_path=meta_path,
        agent_task_spec_path=agent_spec_path,
        agent_task_spec_summary=agent_spec_summary,
    )
