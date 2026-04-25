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

from .policy import GLOBAL_FORBIDDEN_SUBSTRINGS, WorkspacePolicy

_DEFAULT_README_NAMES = ("README.md", "README.MD", "Readme.md", "readme.md")


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


# --------------------------------------------------------------------- helpers


def _resolve_task_root(repo_root: Path, manifest: Mapping[str, object]) -> Path:
    raw = str(manifest.get("source_task_dir") or "")
    if not raw:
        raise ValueError("manifest is missing source_task_dir")
    candidate = Path(raw)
    if not candidate.is_absolute():
        # Manifest source_task_dir is relative to MyEvoSkill/registry/tasks/.
        candidate = (repo_root / "registry" / "tasks" / candidate).resolve()
    return candidate.resolve()


def _read_readme(task_root: Path) -> str:
    for name in _DEFAULT_README_NAMES:
        path = task_root / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _sanitise_readme(text: str, readme_policy: Mapping[str, object]) -> str:
    """Strip references to hidden assets from the README."""

    if not text:
        return ""

    remove_patterns = list(readme_policy.get("remove_path_patterns") or [])
    remove_sections = [s.strip().lower() for s in (readme_policy.get("remove_sections") or [])]
    preserve_sections = [s.strip().lower() for s in (readme_policy.get("preserve_sections") or [])]

    # Always strip global forbidden substrings (case-insensitive).
    extra = list(remove_patterns)
    for sub in GLOBAL_FORBIDDEN_SUBSTRINGS:
        # Skip noisy single-token entries that would clobber legitimate prose.
        if sub in {"/src/", "\\src\\"}:
            continue
        extra.append(re.escape(sub))

    cleaned_lines: list[str] = []
    for line in text.splitlines():
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
    )
