"""Workspace + filesystem policy for one task run.

Single source of truth used by:
    * the workspace builder (decides which task files to copy in)
    * the harness PreToolUse hooks (decides which tool calls to block)
    * the trajectory writer (records the policy that was active)

The policy is intentionally simple:
    * agent_root      -> the only directory the agent may read or write
    * forbidden_subs  -> hard substrings that must never appear in any tool
                          input (e.g. "ground_truth", "evaluation/", "main.py")
    * dangerous_cmds  -> regex patterns we refuse on the Bash tool
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Mapping, Sequence

# Hard-coded substrings that must never appear in any path-shaped tool input,
# regardless of the task. They cover the standard hidden assets and are merged
# with any task-specific denylist coming from the registry manifest.
# These cover hidden task assets (ground truth, evaluation harness, reference
# solutions, reference plans, reference notebooks).  They are matched as
# substrings against tool inputs *in addition to* the workspace-boundary
# check.
#
# Three layers of defence keep the agent away from the reference solution:
#   1. ``registration.py`` refuses to put files under ``src/``, ``notebooks/``,
#      ``plan/`` or any ``main.py`` / ``*.ipynb`` into the public allowlist.
#   2. ``workspace/builder.py`` only copies files that survive (1).
#   3. ``hooks.make_pre_tool_use_hook`` blocks any tool input whose path
#      contains a substring listed below, so even if the agent fabricates a
#      path to ``../tasks/<id>/src/foo.py`` the call is rejected.
#
# We accept that the agent may legitimately want a top-level file called
# ``main.py`` inside its own workspace; the substrings below intentionally
# anchor on a path-separator (``/main.py``) so they only match references to
# the *source-task* tree, not files the agent itself creates in its cwd.
GLOBAL_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "ground_truth",
    "evaluation/",
    "evaluation\\",
    "/evaluation",
    "\\evaluation",
    "reference_outputs",
    "task_contract.json",
    "task_contract.public.json",
    "judge_adapter.py",
    "registration_contract.json",
    # Reference solution / reference notebooks / reference plans.
    "/src/",
    "\\src\\",
    "/notebooks/",
    "\\notebooks\\",
    "/plan/",
    "\\plan\\",
    # NOTE: We deliberately do NOT block ``main.py`` at runtime via a substring
    # match. The contract's ``execution.entrypoint`` for many tasks is
    # ``work/main.py``, which the agent legitimately creates. Reference
    # ``main.py`` files under ``tasks/<id>/`` are already (a) excluded from
    # the workspace by the builder and (b) outside the agent_root, so the
    # path-prefix check in ``is_inside`` will deny them anyway. ``src/``,
    # ``notebooks/``, ``plan/`` and ``.ipynb`` are still blocked here as a
    # defence-in-depth in case a stale workspace ever contained one.
    ".ipynb",
)

# Regexes matched against the *full* Bash command. We intentionally use simple,
# obvious patterns; the goal is to stop trivially-malicious egress, not to be a
# perfect sandbox.
DANGEROUS_BASH_PATTERNS: tuple[str, ...] = (
    r"(?i)\bsudo\b",
    r"(?i)\bcurl\b",
    r"(?i)\bwget\b",
    r"(?i)\bnc\b",
    r"(?i)\bnetcat\b",
    r"(?i)\bssh\b",
    r"(?i)\bscp\b",
    r"(?i)\brsync\b",
    r"(?i)\bpip\s+install\b",
    r"(?i)\bconda\s+install\b",
    # Block leaving the workspace via cd. The harness pins cwd inside the
    # workspace; we additionally forbid `cd <abs path>` and `cd ..` chains.
    r"(?i)(^|;|&&|\|\|)\s*cd\s+/",
    r"(?i)(^|;|&&|\|\|)\s*cd\s+[A-Za-z]:[\\/]",
    r"(?i)(^|;|&&|\|\|)\s*cd\s+\.\.(?:$|\s|[\\/])",
    r"(?i)(^|;|&&|\|\|)\s*cd\s+\.\.(\\|/)?\.\.",
    # Cat/head/tail/less/more/type/grep/find on absolute paths leaving cwd.
    r"(?i)\b(cat|head|tail|less|more|type|grep|rg|find|ls|dir)\b[^\n]*\s(/|[A-Za-z]:[\\/])",
)


@dataclass(frozen=True)
class WorkspacePolicy:
    """Filesystem policy for one run.

    Parameters
    ----------
    agent_root:
        Absolute path the agent treats as its root. All tool calls must stay
        inside this directory. The harness pins ``cwd`` here.
    primary_output_rel:
        Path relative to ``agent_root`` where the judge will read the result.
    forbidden_substrings:
        Substrings (case-sensitive on Windows path separators) that must not
        appear in any tool input. ``GLOBAL_FORBIDDEN_SUBSTRINGS`` are always
        included; this list is appended to it.
    dangerous_bash_patterns:
        Regexes (compiled with ``re.search``) that reject a Bash command
        outright. ``DANGEROUS_BASH_PATTERNS`` are always included.
    writable_subdirs:
        Subdirectories of ``agent_root`` the agent may write to. Defaults to
        ``("work", "output")``.
    """

    agent_root: Path
    primary_output_rel: str
    forbidden_substrings: Sequence[str] = field(default_factory=tuple)
    dangerous_bash_patterns: Sequence[str] = field(default_factory=tuple)
    writable_subdirs: Sequence[str] = ("work", "output")

    # ------------------------------------------------------------------ helpers

    def all_forbidden_substrings(self) -> tuple[str, ...]:
        seen: list[str] = []
        for sub in (*GLOBAL_FORBIDDEN_SUBSTRINGS, *self.forbidden_substrings):
            if sub and sub not in seen:
                seen.append(sub)
        return tuple(seen)

    def all_dangerous_bash_patterns(self) -> tuple[re.Pattern[str], ...]:
        return tuple(
            re.compile(p)
            for p in (*DANGEROUS_BASH_PATTERNS, *self.dangerous_bash_patterns)
        )

    # ---------------------------------------------------------- path checking

    def _resolved_pair(self, candidate: Path | str) -> tuple[Path, Path] | None:
        """Resolve *candidate* and return (resolved_candidate, resolved_root).

        Returns None on OSError. Resolution is non-strict so non-existent
        paths still get fully normalised (handles ``..`` traversals).
        """

        try:
            resolved = Path(candidate).expanduser()
            if not resolved.is_absolute():
                resolved = self.agent_root / resolved
            resolved = resolved.resolve(strict=False)
            root = self.agent_root.resolve(strict=False)
            return resolved, root
        except OSError:
            return None

    def is_inside(self, candidate: Path | str) -> bool:
        """Return True iff *candidate* resolves inside ``agent_root``.

        Uses ``Path.relative_to`` semantics rather than string ``startswith``,
        so e.g. an attacker-supplied ``<agent_root>_evil/foo`` is correctly
        rejected (string prefix match would have falsely accepted it).
        """

        pair = self._resolved_pair(candidate)
        if pair is None:
            return False
        resolved, root = pair
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            return False

    def is_writable_for_write(self, candidate: Path | str) -> bool:
        """Return True iff *candidate* is inside a writable subdir.

        Writable subdirs default to ``("work", "output")``; everything else
        in the workspace (README.md, plan.md, data/, agent_task_spec.json,
        meta_data.json, etc.) is read-only. ``plan.md`` is the one exception
        and is allowed by the plan-guard layer, not by this policy.
        """

        pair = self._resolved_pair(candidate)
        if pair is None:
            return False
        resolved, root = pair
        try:
            tail = resolved.relative_to(root)
        except ValueError:
            return False
        if not tail.parts:
            return False
        head = tail.parts[0]
        return head in tuple(self.writable_subdirs)

    # Backwards-compat alias kept for older callers / tests.
    is_writable = is_writable_for_write


    def find_forbidden(self, blob: str) -> str | None:
        """Return the first forbidden substring found in *blob*, or None."""

        if not blob:
            return None
        lower = blob.replace("\\", "/").lower()
        for sub in self.all_forbidden_substrings():
            needle = sub.replace("\\", "/").lower()
            if needle and needle in lower:
                return sub
        return None

    def find_dangerous_bash(self, command: str) -> str | None:
        """Return the first dangerous regex matched, or None."""

        if not command:
            return None
        for pattern in self.all_dangerous_bash_patterns():
            if pattern.search(command):
                return pattern.pattern
        return None

    # ------------------------------------------------------------- factories

    @classmethod
    def from_manifest(
        cls,
        agent_root: Path,
        manifest: Mapping[str, object],
        primary_output_rel: str,
    ) -> "WorkspacePolicy":
        """Build a policy from a registry manifest."""

        public_policy = dict(manifest.get("public_policy") or {})
        denylist = list(public_policy.get("public_data_denylist") or [])
        return cls(
            agent_root=Path(agent_root).resolve(),
            primary_output_rel=primary_output_rel,
            forbidden_substrings=tuple(str(item) for item in denylist),
        )


def normalise_path_arg(value: str) -> str:
    """Best-effort normalisation of a path-shaped tool argument."""

    if not value:
        return ""
    text = str(value).strip().strip('"').strip("'")
    text = text.replace("\\", "/")
    return str(PurePosixPath(text))


def collect_path_args(tool_input: Mapping[str, object]) -> List[str]:
    """Collect every value from a tool_input dict that *looks* like a path.

    The Claude SDK exposes tool inputs as plain dicts; rather than hardcoding
    the schema for every tool, we treat any string value as a candidate path
    and let the policy decide.
    """

    out: List[str] = []

    def walk(obj: object) -> None:
        if isinstance(obj, str):
            if obj:
                out.append(obj)
        elif isinstance(obj, Mapping):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(tool_input)
    return out
