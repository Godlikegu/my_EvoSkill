"""Reject candidate skills that hardcode answers or leak valid-split content.

A candidate skill is a directory shaped like::

    <skill_id>/
        SKILL.md
        meta.json            (MyEvoSkill private metadata)
        references/...       (optional)
        scripts/...          (optional, executable helpers)

The sanitizer walks every text file under the skill dir and applies a
denylist of patterns. Any hit fails the skill. The intent is *prevention*,
not detection-after-the-fact: a failed sanitizer always rejects the skill.

We deliberately allow ``scripts/`` (per user decision) so skills can ship
reusable helpers (e.g. ``wiener_filter.py``); the sanitizer just refuses
content that would amount to copying answers from the train tasks or
referring to valid tasks at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# File extensions we treat as text and scan.
_TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh"}

# Maximum size constraints (defense against "skill = whole answer").
_MAX_SKILL_MD_LINES = 250
_MAX_TOTAL_BYTES = 200_000


@dataclass
class SanitizerFinding:
    file: str
    line: int
    rule: str
    snippet: str


@dataclass
class SanitizerReport:
    skill_id: str
    skill_dir: Path
    passed: bool
    findings: list[SanitizerFinding] = field(default_factory=list)

    def reason_summary(self) -> str:
        if self.passed:
            return "ok"
        return "; ".join(
            f"{f.rule}@{f.file}:{f.line}" for f in self.findings[:5]
        )


@dataclass
class SkillSanitizer:
    """Stateless sanitizer parameterised by the split.

    Parameters
    ----------
    valid_task_ids:
        Hard denylist - any literal occurrence in any text file rejects.
    train_task_ids:
        Soft denylist - we *do* want skills to be valid for train tasks but
        not to *name* them. Literal occurrences are rejected so skills must
        be phrased in domain-general language.
    """

    valid_task_ids: tuple[str, ...]
    train_task_ids: tuple[str, ...] = ()

    # ------------------------------------------------------------------ rules

    def _patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        rules: list[tuple[str, re.Pattern[str]]] = []
        # 1. Valid-task literal mentions (HARD).
        for tid in self.valid_task_ids:
            rules.append(
                (f"valid_task_literal:{tid}", re.compile(re.escape(tid), re.IGNORECASE))
            )
        # 2. Train-task literal mentions (also forbidden - keep skills
        #    domain-general, not task-specific).
        for tid in self.train_task_ids:
            rules.append(
                (f"train_task_literal:{tid}", re.compile(re.escape(tid), re.IGNORECASE))
            )
        # 3. Hardcoded array shapes such as (512, 512) or (12, 512, 512).
        rules.append(
            (
                "hardcoded_shape",
                re.compile(r"\(\s*\d{1,5}\s*,\s*\d{1,5}\s*(?:,\s*\d{1,5}\s*){0,3}\)"),
            )
        )
        # 4. Hardcoded metric thresholds, e.g. ncc > 0.92, nrmse < 0.1
        rules.append(
            (
                "metric_threshold",
                re.compile(
                    r"\b(ncc|nrmse|psnr|ssim|rmse|nmse)\b\s*[<>]=?\s*\d",
                    re.IGNORECASE,
                ),
            )
        )
        # 5. Specific data filenames the agent shouldn't be told to expect.
        rules.append(
            (
                "specific_data_path",
                re.compile(
                    r"data/(?!\*)[\w\-./]+\.(?:npz|npy|h5|hdf5|mat|tif|tiff|png)\b",
                    re.IGNORECASE,
                ),
            )
        )
        # 6. ground-truth mentions (forbidden by harness anyway).
        rules.append(("ground_truth_mention", re.compile(r"ground[\s\-_]*truth", re.IGNORECASE)))
        # 7. Reference path leaks.
        rules.append(("reference_src_leak", re.compile(r"tasks/[\w\-]+/(?:src|evaluation|notebooks)/")))
        # 8. Hidden file mentions.
        rules.append(
            (
                "hidden_file_mention",
                re.compile(r"\b(judge_adapter\.py|task_contract\.json)\b", re.IGNORECASE),
            )
        )
        # 9. Anti-transfer bias: a guard/baseline may be tried once, but a
        # skill must not canonize cheap train paths as final answers.
        rules.append(
            (
                "baseline_as_final",
                re.compile(
                    r"\b(default\s+final\s+output|treat\s+.*(?:baseline|guard).*"
                    r"(?:as\s+)?(?:the\s+)?final|(?:baseline|guard)\s+and\s+stop|"
                    r"keep\s+the\s+baseline\s+and\s+finish)\b",
                    re.IGNORECASE,
                ),
            )
        )
        rules.append(
            (
                "budget_reduce_epoch_bias",
                re.compile(r"\b(reduce|lower|shorten)\s+epochs?\s+if\s+budget\b", re.IGNORECASE),
            )
        )
        return rules

    # ------------------------------------------------------------------- scan

    def scan(self, skill_dir: Path) -> SanitizerReport:
        skill_dir = Path(skill_dir)
        report = SanitizerReport(
            skill_id=skill_dir.name, skill_dir=skill_dir, passed=True
        )

        if not (skill_dir / "SKILL.md").exists():
            report.passed = False
            report.findings.append(
                SanitizerFinding("<root>", 0, "missing_SKILL_md", "SKILL.md is required")
            )
            return report

        # Size check on SKILL.md.
        skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        nlines = skill_md.count("\n") + 1
        if nlines > _MAX_SKILL_MD_LINES:
            report.passed = False
            report.findings.append(
                SanitizerFinding(
                    "SKILL.md",
                    nlines,
                    "skill_md_too_long",
                    f"{nlines} > {_MAX_SKILL_MD_LINES}",
                )
            )

        # Front-matter sanity (Anthropic Skills require name + description).
        if not skill_md.lstrip().startswith("---"):
            report.passed = False
            report.findings.append(
                SanitizerFinding(
                    "SKILL.md",
                    1,
                    "missing_frontmatter",
                    "SKILL.md must start with YAML front-matter",
                )
            )
        else:
            fm = skill_md.split("---", 2)
            if len(fm) >= 3:
                head = fm[1]
                keys = []
                for raw_line in head.splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    m = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:", line)
                    if m:
                        keys.append(m.group(1))
                extra_keys = sorted(set(keys) - {"name", "description"})
                if extra_keys:
                    report.passed = False
                    report.findings.append(
                        SanitizerFinding(
                            "SKILL.md",
                            1,
                            "unsupported_frontmatter_field",
                            "only `name:` and `description:` are allowed in Anthropic skill front-matter",
                        )
                    )
                if "name:" not in head:
                    report.passed = False
                    report.findings.append(
                        SanitizerFinding(
                            "SKILL.md", 1, "missing_name_field", "no `name:` in front-matter"
                        )
                    )
                if "description:" not in head:
                    report.passed = False
                    report.findings.append(
                        SanitizerFinding(
                            "SKILL.md",
                            1,
                            "missing_description_field",
                            "no `description:` in front-matter",
                        )
                    )

        # Total-bytes ceiling.
        total_bytes = 0
        for f in skill_dir.rglob("*"):
            if f.is_file():
                try:
                    total_bytes += f.stat().st_size
                except OSError:
                    pass
        if total_bytes > _MAX_TOTAL_BYTES:
            report.passed = False
            report.findings.append(
                SanitizerFinding(
                    "<root>",
                    0,
                    "skill_total_too_large",
                    f"{total_bytes}B > {_MAX_TOTAL_BYTES}B",
                )
            )

        # Required high-level schema: keep the LLM output from becoming a
        # free-form essay that lacks routing and metric-driven transfer logic.
        required_sections = {
            "missing_routes_section": re.compile(r"^##\s+Routes\s*$", re.IGNORECASE | re.MULTILINE),
            "missing_metric_diagnostic_section": re.compile(
                r"^##\s+Metric Diagnostic\s*$", re.IGNORECASE | re.MULTILINE
            ),
            "missing_anti_patterns_section": re.compile(
                r"^##\s+Anti-Patterns\s*$", re.IGNORECASE | re.MULTILINE
            ),
        }
        for rule_name, pat in required_sections.items():
            if not pat.search(skill_md):
                report.passed = False
                report.findings.append(
                    SanitizerFinding(
                        "SKILL.md",
                        0,
                        rule_name,
                        "SKILL.md must include Routes, Metric Diagnostic, and Anti-Patterns sections",
                    )
                )

        # Walk all text files and apply the denylist.
        rules = self._patterns()
        for f in sorted(skill_dir.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            rel = f.relative_to(skill_dir).as_posix()
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                for rule_name, pat in rules:
                    if pat.search(line):
                        report.passed = False
                        report.findings.append(
                            SanitizerFinding(
                                file=rel,
                                line=i,
                                rule=rule_name,
                                snippet=line.strip()[:160],
                            )
                        )
        return report

    # ------------------------------------------------------------------- bulk

    def scan_pack(self, pack_dir: Path) -> list[SanitizerReport]:
        out: list[SanitizerReport] = []
        for sub in sorted(Path(pack_dir).iterdir()):
            if sub.is_dir() and (sub / "SKILL.md").exists():
                out.append(self.scan(sub))
        return out
