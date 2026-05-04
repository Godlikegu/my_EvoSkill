"""Sanitizer contract tests.

Each test builds a synthetic skill directory and asserts the sanitizer
verdict. The sanitizer is *prevention*, not detection-after-the-fact:
any rule hit must reject the skill.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from myevoskill.distill.skill_sanitizer import SkillSanitizer


VALID_TIDS = ("reflection_ODT", "xray_ptychography_tike", "plane_wave_ultrasound", "usct_FWI")
TRAIN_TIDS = ("SSNP_ODT", "seismic_FWI_original")


def _make_skill(
    tmp_path: Path,
    *,
    name: str = "wave-optics-recon",
    description: str = "Use for wave-physics inverse reconstruction problems.",
    body: str = (
        "## Routes\n"
        "| public signal | algorithm route | required checks |\n"
        "| --- | --- | --- |\n"
        "| inverse problem | forward model route | inspect public spec |\n\n"
        "## Metric Diagnostic\n"
        "| failed metric signal | first diagnosis | next action | give-up signal |\n"
        "| --- | --- | --- | --- |\n"
        "| structure fails | check geometry | run probe | pattern stays wrong |\n\n"
        "## Anti-Patterns\n"
        "- Do not repeat a failed guard output.\n\n"
        "## Workflow\n1. Identify the forward operator A.\n"
    ),
    extras: dict[str, str] | None = None,
) -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    skill_md = (
        f"---\nname: {name}\ndescription: {description}\n---\n\n"
        f"# {name}\n\n{body}"
    )
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    if extras:
        for rel, content in extras.items():
            target = skill_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
    return skill_dir


@pytest.fixture
def sanitizer() -> SkillSanitizer:
    return SkillSanitizer(valid_task_ids=VALID_TIDS, train_task_ids=TRAIN_TIDS)


# --------------------------------------------------------------- happy path


def test_clean_skill_passes(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    skill = _make_skill(tmp_path)
    report = sanitizer.scan(skill)
    assert report.passed, report.findings


def test_skill_with_clean_scripts_passes(
    tmp_path: Path, sanitizer: SkillSanitizer
) -> None:
    skill = _make_skill(
        tmp_path,
        extras={
            "scripts/wiener.py": (
                "import numpy as np\n"
                "def generalized_wiener(y, otf, eps):\n"
                "    return np.conj(otf) * y / (np.abs(otf)**2 + eps)\n"
            ),
            "references/notes.md": "Generic wave-equation inverse problem notes.\n",
        },
    )
    report = sanitizer.scan(skill)
    assert report.passed, report.findings


# --------------------------------------------------------------- valid leaks


def test_rejects_valid_task_literal_in_skill_md(
    tmp_path: Path, sanitizer: SkillSanitizer
) -> None:
    skill = _make_skill(
        tmp_path, body="Apply this for reflection_ODT-like geometries.\n"
    )
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any("valid_task_literal" in f.rule for f in report.findings)


def test_rejects_valid_task_literal_in_scripts(
    tmp_path: Path, sanitizer: SkillSanitizer
) -> None:
    skill = _make_skill(
        tmp_path,
        extras={"scripts/helper.py": "# tuned for usct_FWI\nA = 1\n"},
    )
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any("valid_task_literal:usct_FWI" in f.rule for f in report.findings)


# --------------------------------------------------------------- train leaks


def test_rejects_train_task_literal(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    skill = _make_skill(tmp_path, body="Especially helpful for SSNP_ODT.\n")
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any("train_task_literal" in f.rule for f in report.findings)


# --------------------------------------------------------------- hardcoding


def test_rejects_hardcoded_shape(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    skill = _make_skill(tmp_path, body="Output shape is (12, 512, 512).\n")
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "hardcoded_shape" for f in report.findings)


def test_rejects_hardcoded_2d_shape(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    skill = _make_skill(tmp_path, body="Output shape is (512, 512).\n")
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "hardcoded_shape" for f in report.findings)


def test_rejects_metric_threshold(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    skill = _make_skill(tmp_path, body="Aim for ncc > 0.92.\n")
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "metric_threshold" for f in report.findings)


def test_rejects_specific_data_path(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    skill = _make_skill(tmp_path, body="Load `data/raw_data.npz`.\n")
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "specific_data_path" for f in report.findings)


def test_rejects_baseline_as_final_bias(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    skill = _make_skill(tmp_path, body=(
        "## Routes\n| public signal | algorithm route | required checks |\n| --- | --- | --- |\n| input | route | check |\n\n"
        "## Metric Diagnostic\n| failed metric signal | first diagnosis | next action | give-up signal |\n| --- | --- | --- | --- |\n| fail | inspect | act | stop |\n\n"
        "## Anti-Patterns\n- Avoid leaks.\n\n"
        "Treat the baseline as the final output when budget is tight.\n"
    ))
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "baseline_as_final" for f in report.findings)


def test_rejects_ground_truth_mention(
    tmp_path: Path, sanitizer: SkillSanitizer
) -> None:
    skill = _make_skill(tmp_path, body="Compare against the ground-truth array.\n")
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "ground_truth_mention" for f in report.findings)


def test_rejects_reference_src_leak(
    tmp_path: Path, sanitizer: SkillSanitizer
) -> None:
    skill = _make_skill(tmp_path, body="See `tasks/seismic_FWI_original/src/foo.py`.\n")
    report = sanitizer.scan(skill)
    assert not report.passed


# --------------------------------------------------------------- structural


def test_rejects_missing_skill_md(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    d = tmp_path / "broken"
    d.mkdir()
    (d / "scripts").mkdir()
    report = sanitizer.scan(d)
    assert not report.passed
    assert any(f.rule == "missing_SKILL_md" for f in report.findings)


def test_rejects_missing_frontmatter(
    tmp_path: Path, sanitizer: SkillSanitizer
) -> None:
    d = tmp_path / "no-fm"
    d.mkdir()
    (d / "SKILL.md").write_text("# no front matter\n\nbody\n", encoding="utf-8")
    report = sanitizer.scan(d)
    assert not report.passed
    assert any(f.rule == "missing_frontmatter" for f in report.findings)


def test_rejects_missing_required_skill_schema(
    tmp_path: Path, sanitizer: SkillSanitizer
) -> None:
    skill = _make_skill(tmp_path, body="## Workflow\nFree-form advice only.\n")
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "missing_routes_section" for f in report.findings)
    assert any(f.rule == "missing_metric_diagnostic_section" for f in report.findings)
    assert any(f.rule == "missing_anti_patterns_section" for f in report.findings)


def test_rejects_non_anthropic_frontmatter_fields(
    tmp_path: Path, sanitizer: SkillSanitizer
) -> None:
    skill = _make_skill(tmp_path)
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    text = text.replace("description: Use", "allowed-tools: Read, Write\ntrained_on_count: 2\ndescription: Use")
    (skill / "SKILL.md").write_text(text, encoding="utf-8")
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "unsupported_frontmatter_field" for f in report.findings)


def test_rejects_oversized_skill_md(tmp_path: Path, sanitizer: SkillSanitizer) -> None:
    body = "line\n" * 400
    skill = _make_skill(tmp_path, body=body)
    report = sanitizer.scan(skill)
    assert not report.passed
    assert any(f.rule == "skill_md_too_long" for f in report.findings)
