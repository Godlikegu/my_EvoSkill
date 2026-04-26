"""Commit F: targeted README sanitisation.

Only **ground-truth** sections / tables are stripped. Sections that describe
the metric, the evaluation channel, or the *reference solution methodology*
(``Solution``, ``Reference Solution``, ``Method``, ``Approach``, ...) are
preserved -- the agent legitimately reads those.
"""

from __future__ import annotations

from myevoskill.workspace.builder import _sanitise_readme


# ---------------------------------------------------------------------------
# headings
# ---------------------------------------------------------------------------


def test_strips_ground_truth_h2_section_with_body() -> None:
    text = "\n".join([
        "# Task",
        "Some preamble.",
        "",
        "## Ground Truth",
        "the secret answer is 42",
        "rho_gt = 1.337",
        "",
        "## Method",
        "use FBP with a Hann filter",
    ])
    out = _sanitise_readme(text, {})
    assert "Ground Truth" not in out
    assert "secret answer" not in out
    assert "1.337" not in out
    # the next section is preserved
    assert "## Method" in out
    assert "Hann filter" in out


def test_strips_ground_dash_truth_and_underscored_variants() -> None:
    for variant in ("Ground-Truth", "Ground_Truth", "GROUND TRUTH", "ground   truth"):
        text = f"# T\n## {variant}\nleak\n## Other\nkeep\n"
        out = _sanitise_readme(text, {})
        assert "leak" not in out, variant
        assert "keep" in out, variant


def test_tolerates_emoji_prefix_on_heading() -> None:
    text = "\n".join([
        "## 🟥 Ground-Truth (private)",
        "secret",
        "## Notes",
        "public",
    ])
    out = _sanitise_readme(text, {})
    assert "secret" not in out
    assert "public" in out


def test_strips_reference_answer_heading() -> None:
    text = "## Reference Answer\nrho=0.42\n## Metric\nuse RMSE\n"
    out = _sanitise_readme(text, {})
    assert "rho=0.42" not in out
    assert "## Metric" in out
    assert "RMSE" in out


# ---------------------------------------------------------------------------
# preserved sections (the regression guard the user explicitly asked for)
# ---------------------------------------------------------------------------


def test_does_not_strip_metric_or_evaluation_or_solution_sections() -> None:
    text = "\n".join([
        "# T",
        "## Evaluation",
        "We compute SSIM on the reconstruction.",
        "## Metric",
        "Threshold: 0.85.",
        "## Solution",
        "FBP + Hann filter, then median denoise.",
        "## Reference Solution",
        "Run main.py with default args.",
        "## Threshold",
        "0.85 SSIM",
        "## Judge",
        "see evaluation/judge_adapter.py for IO; do not read it.",
    ])
    out = _sanitise_readme(text, {})
    # All these should be intact -- they are NOT ground-truth.
    assert "## Evaluation" in out
    assert "## Metric" in out
    assert "## Solution" in out
    assert "## Reference Solution" in out
    assert "## Threshold" in out
    assert "## Judge" in out
    assert "FBP + Hann filter" in out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def test_strips_table_referencing_ground_truth() -> None:
    text = "\n".join([
        "## Files",
        "",
        "| name | path |",
        "| --- | --- |",
        "| measurement | data/y.npz |",
        "| ground_truth | private/gt.npz |",
        "",
        "More prose here.",
    ])
    out = _sanitise_readme(text, {})
    # entire table dropped (the gt row contaminates the whole thing)
    assert "y.npz" not in out
    assert "gt.npz" not in out
    assert "measurement" not in out
    assert "ground_truth" not in out
    # surrounding prose untouched
    assert "## Files" in out
    assert "More prose here." in out
    assert "removed by workspace sanitiser" in out


def test_keeps_metric_threshold_table() -> None:
    text = "\n".join([
        "## Metric",
        "",
        "| metric | threshold |",
        "| --- | --- |",
        "| ssim  | 0.85 |",
        "| psnr  | 28.0 |",
    ])
    out = _sanitise_readme(text, {})
    # generic metric/threshold tables are preserved -- no GT mention.
    assert "| ssim  | 0.85 |" in out
    assert "| psnr  | 28.0 |" in out


def test_strips_ground_truth_table_under_neutral_heading() -> None:
    """A table with a GT cell must be dropped even when nested under a
    non-GT heading (e.g. an ambiguous "Files" or "Layout" section)."""

    text = "\n".join([
        "## Layout",
        "",
        "| field | rel_path |",
        "| --- | --- |",
        "| Reference target | gt.npz |",
        "",
        "End.",
    ])
    out = _sanitise_readme(text, {})
    assert "gt.npz" not in out
    assert "Reference target" not in out
    # heading stays
    assert "## Layout" in out
    assert "End." in out


# ---------------------------------------------------------------------------
# legacy contract knobs still work
# ---------------------------------------------------------------------------


def test_remove_sections_extra_via_contract() -> None:
    text = "## Hidden\nsecret\n## Public\nok\n"
    out = _sanitise_readme(text, {"remove_sections": ["Hidden"]})
    assert "secret" not in out
    assert "## Public" in out


def test_preserve_sections_overrides_default_filter() -> None:
    # Even though "Ground Truth" matches the default GT filter, an
    # explicit preserve from the contract author wins.
    text = "## Ground Truth\nthe metric is 0.85 by convention\n## End\nx\n"
    out = _sanitise_readme(
        text,
        {"preserve_sections": ["Ground Truth"]},
    )
    assert "0.85 by convention" in out
    assert "## Ground Truth" in out


def test_remove_path_patterns_still_applied_per_line() -> None:
    text = "see ground_truth.npz for the answer\nokay line\n"
    out = _sanitise_readme(text, {"remove_path_patterns": [r"ground_truth\.npz"]})
    assert "ground_truth.npz" not in out
    assert "okay line" in out


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_empty_readme_returns_empty() -> None:
    assert _sanitise_readme("", {}) == ""


def test_readme_with_no_ground_truth_is_untouched_modulo_global_subs() -> None:
    text = "# Task\n\nDescribe the data layout and the metric.\n"
    out = _sanitise_readme(text, {})
    assert "Describe the data layout and the metric." in out
