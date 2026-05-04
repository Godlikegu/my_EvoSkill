"""Unit tests for the promote-or-reject gate.

We don't spin up Claude here; we inject a deterministic ``RunnerFn`` and
check the bookkeeping + decision logic under all four corner cases:

  * regression on a previously-passing task  -> REJECT_REGRESSION
  * no new pass anywhere                     -> REJECT_NO_NEW_PASS
  * at least one new pass, no regression     -> PROMOTE
  * all valid tasks already passing          -> REJECT_NO_NEW_PASS
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myevoskill.distill.transfer_validator import (
    VERDICT_PROMOTE,
    VERDICT_REJECT_NO_NEW_PASS,
    VERDICT_REJECT_REGRESSION,
    stamp_promotion,
    validate_skill,
    write_transfer_report,
)
from myevoskill.distill.universe import DistillUniverse


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def universe(tmp_path: Path) -> DistillUniverse:
    return DistillUniverse(
        repo_root=tmp_path,
        train_task_ids=("task_alpha",),
        valid_task_ids=("task_beta", "task_gamma"),
        model_slug="test_model",
    )


@pytest.fixture
def skill_pack(tmp_path: Path) -> Path:
    pack = tmp_path / "skills" / "wave_optics_v1"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text(
        "---\nname: wave_optics_v1\ndescription: test pack\n---\n# body\n",
        encoding="utf-8",
    )
    return pack


def _runner_factory(table):
    """Build a deterministic RunnerFn from {(task, with_skill): verdict}."""

    def _runner(task_id: str, with_skill: bool) -> str:
        return table[(task_id, with_skill)]

    return _runner


# ---------------------------------------------------------------- tests


def test_promote_when_skill_rescues_failing_task(universe, skill_pack):
    table = {
        ("task_beta", False): "FAIL",
        ("task_beta", True): "PASS",   # rescued
        ("task_gamma", False): "PASS",
        ("task_gamma", True): "PASS",  # held
    }
    report = validate_skill(
        universe=universe,
        skill_pack_dir=skill_pack,
        runner=_runner_factory(table),
    )
    assert report.verdict == VERDICT_PROMOTE
    s = report.summary()
    assert s["n_new_pass"] == 1
    assert s["n_regression"] == 0
    assert s["n_baseline_pass"] == 1
    assert s["n_plus_skill_pass"] == 2


def test_reject_on_regression(universe, skill_pack):
    table = {
        ("task_beta", False): "FAIL",
        ("task_beta", True): "PASS",   # rescued
        ("task_gamma", False): "PASS",
        ("task_gamma", True): "FAIL",  # broken!
    }
    report = validate_skill(
        universe=universe,
        skill_pack_dir=skill_pack,
        runner=_runner_factory(table),
    )
    assert report.verdict == VERDICT_REJECT_REGRESSION
    assert any("task_gamma" in r for r in report.rejection_reasons)


def test_reject_when_no_new_pass(universe, skill_pack):
    table = {
        ("task_beta", False): "FAIL",
        ("task_beta", True): "FAIL",   # still failing
        ("task_gamma", False): "PASS",
        ("task_gamma", True): "PASS",  # held but nothing new
    }
    report = validate_skill(
        universe=universe,
        skill_pack_dir=skill_pack,
        runner=_runner_factory(table),
    )
    assert report.verdict == VERDICT_REJECT_NO_NEW_PASS


def test_reject_when_all_already_passing(universe, skill_pack):
    table = {
        ("task_beta", False): "PASS",
        ("task_beta", True): "PASS",
        ("task_gamma", False): "PASS",
        ("task_gamma", True): "PASS",
    }
    report = validate_skill(
        universe=universe,
        skill_pack_dir=skill_pack,
        runner=_runner_factory(table),
    )
    # No "new" pass possible, even though nothing regressed.
    assert report.verdict == VERDICT_REJECT_NO_NEW_PASS


def test_skill_only_validation_runs_no_baseline(universe, skill_pack):
    calls = []

    def _runner(task_id: str, with_skill: bool) -> str:
        calls.append((task_id, with_skill))
        assert with_skill is True
        return "PASS"

    report = validate_skill(
        universe=universe,
        skill_pack_dir=skill_pack,
        runner=_runner,
        valid_task_ids=["task_beta"],
        compare_baseline=False,
    )

    assert calls == [("task_beta", True)]
    assert report.mode == "skill_only"
    assert report.verdict == VERDICT_PROMOTE
    assert report.comparisons[0].baseline_verdict == "SKIPPED"


def test_skill_only_validation_rejects_failed_skill(universe, skill_pack):
    report = validate_skill(
        universe=universe,
        skill_pack_dir=skill_pack,
        runner=lambda task_id, with_skill: "TIMEOUT",
        valid_task_ids=["task_beta"],
        compare_baseline=False,
    )

    assert report.mode == "skill_only"
    assert report.verdict == VERDICT_REJECT_NO_NEW_PASS
    assert any("task_beta" in r for r in report.rejection_reasons)


def test_refuses_train_tasks(universe, skill_pack):
    runner = _runner_factory({})  # never called
    with pytest.raises(PermissionError):
        validate_skill(
            universe=universe,
            skill_pack_dir=skill_pack,
            runner=runner,
            valid_task_ids=["task_alpha"],  # train!
        )


def test_requires_skill_md(tmp_path, universe):
    bogus_pack = tmp_path / "bogus"
    bogus_pack.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_skill(
            universe=universe,
            skill_pack_dir=bogus_pack,
            runner=_runner_factory({}),
        )


def test_write_report_and_stamp_promotion(universe, skill_pack, tmp_path):
    table = {
        ("task_beta", False): "FAIL",
        ("task_beta", True): "PASS",
        ("task_gamma", False): "PASS",
        ("task_gamma", True): "PASS",
    }
    report = validate_skill(
        universe=universe,
        skill_pack_dir=skill_pack,
        runner=_runner_factory(table),
    )
    out = write_transfer_report(report)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["verdict"] == VERDICT_PROMOTE
    assert payload["summary"]["n_new_pass"] == 1

    promo = stamp_promotion(report, model_slug="vendor2_claude_4_6_opus")
    assert promo is not None
    assert promo.exists()
    assert json.loads(promo.read_text())["model_slug"] == "vendor2_claude_4_6_opus"


def test_no_promotion_stamp_on_reject(universe, skill_pack):
    table = {
        ("task_beta", False): "PASS",
        ("task_beta", True): "FAIL",   # regression
        ("task_gamma", False): "PASS",
        ("task_gamma", True): "PASS",
    }
    report = validate_skill(
        universe=universe,
        skill_pack_dir=skill_pack,
        runner=_runner_factory(table),
    )
    assert report.verdict == VERDICT_REJECT_REGRESSION
    assert stamp_promotion(report, model_slug="m") is None
    assert not (skill_pack / "promotion.json").exists()
