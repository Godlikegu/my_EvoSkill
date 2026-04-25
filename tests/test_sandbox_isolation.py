"""Unit tests for myevoskill.harness.sandbox.

These tests don't spawn the Claude SDK; they only verify the contract
of :func:`make_isolated_home`, :func:`env_overrides_for`, and
:func:`cleanup_isolated_home` so that runner.py and pool.py can rely
on the same primitives.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from myevoskill.harness.sandbox import (
    cleanup_isolated_home,
    default_sandbox_root,
    env_overrides_for,
    make_isolated_home,
)


def test_default_sandbox_root_layout(tmp_path: Path) -> None:
    p = default_sandbox_root(tmp_path, "mytask", "run-x")
    # Must live under <repo>/artifacts/sandboxes/<task>/<run>/home
    assert p == tmp_path / "artifacts" / "sandboxes" / "mytask" / "run-x" / "home"


def test_make_isolated_home_creates_directory(tmp_path: Path) -> None:
    home = make_isolated_home(
        repo_root=tmp_path, task_id="mytask", run_id="run-1"
    )
    assert home.home_root.exists()
    assert home.home_root.is_dir()
    # Path is canonical - no symlinks, absolute.
    assert home.home_root.is_absolute()


def test_make_isolated_home_explicit_root(tmp_path: Path) -> None:
    custom = tmp_path / "custom_sandbox"
    home = make_isolated_home(
        repo_root=tmp_path,
        task_id="t",
        run_id="r",
        sandbox_root=custom,
    )
    assert home.home_root == custom.resolve()


def test_make_isolated_home_wipes_existing(tmp_path: Path) -> None:
    """If the target directory already exists with stale data, it must be
    wiped before the new sandbox is created. Otherwise we would inherit a
    previous run's claude history."""

    target = tmp_path / "sb"
    target.mkdir()
    (target / "stale.txt").write_text("leftover", encoding="utf-8")
    home = make_isolated_home(
        repo_root=tmp_path, task_id="t", run_id="r", sandbox_root=target
    )
    assert home.home_root.exists()
    assert not (home.home_root / "stale.txt").exists()


def test_env_overrides_sets_home_and_userprofile(tmp_path: Path) -> None:
    home = make_isolated_home(
        repo_root=tmp_path, task_id="t", run_id="r"
    )
    env = env_overrides_for(home)
    assert env["HOME"] == str(home.home_root)
    assert env["USERPROFILE"] == str(home.home_root)
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_cleanup_isolated_home_removes_dir(tmp_path: Path) -> None:
    home = make_isolated_home(
        repo_root=tmp_path, task_id="t", run_id="r"
    )
    # Drop a fake claude session into it to mimic real usage.
    (home.home_root / ".claude").mkdir(exist_ok=True)
    (home.home_root / ".claude" / "session.json").write_text("x", encoding="utf-8")

    cleanup_isolated_home(home, keep=False)
    assert not home.home_root.exists()


def test_cleanup_isolated_home_keep_flag_preserves_dir(tmp_path: Path) -> None:
    home = make_isolated_home(
        repo_root=tmp_path, task_id="t", run_id="r"
    )
    cleanup_isolated_home(home, keep=True)
    assert home.home_root.exists()


def test_seeding_whitelist_only_copies_known_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seeder must copy only the whitelisted JSON files - never
    ``projects/``, ``sessions/``, plans, or caches - so we cannot leak
    prior conversation history into a fresh task."""

    fake_home = tmp_path / "fake_real_home"
    real_claude = fake_home / ".claude"
    real_claude.mkdir(parents=True)
    # Whitelisted: should be copied.
    (real_claude / "settings.json").write_text(
        '{"model": "test-model"}', encoding="utf-8"
    )
    (real_claude / "config.json").write_text("{}", encoding="utf-8")
    # Not whitelisted: must be skipped.
    (real_claude / "projects").mkdir()
    (real_claude / "projects" / "leak.json").write_text(
        "PRIOR_CONVERSATION", encoding="utf-8"
    )
    (real_claude / "sessions").mkdir()
    (real_claude / "sessions" / "old.json").write_text("PRIOR", encoding="utf-8")

    # Point HOME / USERPROFILE at our fake real home.
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    sandbox_target = tmp_path / "sandbox_home"
    home = make_isolated_home(
        repo_root=tmp_path,
        task_id="t",
        run_id="r",
        sandbox_root=sandbox_target,
    )

    sandbox_claude = home.home_root / ".claude"
    assert sandbox_claude.exists()
    # Whitelisted files copied.
    assert (sandbox_claude / "settings.json").exists()
    assert (sandbox_claude / "config.json").exists()
    assert "settings.json" in home.seeded_files
    assert "config.json" in home.seeded_files
    # Forbidden subdirs absolutely must not be present.
    assert not (sandbox_claude / "projects").exists(), (
        "projects/ leaked into isolated HOME - prior conversation hack vector"
    )
    assert not (sandbox_claude / "sessions").exists(), (
        "sessions/ leaked into isolated HOME"
    )


def test_two_runs_get_distinct_homes(tmp_path: Path) -> None:
    """Concurrent runs of the same task_id must not share a HOME."""

    h1 = make_isolated_home(repo_root=tmp_path, task_id="same", run_id="r1")
    h2 = make_isolated_home(repo_root=tmp_path, task_id="same", run_id="r2")
    assert h1.home_root != h2.home_root
    cleanup_isolated_home(h1, keep=False)
    cleanup_isolated_home(h2, keep=False)
