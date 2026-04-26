"""Commit D: regression tests for the bash-only / no-pyproject packaging.

These are pure offline checks against the repository layout. They guarantee
that we do not silently re-introduce ``pyproject.toml`` or
``pip install -e .`` workflows, and that the pytest configuration + helper
scripts continue to set ``PYTHONPATH`` so ``python -m myevoskill.cli`` works
without an editable install.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# pyproject.toml must not exist
# ---------------------------------------------------------------------------


def test_no_pyproject_toml() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    assert not pyproject.exists(), (
        "MyEvoSkill is intentionally not pip-installable; "
        f"{pyproject} must not be re-introduced. Use scripts/setup_env.sh + "
        "PYTHONPATH=src instead."
    )


# ---------------------------------------------------------------------------
# pytest.ini is the canonical test config
# ---------------------------------------------------------------------------


def test_pytest_ini_exists_and_is_well_formed() -> None:
    cfg_path = REPO_ROOT / "pytest.ini"
    assert cfg_path.exists(), f"missing {cfg_path}"

    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    assert cfg.has_section("pytest"), "pytest.ini missing [pytest] section"
    section = cfg["pytest"]

    # testpaths / pythonpath must be present and point at the in-tree dirs.
    assert section.get("testpaths", "").strip() == "tests"
    assert section.get("pythonpath", "").strip() == "src"

    # external_network marker must remain registered so smoke runs do not
    # warn-as-error in -W mode.
    markers = section.get("markers", "")
    assert "external_network" in markers


# ---------------------------------------------------------------------------
# Helper scripts must export PYTHONPATH and not rely on `pip install -e`
# ---------------------------------------------------------------------------


SCRIPTS_THAT_NEED_PYTHONPATH = [
    "scripts/register_task.sh",
    "scripts/run_task.sh",
    "scripts/run_smoke_three.sh",
]


@pytest.mark.parametrize("rel_path", SCRIPTS_THAT_NEED_PYTHONPATH)
def test_helper_scripts_export_pythonpath(rel_path: str) -> None:
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert "PYTHONPATH" in text, (
        f"{rel_path} must export PYTHONPATH so `python -m myevoskill.cli` "
        "resolves without `pip install -e .`."
    )
    # Must reference the in-tree src dir (with either ${REPO_ROOT} or a
    # relative form). We accept any string that contains '/src' near
    # PYTHONPATH.
    assert "src" in text


def _strip_shell_comments(text: str) -> str:
    """Drop ``#``-comments so prose mentioning forbidden commands does not
    trip the executable-line scan. Shebangs are preserved (start of line)."""
    out: list[str] = []
    for idx, line in enumerate(text.splitlines()):
        if idx == 0 and line.startswith("#!"):
            out.append(line)
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # strip trailing inline comment when not inside quotes (rough but
        # sufficient for our hand-written scripts).
        if "#" in line and "'" not in line and '"' not in line:
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def test_setup_env_does_not_pip_install_repo() -> None:
    text = (REPO_ROOT / "scripts" / "setup_env.sh").read_text(encoding="utf-8")
    code = _strip_shell_comments(text)
    forbidden = [
        "pip install -e .",
        'pip install -e "${REPO_ROOT}"',
        "pip install -e ${REPO_ROOT}",
    ]
    for needle in forbidden:
        assert needle not in code, (
            f"setup_env.sh must not run `{needle}` in an executable line; "
            "the harness is run via `python -m myevoskill.cli` with "
            "PYTHONPATH=src."
        )


# ---------------------------------------------------------------------------
# No orphan / dead modules under src/myevoskill
# ---------------------------------------------------------------------------


ORPHAN_MODULE_NAMES = {
    "task_registration.py",
    "task_live.py",
    "live_runner.py",
}


def test_no_orphan_modules_in_src() -> None:
    pkg_root = REPO_ROOT / "src" / "myevoskill"
    found = {p.name for p in pkg_root.iterdir() if p.is_file()}
    leftovers = ORPHAN_MODULE_NAMES & found
    assert not leftovers, (
        f"orphan / pre-cleanup modules still present under src/myevoskill: "
        f"{sorted(leftovers)}; the canonical entry point is "
        "`python -m myevoskill.cli`."
    )
