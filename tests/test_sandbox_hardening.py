"""Regression tests for the hardened sandbox.

Each test corresponds to one concrete "hack vector" we observed (or
reasonably anticipated) the agent attempting:

  H1  read ground-truth via ``cat ../tasks/<id>/ground_truth/...``
  H2  write outside writable subdirs via ``Write file_path=README.md``
  H3  smuggle a write through ``python -c "open('../foo','w')"``
  H4  prefix-confusion: ``<agent_root>_evil/foo`` (``startswith`` bug)
  H5  command substitution / ``eval`` to obtain a path at runtime

These are *unit* tests against ``WorkspacePolicy`` + ``parse_bash_writes``
+ the ``_check_paths`` / Bash branch of the hooks. The full hook closure
requires a Claude SDK context object, so we test the inner pieces
directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from myevoskill.workspace.bash_parser import parse_bash_writes
from myevoskill.workspace.policy import WorkspacePolicy


@pytest.fixture
def policy(tmp_path: Path) -> WorkspacePolicy:
    root = tmp_path / "ws"
    (root / "work").mkdir(parents=True)
    (root / "output").mkdir()
    return WorkspacePolicy(
        agent_root=root.resolve(),
        primary_output_rel="output/result.npy",
    )


# ---------------------------------------------------------------------------
# H1: substring deny on ground_truth references
# ---------------------------------------------------------------------------
def test_h1_substring_deny_blocks_ground_truth(policy: WorkspacePolicy) -> None:
    assert policy.find_forbidden("../tasks/mri_grappa/ground_truth/x.npy") == "ground_truth"
    assert policy.find_forbidden("evaluation/judge_adapter.py") is not None


# ---------------------------------------------------------------------------
# H2: Write/Edit can't drop files in workspace root or read-only subdirs
# ---------------------------------------------------------------------------
def test_h2_root_write_is_not_writable(policy: WorkspacePolicy) -> None:
    # README.md sits in the workspace root: inside, but not writable.
    target = policy.agent_root / "README.md"
    assert policy.is_inside(target) is True
    assert policy.is_writable_for_write(target) is False
    # work/foo.py is inside *and* writable.
    target_ok = policy.agent_root / "work" / "foo.py"
    assert policy.is_writable_for_write(target_ok) is True
    # data/ is read-only (not in default writable_subdirs).
    target_data = policy.agent_root / "data" / "x.npy"
    assert policy.is_writable_for_write(target_data) is False


# ---------------------------------------------------------------------------
# H3: python -c "open('...', 'w')" must surface the write target
# ---------------------------------------------------------------------------
def test_h3_python_dash_c_open_write_detected() -> None:
    cmd = "python -c \"open('output/x.txt', 'w').write('hi')\""
    access = parse_bash_writes(cmd)
    assert "output/x.txt" in access.writes
    assert access.dynamic == []


def test_h3_python_dash_c_open_read_outside_detected() -> None:
    cmd = "python -c \"open('../tasks/mri_grappa/ground_truth/k.npy').read()\""
    access = parse_bash_writes(cmd)
    assert "../tasks/mri_grappa/ground_truth/k.npy" in access.reads


def test_h3_json_load_open_literal_is_not_dynamic() -> None:
    cmd = "python -c \"import json; meta = json.load(open('data/meta_data.json')); print(meta)\""
    access = parse_bash_writes(cmd)
    assert "data/meta_data.json" in access.reads
    assert access.dynamic == []


def test_h3_json_load_file_object_is_not_dynamic() -> None:
    cmd = "python -c \"import json; f = open('data/meta_data.json'); meta = json.load(f); f.close()\""
    access = parse_bash_writes(cmd)
    assert "data/meta_data.json" in access.reads
    assert access.dynamic == []


def test_h3_python_dash_c_dynamic_path_marked_dynamic() -> None:
    cmd = "python -c \"import sys; open(sys.argv[1], 'w').write('x')\""
    access = parse_bash_writes(cmd)
    assert access.dynamic, "non-literal open() must be marked dynamic"


def test_h3_numpy_load_dynamic_path_marked_dynamic() -> None:
    cmd = "python -c \"import numpy as np, sys; np.load(sys.argv[1])\""
    access = parse_bash_writes(cmd)
    assert access.dynamic, "non-literal np.load() must be marked dynamic"


def test_h3_pathlib_write_text_detected() -> None:
    cmd = "python -c \"from pathlib import Path; Path('work/a.txt').write_text('x')\""
    access = parse_bash_writes(cmd)
    assert "work/a.txt" in access.writes


# ---------------------------------------------------------------------------
# H4: startswith-prefix confusion is now rejected by relative_to logic
# ---------------------------------------------------------------------------
def test_h4_prefix_sibling_is_outside(policy: WorkspacePolicy, tmp_path: Path) -> None:
    sibling = tmp_path / "ws_evil" / "secret.txt"
    sibling.parent.mkdir()
    assert policy.is_inside(sibling) is False
    assert policy.is_inside(policy.agent_root / "work" / "ok.py") is True


def test_h4_dotdot_traversal_is_outside(policy: WorkspacePolicy) -> None:
    escape = policy.agent_root / "work" / ".." / ".." / "etc" / "passwd"
    assert policy.is_inside(escape) is False


# ---------------------------------------------------------------------------
# H5: command substitution / eval / heredoc dynamic paths are blocked
# ---------------------------------------------------------------------------
def test_h5_command_substitution_marked_dynamic() -> None:
    cmd = "cat $(ls work)"
    access = parse_bash_writes(cmd)
    assert access.dynamic
    assert "command/process substitution" in access.dynamic[0]


def test_h5_eval_blocked_outright() -> None:
    cmd = "eval \"echo $HOME > /tmp/x\""
    access = parse_bash_writes(cmd)
    assert access.dynamic and "eval" in access.dynamic[0]


def test_h5_redirect_to_root_detected() -> None:
    cmd = "echo hi > /tmp/leak"
    access = parse_bash_writes(cmd)
    assert "/tmp/leak" in access.writes


def test_h5_tee_append_detected() -> None:
    cmd = "echo hi | tee -a output/log.txt"
    access = parse_bash_writes(cmd)
    assert "output/log.txt" in access.writes


# ---------------------------------------------------------------------------
# Sanity: dangerous-bash regex still active.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd",
    [
        "sudo rm -rf /",
        "curl -O http://x/y",
        "pip install requests",
        "cd /etc",
        "cd C:\\Windows",
        "cd ..",
        "cd ../x",
        "echo ok && cd ..",
        "cd ../../..",
    ],
)
def test_dangerous_bash_patterns_match(policy: WorkspacePolicy, cmd: str) -> None:
    assert policy.find_dangerous_bash(cmd) is not None
