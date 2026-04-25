"""Per-run isolated $HOME for the Claude CLI.

The Claude CLI (``claude-code`` Node.js binary, spawned by the Python
``claude_agent_sdk``) persists conversation history, MCP cache, plan
sidecars, and various other artefacts under the user's real
``~/.claude/`` directory. That is undesirable for our harness:

* it leaks state between tasks (and between concurrent runs),
* it bloats the user's home directory with hundreds of MB of caches,
* it makes deterministic cleanup hard.

The fix is simple: every harness run gets its own ``$HOME`` (pointing at
a freshly-created sandbox dir under
``MyEvoSkill/artifacts/sandboxes/<task_id>/<run_id>/home``), and the only
files copied across from the real ``~/.claude`` are the small
``settings.json`` / ``config.json`` files that hold the model name and
gateway URL. Everything else (``projects/``, ``sessions/``, ``plan/``,
caches) starts empty and is wiped on shutdown unless ``keep=True``.

This module is shared by both ``harness.runner`` (in-process SDK,
``ClaudeAgentOptions.env``) and ``concurrency.pool`` (subprocess,
``subprocess.run(env=...)``) so the cleanup story is the same.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Files that we *do* want to preserve from the user's real ~/.claude into the
# isolated sandbox. These are small JSON config files holding the model name
# and any 3rd-party gateway URL; without them the SDK falls back to its own
# default which often is wrong for users on a private endpoint.
_WHITELIST_CLAUDE_FILES = ("settings.json", "config.json")


@dataclass(frozen=True)
class IsolatedHome:
    """Handle returned by :func:`make_isolated_home`.

    Attributes
    ----------
    home_root:
        Absolute path that should be exported as ``HOME`` /
        ``USERPROFILE`` for the agent process. Always exists on disk.
    seeded_files:
        Tuple of file basenames that were successfully copied from the
        real ``~/.claude`` into ``home_root/.claude``. Useful for
        debugging - if this is empty the agent will use SDK defaults.
    """

    home_root: Path
    seeded_files: tuple[str, ...]


def default_sandbox_root(repo_root: Path, task_id: str, run_id: str) -> Path:
    """Compute the canonical sandbox path under the repo's artifacts dir."""

    return Path(repo_root) / "artifacts" / "sandboxes" / task_id / run_id / "home"


def make_isolated_home(
    *,
    repo_root: Path,
    task_id: str,
    run_id: str,
    sandbox_root: Path | None = None,
) -> IsolatedHome:
    """Materialise an isolated ``$HOME`` for one harness run.

    Parameters
    ----------
    repo_root:
        Repo root, used only when ``sandbox_root`` is not provided.
    task_id, run_id:
        Used to compute the default path. They are also recorded as the
        directory layout so concurrent runs of the same task never
        collide.
    sandbox_root:
        Optional explicit override. If given, the caller takes full
        responsibility for the path (e.g. tests passing a
        ``tempfile.mkdtemp`` location).
    """

    target = (
        Path(sandbox_root)
        if sandbox_root is not None
        else default_sandbox_root(repo_root, task_id, run_id)
    )
    target = target.resolve()
    if target.exists():
        # Should not happen with a fresh run_id, but be defensive: wipe.
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    seeded: list[str] = []
    real_home = Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or "")
    real_claude = real_home / ".claude" if str(real_home) else None
    if real_claude is not None and real_claude.is_dir():
        dest_claude = target / ".claude"
        dest_claude.mkdir(parents=True, exist_ok=True)
        for fname in _WHITELIST_CLAUDE_FILES:
            src_f = real_claude / fname
            if src_f.exists() and src_f.is_file():
                try:
                    shutil.copy2(src_f, dest_claude / fname)
                    seeded.append(fname)
                except OSError as exc:
                    logger.warning(
                        "could not seed %s into isolated HOME (%s)", fname, exc
                    )

    logger.debug(
        "isolated HOME at %s (seeded: %s)", target, ",".join(seeded) or "<none>"
    )
    return IsolatedHome(home_root=target, seeded_files=tuple(seeded))


def env_overrides_for(home: IsolatedHome) -> dict[str, str]:
    """Return the env-var dict that the spawned process must receive.

    Both POSIX (``HOME``) and Windows (``USERPROFILE``) are set so the
    same code works on either platform.
    """

    h = str(home.home_root)
    return {
        "HOME": h,
        "USERPROFILE": h,
        # Pin a Unicode-safe encoding for the Node child process - the
        # claude CLI emits emoji / non-ASCII tool output that otherwise
        # crashes on cp936 Windows shells.
        "PYTHONIOENCODING": "utf-8",
    }


def cleanup_isolated_home(home: IsolatedHome, *, keep: bool = False) -> None:
    """Wipe the sandbox directory unless ``keep`` is True.

    ``keep=True`` is meant for ``--keep-sandbox`` debugging - the
    operator can then inspect ``.claude/projects/...`` to see the raw
    SDK conversation history.
    """

    if keep:
        logger.info("keeping sandbox at %s (--keep-sandbox)", home.home_root)
        return
    try:
        shutil.rmtree(home.home_root, ignore_errors=True)
    except OSError as exc:
        logger.warning("could not remove sandbox %s (%s)", home.home_root, exc)
