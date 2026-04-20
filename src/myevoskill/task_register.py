"""Module CLI for confirmed task registration."""

from __future__ import annotations

from typing import Optional, Sequence

from .task_registration import load_task_registration_notes, main as register_main, register_task


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Forward to the canonical confirmed-registration CLI implementation."""

    return register_main(argv)


__all__ = [
    "load_task_registration_notes",
    "main",
    "register_task",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
