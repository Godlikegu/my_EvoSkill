"""Module CLI for registration contract draft generation."""

from __future__ import annotations

from typing import Optional, Sequence

from .task_registration import contract_draft_main, draft_task_contract, load_task_registration_notes


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Forward to the canonical draft CLI implementation."""

    return contract_draft_main(argv)


__all__ = [
    "draft_task_contract",
    "load_task_registration_notes",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
