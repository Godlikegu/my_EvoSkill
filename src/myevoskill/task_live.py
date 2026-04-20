"""Module CLI for manifest-driven live task execution."""

from __future__ import annotations

from typing import Sequence

from .live_runner import load_registered_manifest, main as live_main, run_registered_task_live


def main(argv: Sequence[str] | None = None) -> int:
    """Forward to the canonical live-runner CLI implementation."""

    return live_main(argv)


__all__ = [
    "load_registered_manifest",
    "main",
    "run_registered_task_live",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
