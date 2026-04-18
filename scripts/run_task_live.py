"""Thin wrapper for the manifest-driven live runner."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from myevoskill.live_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
