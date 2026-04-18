"""Structured run logging utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


class RunLogger:
    """Write per-run summary and text logs."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, run_id: str) -> Path:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_summary(self, run_dir: Path, summary: Dict[str, Any]) -> Path:
        path = Path(run_dir) / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def append_text_log(self, run_dir: Path, name: str, content: str) -> Path:
        path = Path(run_dir) / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")
        return path

    def write_json_artifact(self, run_dir: Path, name: str, payload: Any) -> Path:
        path = Path(run_dir) / name
        if is_dataclass(payload):
            payload = asdict(payload)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return path
