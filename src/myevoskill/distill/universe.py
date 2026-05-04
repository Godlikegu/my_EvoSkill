"""Train/valid isolation boundary for skill distillation.

The distillation pipeline must *never* read content from valid-split tasks
(source code, trajectories, workspaces, registry notes). This module is
the single chokepoint that all distillation reads go through.

Usage::

    universe = DistillUniverse.from_split_file(repo_root, split_path)
    universe.bind_audit_log(audit_path)

    # OK
    text = universe.read_task_file("seismic_FWI_original", "src/main.py")

    # raises ValidationLeakError
    text = universe.read_task_file("reflection_ODT", "src/main.py")

The audit log is JSONL, one record per access (allowed or denied), and
``assert_no_valid_access`` is called at pipeline end as a defense in depth.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class ValidationLeakError(PermissionError):
    """Raised when distillation code tries to read a valid-split asset."""


@dataclass
class _AuditRecord:
    ts: float
    task_id: str
    rel_path: str
    kind: str  # "task_src" | "log" | "workspace" | "registry"
    allowed: bool
    reason: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": self.ts,
                "task_id": self.task_id,
                "rel_path": self.rel_path,
                "kind": self.kind,
                "allowed": self.allowed,
                "reason": self.reason,
            },
            ensure_ascii=False,
        )


@dataclass
class DistillUniverse:
    """Bounded view over the repository for distillation reads.

    Parameters
    ----------
    repo_root:
        Repository root, the ``MyEvoSkill/`` directory.
    train_task_ids:
        Tasks the pipeline is allowed to read.
    valid_task_ids:
        Tasks the pipeline is forbidden to read.
    model_slug:
        The agent model whose logs/workspaces this distillation runs on.
    """

    repo_root: Path
    train_task_ids: tuple[str, ...]
    valid_task_ids: tuple[str, ...]
    model_slug: str
    _audit_path: Path | None = field(default=None, init=False)
    _records: list[_AuditRecord] = field(default_factory=list, init=False)

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_split_file(cls, repo_root: Path, split_path: Path) -> "DistillUniverse":
        spec = json.loads(Path(split_path).read_text(encoding="utf-8"))
        return cls(
            repo_root=Path(repo_root).resolve(),
            train_task_ids=tuple(spec["train"]),
            valid_task_ids=tuple(spec["valid"]),
            model_slug=str(spec["model_slug"]),
        )

    # ------------------------------------------------------------ audit

    def bind_audit_log(self, audit_path: Path) -> None:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate at bind time so each pipeline run starts fresh.
        path.write_text("", encoding="utf-8")
        self._audit_path = path

    def _audit(self, rec: _AuditRecord) -> None:
        self._records.append(rec)
        if self._audit_path is not None:
            with self._audit_path.open("a", encoding="utf-8") as fh:
                fh.write(rec.to_json() + "\n")

    def all_records(self) -> tuple[_AuditRecord, ...]:
        return tuple(self._records)

    # ------------------------------------------------------------ checks

    def is_train(self, task_id: str) -> bool:
        return task_id in self.train_task_ids

    def is_valid(self, task_id: str) -> bool:
        return task_id in self.valid_task_ids

    def assert_train(self, task_id: str, kind: str, rel_path: str) -> None:
        if self.is_valid(task_id):
            self._audit(
                _AuditRecord(
                    ts=time.time(),
                    task_id=task_id,
                    rel_path=rel_path,
                    kind=kind,
                    allowed=False,
                    reason="task is in valid split",
                )
            )
            raise ValidationLeakError(
                f"Distillation attempted to read valid-split task {task_id!r} "
                f"(rel={rel_path!r}, kind={kind!r}). This is forbidden."
            )
        if not self.is_train(task_id):
            self._audit(
                _AuditRecord(
                    ts=time.time(),
                    task_id=task_id,
                    rel_path=rel_path,
                    kind=kind,
                    allowed=False,
                    reason="task not in any split",
                )
            )
            raise ValidationLeakError(
                f"Distillation attempted to read task {task_id!r} which is not "
                "registered in either the train or valid split."
            )

    def assert_no_valid_access(self) -> None:
        """Defense-in-depth tail check.

        Call at the very end of the distillation pipeline. If any record in
        the audit references a valid task, raise.
        """

        violations = [
            r
            for r in self._records
            if r.task_id in self.valid_task_ids and r.allowed
        ]
        if violations:
            raise ValidationLeakError(
                f"audit_log shows {len(violations)} accesses to valid-split tasks; "
                f"first offender: task={violations[0].task_id} "
                f"path={violations[0].rel_path}"
            )

    # ------------------------------------------------------------ readers

    def task_dir(self, task_id: str) -> Path:
        """Path to ``tasks/<task_id>/`` (reference source dir)."""
        self.assert_train(task_id, "task_src", "<dir>")
        return (self.repo_root.parent / "tasks" / task_id).resolve()

    def read_task_file(self, task_id: str, rel: str) -> str:
        self.assert_train(task_id, "task_src", rel)
        target = (self.repo_root.parent / "tasks" / task_id / rel).resolve()
        # Confine to the task dir.
        td = (self.repo_root.parent / "tasks" / task_id).resolve()
        if not str(target).startswith(str(td)):
            raise ValidationLeakError(f"path escape: {target} not under {td}")
        text = target.read_text(encoding="utf-8", errors="replace")
        self._audit(
            _AuditRecord(
                ts=time.time(),
                task_id=task_id,
                rel_path=rel,
                kind="task_src",
                allowed=True,
                reason="train task_src read",
            )
        )
        return text

    def task_log_root(self, task_id: str) -> Path:
        """Path to ``artifacts/logs/<model_slug>/<task_id>/``."""
        self.assert_train(task_id, "log", "<dir>")
        return (
            self.repo_root / "artifacts" / "logs" / self.model_slug / task_id
        ).resolve()

    def list_runs(self, task_id: str) -> list[Path]:
        root = self.task_log_root(task_id)
        if not root.exists():
            return []
        return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("run-"))

    def read_log_file(self, task_id: str, run_dir: Path, rel: str) -> str:
        self.assert_train(task_id, "log", f"{run_dir.name}/{rel}")
        target = (run_dir / rel).resolve()
        if not str(target).startswith(str(self.task_log_root(task_id))):
            raise ValidationLeakError(f"log path escape: {target}")
        text = target.read_text(encoding="utf-8", errors="replace")
        self._audit(
            _AuditRecord(
                ts=time.time(),
                task_id=task_id,
                rel_path=f"{run_dir.name}/{rel}",
                kind="log",
                allowed=True,
                reason="train log read",
            )
        )
        return text

    def workspace_root(self, task_id: str) -> Path:
        self.assert_train(task_id, "workspace", "<dir>")
        return (
            self.repo_root / "artifacts" / "workspaces" / self.model_slug / task_id
        ).resolve()

    def read_workspace_file(self, task_id: str, run_id: str, rel: str) -> str:
        self.assert_train(task_id, "workspace", f"{run_id}/{rel}")
        root = self.workspace_root(task_id)
        target = (root / run_id / rel).resolve()
        run_root = (root / run_id).resolve()
        if not str(target).startswith(str(run_root)):
            raise ValidationLeakError(f"workspace path escape: {target}")
        text = target.read_text(encoding="utf-8", errors="replace")
        self._audit(
            _AuditRecord(
                ts=time.time(),
                task_id=task_id,
                rel_path=f"{run_id}/{rel}",
                kind="workspace",
                allowed=True,
                reason="train workspace read",
            )
        )
        return text


# ----------------------------------------------------------------- helpers


def load_split(split_path: Path) -> Mapping[str, object]:
    """Convenience: parse split JSON without constructing a Universe."""
    return json.loads(Path(split_path).read_text(encoding="utf-8"))
