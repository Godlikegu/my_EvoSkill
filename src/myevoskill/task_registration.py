"""User-facing contract draft and task registration APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .registration_contract import (
    create_registration_contract_draft,
    load_task_registration_notes,
    register_confirmed_task,
)
from .models import ContractDraftResult, TaskRegistrationResult


def draft_task_contract(
    task_root: Path,
    *,
    output_root: Path,
    registration_input_path: Optional[Path] = None,
) -> ContractDraftResult:
    """Generate `registration_contract.draft.json` for one raw task."""

    return create_registration_contract_draft(
        task_root,
        output_root=output_root,
        registration_input_path=registration_input_path,
    )


def register_task(
    task_root: Path,
    *,
    output_root: Path,
) -> TaskRegistrationResult:
    """Register one confirmed contract into the manifest-driven control plane."""

    return register_confirmed_task(task_root, output_root=output_root)


def contract_draft_main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for registration contract draft generation."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate tasks/<task_id>/evaluation/registration_contract.draft.json "
            "from one raw task directory using registration_input.json and the Claude registration agent."
        )
    )
    parser.add_argument("--task-root", required=True, help="Path to the raw task directory")
    parser.add_argument(
        "--output-root",
        default="",
        help="Project root used for registry note mirrors (defaults to cwd).",
    )
    parser.add_argument(
        "--registration-input",
        default="",
        help=(
            "Optional path to tasks/<task_id>/evaluation/registration_input.json. "
            "Defaults to the task-local evaluation directory."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(args.output_root).resolve() if args.output_root else Path.cwd().resolve()
    result = draft_task_contract(
        Path(args.task_root),
        output_root=project_root,
        registration_input_path=Path(args.registration_input) if args.registration_input else None,
    )
    print(
        json.dumps(
            {
                "task_id": result.task_id,
                "draft_path": str(result.draft_path),
                "notes_path": str(result.notes_path),
                "missing_items": list(result.missing_items),
                "warnings": list(result.warnings),
                "attempt_count": result.attempt_count,
                "attempt_summaries": list(result.attempt_summaries),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for confirmed task registration."""

    parser = argparse.ArgumentParser(
        description=(
            "Register a confirmed tasks/<task_id>/evaluation/registration_contract.json "
            "into registry/tasks/ and generate a ready judge adapter."
        )
    )
    parser.add_argument("--task-root", required=True, help="Path to the raw task directory")
    parser.add_argument(
        "--output-root",
        default="",
        help="Project root where registry/tasks should be written (defaults to cwd).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(args.output_root).resolve() if args.output_root else Path.cwd().resolve()
    result = register_task(Path(args.task_root), output_root=project_root)
    print(
        json.dumps(
            {
                "task_id": result.task_id,
                "manifest_path": str(result.manifest_path),
                "judge_path": str(result.judge_path),
                "notes_path": str(result.notes_path),
                "missing_items": list(result.missing_items),
                "warnings": list(result.warnings),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


__all__ = [
    "contract_draft_main",
    "draft_task_contract",
    "load_task_registration_notes",
    "main",
    "register_task",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
