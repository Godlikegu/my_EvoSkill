"""Workspace builder + sandbox policy for the harness.

The workspace is the only place the agent can read/write. It is built from a
registered task by copying the *public* slice of the task tree:

    workspace/
        README.md          (sanitised README from the task)
        meta_data.json     (task-level meta)
        data/              (only files in public_data_allowlist)
        work/              (writable scratch)
        output/            (writable; primary output goes here)

Anything else (ground_truth, evaluation/, src/, main.py, notebooks/, plan/) is
NEVER copied. The harness hooks reject any tool call that touches a path
outside the workspace or hits a forbidden substring.
"""

from .policy import WorkspacePolicy
from .builder import WorkspaceBuild, build_workspace

__all__ = ["WorkspacePolicy", "WorkspaceBuild", "build_workspace"]
