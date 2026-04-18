# Dev Environment

## Goal

Provide a dedicated conda environment for developing and testing MyEvoSkill.

## Requirements

The dev environment should include at least:

- Python 3.10 or newer
- Node.js / npm for Claude Code CLI installation when using the Claude SDK
  workspace backend
- pytest
- pyyaml
- pydantic
- optional inspect_ai
- bridge/helper dependencies used by MyEvoSkill control-plane code

`inspect-ai` is published on PyPI and currently requires Python >= 3.10, so the
development environment must not be pinned to Python 3.9 if Inspect integration
is desired.

Task-specific scientific dependencies stay in task runtime environments.

For the current Claude SDK workspace path, the Python package alone is not
enough for live runs. The active machine also needs the Claude Code CLI
available on `PATH`.

## Setup Files

- `environment.yml`
- `scripts/create_dev_env.sh`
- `scripts/print_env_info.py`

## Policy

- run MyEvoSkill development and tests inside this dedicated environment
- do not treat the dev environment as the task runtime environment
- a repo-local conda prefix is acceptable when writing to the global conda env
  path is undesirable
- the repo-local `./.conda_env` may legitimately include optional bridge
  dependencies such as `inspect_ai`
- the repo-local `./.conda_env` may also include `claude-agent-sdk`, but live
  Claude SDK runs additionally require the external `claude` CLI
- tests must therefore not assume optional packages are absent; missing-package
  paths should be exercised via injected dependency detection or monkeypatching
- default regression tests may run on Linux without Claude Code CLI, but the
  live Claude SDK smoke test should be treated as host-dependent

## Claude SDK Live Prerequisites

To run the live Claude SDK smoke path, the active host should provide:

- `claude-agent-sdk` in the Python environment
- Claude Code CLI installed and visible as `claude`
- `MYEVOSKILL_CLAUDE_API_KEY`
- `ANTHROPIC_API_KEY`
- optional `MYEVOSKILL_CLAUDE_MODEL`

Recommended shell setup:

```bash
export MYEVOSKILL_CLAUDE_API_KEY='...'
export ANTHROPIC_API_KEY="$MYEVOSKILL_CLAUDE_API_KEY"
# Optional: set a specific Claude model alias or full model name.
# If omitted, Claude Code CLI uses its current default model.
# export MYEVOSKILL_CLAUDE_MODEL='sonnet'
```

## Host Guidance

- Linux development without Claude Code CLI is acceptable for default unit and
  integration regression work.
- Windows is currently the preferred fallback host for live Claude SDK task
  runs when the Linux server cannot install Claude Code CLI.
- Documentation and test commands should make this split explicit so developers
  do not confuse "default regression environment" with "live SDK execution
  environment".
