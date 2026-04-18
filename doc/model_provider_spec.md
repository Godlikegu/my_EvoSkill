# Model Provider Specification

## Goal

Separate agent/runtime execution from the underlying LLM API provider.

## Interface

Each model provider implements:

`describe() -> dict`

Providers may also expose:

- `resolve_api_key(env=None) -> str`
- `safe_log_config() -> dict`
- `build_request(messages, env=None) -> (url, headers, payload)` for
  HTTP-backed chat providers

## Built-in Providers

- `ClaudeSDKAdapter`
  Formal model-backed execution provider for the current workspace harness.
- `OpenAICompatibleAdapter`
  Legacy config helper kept only for compatibility paths such as older bridge tests.
- `AnthropicCompatibleAdapter`
  Reserved compatibility/config surface, not a formal execution backend today.
- `CustomHTTPAdapter`
  Reserved compatibility/config surface, not a formal execution backend today.

## ModelConfig

`ModelConfig` includes:

- `provider_name`
- `base_url`
- `api_key_env`
- `model_name`
- `timeout`
- `max_tokens`
- `temperature`
- `extra_headers`

## Usage

Executors such as Claude, OpenHands, and Inspect bridge adapters may reference
the same `ModelConfig`, which lets the system support user-supplied LLM API
endpoints without redesigning the executor interface.

Security rules:

- only `api_key_env` is stored in config and logs
- the resolved API key must come from environment variables
- real secret values must never be written to repo files or run logs
- live external-network tests must inject the secret through the current shell
  environment only

## Claude SDK Notes

`ClaudeSDKAdapter` is the provider description used by the primary
`ClaudeWorkspaceAdapter` path when `provider_name="claude-sdk"`.

Current expectations:

- `api_key_env` should usually be `MYEVOSKILL_CLAUDE_API_KEY`
- if `MYEVOSKILL_CLAUDE_API_KEY` is unset, resolution falls back to
  `CLAUDE_API_KEY`, then `ANTHROPIC_API_KEY`
- the SDK subprocess environment mirrors the resolved secret into
  `MYEVOSKILL_CLAUDE_API_KEY`, `CLAUDE_API_KEY`, and `ANTHROPIC_API_KEY`
- the host should also export `ANTHROPIC_API_KEY` for Claude Code CLI
- `model_name` may be omitted in favor of SDK defaults, but live tests may set
  `MYEVOSKILL_CLAUDE_MODEL`
- Claude SDK execution is host-dependent because it requires the external
  `claude` CLI in addition to the Python package
