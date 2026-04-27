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

- `config/llm.yaml` is local-only and ignored by git
- `config/llm.example.yaml` documents the shareable schema
- inline `api_key` is tolerated only for local compatibility and must be
  redacted from every log, summary, probe result, and CLI message
- `api_key_env` is preferred over inline `api_key` when both are present
- real secret values must never be written to repo files or run logs
- live external-network tests must inject the secret through the current shell
  environment only

## Local `llm.yaml`

The local config shape is:

- `models.<model_id>.api_type`
- `models.<model_id>.base_url`
- `models.<model_id>.api_key_env` or local-only `api_key`
- `models.<model_id>.model_name`
- `models.<model_id>.gateway_protocol`
- optional `timeout`, `max_tokens`, `temperature`, `extra_headers`

For the current Claude Code SDK harness, only entries with
`api_type="claude_gateway"` and `gateway_protocol="anthropic"` may be used
through `--model-id`. Plain `api_type="openai"` entries are parsed as config
only and fail before the harness starts with a protocol-compatibility error.

Use `scripts/probe_llm_gateways.py` before marking an endpoint as
`claude_gateway`. The probe checks Anthropic Messages candidates such as
`/v1/messages`, records only redacted diagnostics under `runtime_logs/`, and
classifies entries as `anthropic_compatible`, `openai_only`, `auth_failed`,
`network_error`, or `unknown_error`.

When a model id resolves to a Claude gateway, the CLI injects these environment
variables into the Claude SDK subprocess:

- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_AUTH_TOKEN`
- `ANTHROPIC_API_KEY`
- `CLAUDE_API_KEY`
- `MYEVOSKILL_MODEL`

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
