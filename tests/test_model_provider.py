import json
from pathlib import Path

import pytest

from myevoskill.model_provider import (
    ModelGatewayError,
    ModelNotFoundError,
    ModelProviderRegistry,
    build_claude_gateway_env,
    resolve_model_config,
)


def _write_config(path: Path) -> Path:
    path.write_text(
        """
models:
  openai-only:
    api_type: openai
    base_url: https://openai.example/v1
    api_key: test-openai-secret
    model_name: openai-model
    temperature: 0.7
  claude-gateway:
    api_type: claude_gateway
    gateway_protocol: anthropic
    base_url: https://gateway.example
    api_key: test-gateway-secret
    model_name: gateway-model
    temperature: 0.2
  env-gateway:
    api_type: claude_gateway
    gateway_protocol: anthropic
    base_url: https://env-gateway.example
    api_key_env: TEST_GATEWAY_KEY
    api_key: test-inline-fallback
    model_name: env-model
""",
        encoding="utf-8",
    )
    return path


def test_loads_yaml_and_resolves_model_config(tmp_path):
    config = _write_config(tmp_path / "llm.yaml")

    model = resolve_model_config("openai-only", config)

    assert model.provider_name == "openai"
    assert model.model_name == "openai-model"
    assert model.base_url == "https://openai.example/v1"
    assert model.temperature == 0.7


def test_safe_log_config_redacts_inline_key(tmp_path):
    config = _write_config(tmp_path / "llm.yaml")
    registry = ModelProviderRegistry.from_file(config)

    safe = registry.safe_log_config("openai-only")
    payload = json.dumps(safe, sort_keys=True)

    assert "test-openai-secret" not in payload
    assert safe["api_key_source"] == "inline"
    assert safe["uses_inline_api_key"] is True


def test_openai_model_cannot_build_claude_gateway_env(tmp_path):
    config = _write_config(tmp_path / "llm.yaml")

    with pytest.raises(ModelGatewayError, match="requires api_type='claude_gateway'"):
        build_claude_gateway_env("openai-only", config)


def test_claude_gateway_builds_expected_env_without_logging_key(tmp_path):
    config = _write_config(tmp_path / "llm.yaml")
    registry = ModelProviderRegistry.from_file(config)

    runtime = registry.resolve_claude_gateway_runtime("claude-gateway", env={})

    assert runtime.env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert runtime.env["ANTHROPIC_AUTH_TOKEN"] == "test-gateway-secret"
    assert runtime.env["ANTHROPIC_API_KEY"] == "test-gateway-secret"
    assert runtime.env["CLAUDE_API_KEY"] == "test-gateway-secret"
    assert runtime.env["MYEVOSKILL_MODEL"] == "gateway-model"
    assert "test-gateway-secret" not in json.dumps(runtime.safe_log_config())


def test_api_key_env_wins_over_inline_key(tmp_path):
    config = _write_config(tmp_path / "llm.yaml")
    registry = ModelProviderRegistry.from_file(config)

    runtime = registry.resolve_claude_gateway_runtime(
        "env-gateway", env={"TEST_GATEWAY_KEY": "from-env"}
    )

    assert runtime.env["ANTHROPIC_AUTH_TOKEN"] == "from-env"
    assert runtime.safe_log_config()["api_key_source"] == "env:TEST_GATEWAY_KEY"


def test_missing_model_reports_available_models(tmp_path):
    config = _write_config(tmp_path / "llm.yaml")
    registry = ModelProviderRegistry.from_file(config)

    with pytest.raises(ModelNotFoundError, match="available:"):
        registry.get("missing")
