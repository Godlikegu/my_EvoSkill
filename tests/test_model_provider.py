import pytest

from myevoskill.model_provider import (
    AnthropicCompatibleAdapter,
    ClaudeSDKAdapter,
    CustomHTTPAdapter,
    OpenAICompatibleAdapter,
)
from myevoskill.models import ModelConfig


def test_model_provider_adapters_describe_config():
    config = ModelConfig(
        provider_name="custom",
        model_name="m1",
        base_url="https://example.invalid",
        api_key_env="TEST_API_KEY",
        timeout=30,
        max_tokens=256,
        temperature=0.1,
    )
    assert OpenAICompatibleAdapter(config).describe()["model_name"] == "m1"
    assert AnthropicCompatibleAdapter(config).describe()["api_key_env"] == "TEST_API_KEY"
    assert CustomHTTPAdapter(config).describe()["kind"] == "custom_http"
    assert ClaudeSDKAdapter(config).describe()["kind"] == "claude_sdk"


def test_openai_compatible_adapter_resolves_api_key_and_builds_request(monkeypatch):
    config = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        max_tokens=128,
        temperature=0.2,
    )
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    adapter = OpenAICompatibleAdapter(config)
    url, headers, payload = adapter.build_request(
        [{"role": "user", "content": "hello"}]
    )
    assert url == "https://example.invalid/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret-value"
    assert payload["model"] == "m1"
    assert payload["max_tokens"] == 128
    assert adapter.safe_log_config()["api_key_env"] == "TEST_API_KEY"


def test_openai_compatible_adapter_requires_api_key_env(monkeypatch):
    config = ModelConfig(
        provider_name="openai-compatible",
        model_name="m1",
        base_url="https://example.invalid/v1",
        api_key_env="MISSING_KEY",
    )
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="environment variable 'MISSING_KEY'"):
        OpenAICompatibleAdapter(config).resolve_api_key()


def test_claude_sdk_adapter_uses_default_anthropic_env(monkeypatch):
    config = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sdk-secret")
    adapter = ClaudeSDKAdapter(config)
    assert adapter.resolve_api_key() == "sdk-secret"
    assert adapter.describe()["api_key_env"] == "ANTHROPIC_API_KEY"


def test_claude_sdk_adapter_falls_back_from_myevoskill_env_to_claude_api_key(monkeypatch):
    config = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="MYEVOSKILL_CLAUDE_API_KEY",
    )
    monkeypatch.delenv("MYEVOSKILL_CLAUDE_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "sdk-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = ClaudeSDKAdapter(config)
    assert adapter.resolve_api_key() == "sdk-secret"
    assert adapter.resolve_api_key_env_name() == "CLAUDE_API_KEY"


def test_claude_sdk_adapter_builds_sdk_env_with_aliases(monkeypatch):
    config = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-test",
        api_key_env="MYEVOSKILL_CLAUDE_API_KEY",
    )
    monkeypatch.delenv("MYEVOSKILL_CLAUDE_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "sdk-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = ClaudeSDKAdapter(config)
    sdk_env = adapter.build_sdk_env()
    assert sdk_env["MYEVOSKILL_CLAUDE_API_KEY"] == "sdk-secret"
    assert sdk_env["CLAUDE_API_KEY"] == "sdk-secret"
    assert sdk_env["ANTHROPIC_API_KEY"] == "sdk-secret"


def test_claude_sdk_adapter_resolve_model_name_prefers_explicit_config(monkeypatch):
    config = ModelConfig(
        provider_name="claude-sdk",
        model_name="claude-explicit",
    )
    monkeypatch.setenv("MYEVOSKILL_CLAUDE_MODEL", "claude-from-env")
    adapter = ClaudeSDKAdapter(config)
    assert adapter.resolve_model_name() == "claude-explicit"


def test_claude_sdk_adapter_resolve_model_name_falls_back_to_env(monkeypatch):
    config = ModelConfig(
        provider_name="claude-sdk",
        model_name="",
    )
    monkeypatch.setenv("MYEVOSKILL_CLAUDE_MODEL", "sonnet")
    adapter = ClaudeSDKAdapter(config)
    assert adapter.resolve_model_name() == "sonnet"


def test_claude_sdk_adapter_resolve_model_name_allows_cli_default(monkeypatch):
    config = ModelConfig(
        provider_name="claude-sdk",
        model_name="",
    )
    monkeypatch.delenv("MYEVOSKILL_CLAUDE_MODEL", raising=False)
    adapter = ClaudeSDKAdapter(config)
    assert adapter.resolve_model_name() == ""
