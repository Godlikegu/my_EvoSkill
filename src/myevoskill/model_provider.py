"""Model provider configuration adapters."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Dict, Mapping, Tuple

from .models import ModelConfig


class ModelProviderAdapter(ABC):
    """Base interface for model provider configuration adapters."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def describe(self) -> Dict[str, object]:
        """Return a serializable description of the configured provider."""

    def resolve_api_key(self, env: Mapping[str, str] | None = None) -> str:
        """Resolve the configured API key from the environment."""

        env = env or os.environ
        if not self.config.api_key_env:
            raise RuntimeError(
                f"model provider '{self.config.provider_name}' is missing api_key_env"
            )
        api_key = env.get(self.config.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"environment variable '{self.config.api_key_env}' is required for "
                f"model provider '{self.config.provider_name}'"
            )
        return api_key

    def safe_log_config(self) -> Dict[str, object]:
        """Return a log-safe provider config without secret values."""

        payload = dict(self.describe())
        payload.pop("resolved_api_key", None)
        return payload

    def build_request(
        self,
        messages: list[dict[str, object]],
        env: Mapping[str, str] | None = None,
    ) -> Tuple[str, Dict[str, str], Dict[str, object]]:
        """Build an HTTP request triple for providers that support chat APIs."""

        raise RuntimeError(
            f"provider '{self.config.provider_name}' does not implement build_request"
        )


class OpenAICompatibleAdapter(ModelProviderAdapter):
    """Configuration holder for OpenAI-compatible model providers."""

    def describe(self) -> Dict[str, object]:
        return {"kind": "openai_compatible", **asdict(self.config)}

    def build_request(
        self,
        messages: list[dict[str, object]],
        env: Mapping[str, str] | None = None,
    ) -> Tuple[str, Dict[str, str], Dict[str, object]]:
        api_key = self.resolve_api_key(env)
        base_url = self.config.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError("openai-compatible provider requires a non-empty base_url")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **dict(self.config.extra_headers),
        }
        payload: Dict[str, object] = {
            "model": self.config.model_name,
            "messages": messages,
        }
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        payload["temperature"] = self.config.temperature
        return f"{base_url}/chat/completions", headers, payload


class AnthropicCompatibleAdapter(ModelProviderAdapter):
    """Configuration holder for Anthropic-compatible model providers."""

    def describe(self) -> Dict[str, object]:
        return {"kind": "anthropic_compatible", **asdict(self.config)}


class ClaudeSDKAdapter(ModelProviderAdapter):
    """Configuration holder for Claude Agent SDK providers."""

    DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
    DEFAULT_MODEL_ENV = "MYEVOSKILL_CLAUDE_MODEL"
    API_KEY_ENV_ALIASES = (
        "MYEVOSKILL_CLAUDE_API_KEY",
        "CLAUDE_API_KEY",
        "ANTHROPIC_API_KEY",
    )

    def describe(self) -> Dict[str, object]:
        payload = asdict(self.config)
        if not payload.get("api_key_env"):
            payload["api_key_env"] = self.DEFAULT_API_KEY_ENV
        return {"kind": "claude_sdk", **payload}

    def resolve_api_key(self, env: Mapping[str, str] | None = None) -> str:
        env = env or os.environ
        key_env = self.config.api_key_env or self.DEFAULT_API_KEY_ENV
        for candidate in self._candidate_api_key_envs():
            api_key = env.get(candidate, "")
            if api_key:
                return api_key
        alias_text = ", ".join(f"'{name}'" for name in self._candidate_api_key_envs())
        raise RuntimeError(
            f"one of environment variables {alias_text} is required for "
            f"model provider '{self.config.provider_name}'"
        )

    def resolve_api_key_env_name(self, env: Mapping[str, str] | None = None) -> str:
        env = env or os.environ
        for candidate in self._candidate_api_key_envs():
            if env.get(candidate, ""):
                return candidate
        return self.config.api_key_env or self.DEFAULT_API_KEY_ENV

    def resolve_model_name(self, env: Mapping[str, str] | None = None) -> str:
        explicit = str(self.config.model_name or "").strip()
        if explicit:
            return explicit
        env = env or os.environ
        return str(env.get(self.DEFAULT_MODEL_ENV, "") or "").strip()

    def build_sdk_env(self, env: Mapping[str, str] | None = None) -> Dict[str, str]:
        env = env or os.environ
        api_key = self.resolve_api_key(env)
        resolved_env = dict(env)
        for candidate in self._candidate_api_key_envs():
            resolved_env[candidate] = api_key
        return {
            key: resolved_env[key]
            for key in self._candidate_api_key_envs()
        }

    def _candidate_api_key_envs(self) -> tuple[str, ...]:
        preferred = self.config.api_key_env or self.DEFAULT_API_KEY_ENV
        ordered: list[str] = [preferred]
        ordered.extend(self.API_KEY_ENV_ALIASES)
        deduped: list[str] = []
        for item in ordered:
            if item and item not in deduped:
                deduped.append(item)
        return tuple(deduped)


class CustomHTTPAdapter(ModelProviderAdapter):
    """Configuration holder for custom HTTP LLM APIs."""

    def describe(self) -> Dict[str, object]:
        return {"kind": "custom_http", **asdict(self.config)}

    def build_request(
        self,
        messages: list[dict[str, object]],
        env: Mapping[str, str] | None = None,
    ) -> Tuple[str, Dict[str, str], Dict[str, object]]:
        api_key = self.resolve_api_key(env)
        base_url = self.config.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError("custom_http provider requires a non-empty base_url")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **dict(self.config.extra_headers),
        }
        payload: Dict[str, object] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        return f"{base_url}/chat/completions", headers, payload
