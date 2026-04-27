"""Model provider config loading for local LLM gateway settings.

This module intentionally does not implement a native OpenAI/Gemini agent
backend. It only bridges Anthropic-compatible gateways into the existing
Claude Code SDK harness.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import ModelConfig


class ModelProviderError(ValueError):
    """Base error for model provider configuration problems."""


class ModelNotFoundError(ModelProviderError):
    """Raised when a requested model id is not present in the config."""


class ModelGatewayError(ModelProviderError):
    """Raised when a model cannot be used with the Claude gateway harness."""


@dataclass(frozen=True)
class LlmModelEntry:
    """One model entry from ``config/llm.yaml``."""

    model_id: str
    api_type: str = "openai"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    model_name: str = ""
    gateway_protocol: str = ""
    timeout: int = 0
    max_tokens: int = 0
    temperature: float = 0.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, model_id: str, raw: Mapping[str, Any]) -> "LlmModelEntry":
        if not isinstance(raw, Mapping):
            raise ModelProviderError(f"model entry {model_id!r} must be a mapping")

        headers = raw.get("extra_headers") or {}
        if not isinstance(headers, Mapping):
            raise ModelProviderError(
                f"model entry {model_id!r} has non-mapping extra_headers"
            )

        return cls(
            model_id=str(model_id),
            api_type=str(raw.get("api_type") or "openai").strip(),
            base_url=str(raw.get("base_url") or "").strip(),
            api_key=str(raw.get("api_key") or "").strip(),
            api_key_env=str(raw.get("api_key_env") or "").strip(),
            model_name=str(raw.get("model_name") or model_id).strip(),
            gateway_protocol=str(raw.get("gateway_protocol") or "").strip(),
            timeout=_coerce_int(raw.get("timeout")),
            max_tokens=_coerce_int(raw.get("max_tokens")),
            temperature=_coerce_float(raw.get("temperature")),
            extra_headers={str(k): str(v) for k, v in headers.items()},
        )

    @property
    def public_model_name(self) -> str:
        return self.model_name or self.model_id

    @property
    def is_claude_gateway(self) -> bool:
        return (
            self.api_type.lower() == "claude_gateway"
            and self.gateway_protocol.lower() == "anthropic"
        )

    @property
    def key_source_label(self) -> str:
        if self.api_key_env:
            return f"env:{self.api_key_env}"
        if self.api_key:
            return "inline"
        return "missing"

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            provider_name=self.api_type,
            model_name=self.public_model_name,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_headers=dict(self.extra_headers),
        )

    def safe_log_config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.public_model_name,
            "api_type": self.api_type,
            "gateway_protocol": self.gateway_protocol,
            "base_url": self.base_url,
            "api_key_source": self.key_source_label,
            "uses_inline_api_key": bool(self.api_key and not self.api_key_env),
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra_header_names": sorted(self.extra_headers),
        }

    def resolve_api_key(self, env: Mapping[str, str] | None = None) -> tuple[str, str]:
        env = os.environ if env is None else env
        if self.api_key_env:
            value = str(env.get(self.api_key_env) or "").strip()
            if value:
                return value, f"env:{self.api_key_env}"
        if self.api_key:
            return self.api_key, "inline"
        raise ModelGatewayError(
            f"model {self.model_id!r} has no usable API key; set api_key_env "
            "or provide a local inline api_key"
        )

    def build_claude_gateway_env(
        self, env: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        if not self.is_claude_gateway:
            raise ModelGatewayError(
                f"model {self.model_id!r} is api_type={self.api_type!r}, "
                f"gateway_protocol={self.gateway_protocol!r}; the current "
                "Claude Code harness requires api_type='claude_gateway' and "
                "gateway_protocol='anthropic'. Use an Anthropic-compatible "
                "gateway, LiteLLM proxy, or a future native OpenAI/Gemini backend."
            )
        if not self.base_url:
            raise ModelGatewayError(f"model {self.model_id!r} is missing base_url")

        token, _source = self.resolve_api_key(env)
        return {
            "ANTHROPIC_BASE_URL": self.base_url,
            "ANTHROPIC_AUTH_TOKEN": token,
            "ANTHROPIC_API_KEY": token,
            "CLAUDE_API_KEY": token,
            "MYEVOSKILL_MODEL": self.public_model_name,
        }


@dataclass(frozen=True)
class ProviderRuntime:
    """Resolved runtime settings for one configured model."""

    entry: LlmModelEntry
    model_config: ModelConfig
    env: Mapping[str, str]
    key_source: str

    def safe_log_config(self) -> dict[str, Any]:
        data = self.entry.safe_log_config()
        data["api_key_source"] = self.key_source
        return data


class ModelProviderRegistry:
    """Loader and lookup surface for ``config/llm.yaml``."""

    def __init__(self, models: Mapping[str, LlmModelEntry], config_path: Path):
        self._models = dict(models)
        self.config_path = Path(config_path)

    @classmethod
    def from_file(cls, config_path: Path) -> "ModelProviderRegistry":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"LLM config not found: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise ModelProviderError(f"LLM config must be a mapping: {path}")
        raw_models = data.get("models") or {}
        if not isinstance(raw_models, Mapping):
            raise ModelProviderError(f"LLM config 'models' must be a mapping: {path}")

        models = {
            str(model_id): LlmModelEntry.from_mapping(str(model_id), raw)
            for model_id, raw in raw_models.items()
        }
        return cls(models=models, config_path=path)

    def get(self, model_id: str) -> LlmModelEntry:
        try:
            return self._models[model_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._models)) or "<none>"
            raise ModelNotFoundError(
                f"model_id {model_id!r} not found in {self.config_path}; "
                f"available: {available}"
            ) from exc

    def model_ids(self) -> list[str]:
        return sorted(self._models)

    def safe_log_config(self, model_id: str) -> dict[str, Any]:
        return self.get(model_id).safe_log_config()

    def resolve_model_config(self, model_id: str) -> ModelConfig:
        return self.get(model_id).to_model_config()

    def resolve_claude_gateway_runtime(
        self, model_id: str, env: Mapping[str, str] | None = None
    ) -> ProviderRuntime:
        entry = self.get(model_id)
        token, source = entry.resolve_api_key(env)
        runtime_env = entry.build_claude_gateway_env(
            {**dict(env or os.environ), "__MYEVOSKILL_RESOLVED_KEY": token}
        )
        # build_claude_gateway_env resolves again from the merged env; restore the
        # exact token we already selected so source reporting stays deterministic.
        runtime_env.update(
            {
                "ANTHROPIC_AUTH_TOKEN": token,
                "ANTHROPIC_API_KEY": token,
                "CLAUDE_API_KEY": token,
            }
        )
        return ProviderRuntime(
            entry=entry,
            model_config=entry.to_model_config(),
            env=runtime_env,
            key_source=source,
        )


def default_llm_config_path(repo_root: Path) -> Path:
    return Path(repo_root) / "config" / "llm.yaml"


def load_model_provider_registry(config_path: Path) -> ModelProviderRegistry:
    return ModelProviderRegistry.from_file(config_path)


def resolve_model_config(model_id: str, config_path: Path) -> ModelConfig:
    return load_model_provider_registry(config_path).resolve_model_config(model_id)


def build_claude_gateway_env(
    model_id: str,
    config_path: Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    runtime = load_model_provider_registry(config_path).resolve_claude_gateway_runtime(
        model_id, env=env
    )
    return dict(runtime.env)


def _coerce_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ModelProviderError(f"expected integer value, got {value!r}") from None


def _coerce_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ModelProviderError(f"expected float value, got {value!r}") from None
