#!/usr/bin/env python
"""Probe configured LLM endpoints for Anthropic Messages compatibility.

The output is safe to keep in ``runtime_logs``: it never includes real API keys.
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from myevoskill.model_provider import (  # noqa: E402
    LlmModelEntry,
    ModelProviderRegistry,
    default_llm_config_path,
)


CANDIDATE_MESSAGE_PATHS = (
    "/v1/messages",
    "/anthropic/v1/messages",
    "/v1/anthropic/messages",
    "/v1/claude/messages",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe llm.yaml endpoints for Anthropic /v1/messages support."
    )
    parser.add_argument(
        "--config",
        default=str(default_llm_config_path(REPO_ROOT)),
        help="Path to llm.yaml (default: config/llm.yaml)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path (default: runtime_logs/llm_probe_<ts>.json)",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)

    registry = ModelProviderRegistry.from_file(Path(args.config))
    out_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "runtime_logs" / f"llm_probe_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for model_id in registry.model_ids():
        entry = registry.get(model_id)
        result = probe_entry(entry, timeout=float(args.timeout))
        results.append(result)
        print(f"{model_id}: {result['classification']}")

    payload = {
        "config_path": str(Path(args.config).resolve()),
        "created_at": int(time.time()),
        "candidate_paths": list(CANDIDATE_MESSAGE_PATHS),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"probe results: {out_path}")
    return 0


def probe_entry(
    entry: LlmModelEntry,
    *,
    timeout: float,
    opener: Callable[[urllib.request.Request, float], tuple[int, str]] | None = None,
) -> dict[str, Any]:
    opener = opener or _default_opener
    key, key_source = _safe_resolve_key(entry)
    attempts: list[dict[str, Any]] = []

    if not entry.base_url:
        return {
            "model_id": entry.model_id,
            "model_name": entry.public_model_name,
            "base_url": entry.base_url,
            "api_type": entry.api_type,
            "gateway_protocol": entry.gateway_protocol,
            "api_key_source": key_source,
            "classification": "unknown_error",
            "attempts": [{"error": "missing_base_url"}],
        }
    if not key:
        return {
            "model_id": entry.model_id,
            "model_name": entry.public_model_name,
            "base_url": entry.base_url,
            "api_type": entry.api_type,
            "gateway_protocol": entry.gateway_protocol,
            "api_key_source": key_source,
            "classification": "auth_failed",
            "attempts": [{"error": "missing_api_key"}],
        }

    for path in CANDIDATE_MESSAGE_PATHS:
        url = candidate_url(entry.base_url, path)
        request = build_anthropic_probe_request(
            url=url,
            model=entry.public_model_name,
            api_key=key,
        )
        attempt = {"path": path, "url": url}
        try:
            status, text = opener(request, timeout)
            attempt.update(
                {
                    "status": status,
                    "body_preview": _redact_text(text[:300]),
                    "classification": classify_http_result(status, text),
                }
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            attempt.update(
                {
                    "status": int(exc.code),
                    "body_preview": _redact_text(body[:300]),
                    "classification": classify_http_result(int(exc.code), body),
                }
            )
        except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as exc:
            attempt.update(
                {
                    "error": type(exc).__name__,
                    "error_message": _redact_text(str(exc)),
                    "classification": "network_error",
                }
            )
        attempts.append(attempt)
        if attempt["classification"] == "anthropic_compatible":
            break

    return {
        "model_id": entry.model_id,
        "model_name": entry.public_model_name,
        "base_url": entry.base_url,
        "api_type": entry.api_type,
        "gateway_protocol": entry.gateway_protocol,
        "api_key_source": key_source,
        "classification": summarize_attempts(attempts),
        "attempts": attempts,
    }


def build_anthropic_probe_request(
    *, url: str, model: str, api_key: str
) -> urllib.request.Request:
    payload = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": api_key,
            "authorization": f"Bearer {api_key}",
        },
    )


def candidate_url(base_url: str, candidate_path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    origin = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    base_path = parsed.path.rstrip("/")
    if candidate_path == "/v1/messages" and base_path.endswith("/v1"):
        return urllib.parse.urlunparse(
            (parsed.scheme, parsed.netloc, f"{base_path}/messages", "", "", "")
        )
    return urllib.parse.urljoin(origin + "/", candidate_path.lstrip("/"))


def classify_http_result(status: int, body: str) -> str:
    if 200 <= status < 300 and looks_like_anthropic_message(body):
        return "anthropic_compatible"
    if 200 <= status < 300:
        return "unknown_error"
    lowered = body.lower()
    if status in (401, 403) or "api key" in lowered or "unauthorized" in lowered:
        return "auth_failed"
    if status in (400, 404, 405, 415):
        return "openai_only"
    return "unknown_error"


def looks_like_anthropic_message(body: str) -> bool:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, Mapping):
        return False
    if data.get("type") != "message":
        return False
    content = data.get("content")
    return isinstance(content, list)


def summarize_attempts(attempts: list[Mapping[str, Any]]) -> str:
    classes = [str(a.get("classification") or "") for a in attempts]
    if "anthropic_compatible" in classes:
        return "anthropic_compatible"
    if "auth_failed" in classes:
        return "auth_failed"
    if classes and all(c == "network_error" for c in classes):
        return "network_error"
    if "openai_only" in classes:
        return "openai_only"
    return "unknown_error"


def _default_opener(request: urllib.request.Request, timeout: float) -> tuple[int, str]:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(response.status)
        body = response.read().decode("utf-8", "replace")
    return status, body


def _safe_resolve_key(entry: LlmModelEntry) -> tuple[str, str]:
    try:
        key, source = entry.resolve_api_key()
    except Exception:
        return "", "missing"
    if source == "inline":
        return key, "inline"
    return key, source


def _redact_text(text: str) -> str:
    # Keep this intentionally simple; the most important guarantee is that the
    # script never writes the configured key value itself.
    return text.replace("\r", " ").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
