import importlib.util
import json
import urllib.error
from pathlib import Path

from myevoskill.model_provider import LlmModelEntry


def _load_probe_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "probe_llm_gateways.py"
    spec = importlib.util.spec_from_file_location("probe_llm_gateways", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _entry(**overrides):
    data = {
        "api_type": "openai",
        "base_url": "https://gateway.example/v1",
        "api_key": "test-probe-secret",
        "model_name": "probe-model",
    }
    data.update(overrides)
    return LlmModelEntry.from_mapping("probe", data)


def test_probe_classifies_200_as_anthropic_compatible_and_redacts_key():
    probe = _load_probe_module()

    def opener(_request, _timeout):
        return 200, '{"id":"msg_1","type":"message","content":[]}'

    result = probe.probe_entry(_entry(), timeout=1.0, opener=opener)

    assert result["classification"] == "anthropic_compatible"
    assert "test-probe-secret" not in json.dumps(result)


def test_probe_does_not_treat_200_html_as_anthropic_compatible():
    probe = _load_probe_module()

    def opener(_request, _timeout):
        return 200, "<!doctype html><html></html>"

    result = probe.probe_entry(_entry(), timeout=1.0, opener=opener)

    assert result["classification"] == "unknown_error"


def test_probe_classifies_404_as_openai_only():
    probe = _load_probe_module()

    def opener(_request, _timeout):
        return 404, '{"error":"not found"}'

    result = probe.probe_entry(_entry(), timeout=1.0, opener=opener)

    assert result["classification"] == "openai_only"
    assert len(result["attempts"]) == 4


def test_probe_classifies_401_as_auth_failed():
    probe = _load_probe_module()

    def opener(_request, _timeout):
        return 401, '{"error":"unauthorized"}'

    result = probe.probe_entry(_entry(), timeout=1.0, opener=opener)

    assert result["classification"] == "auth_failed"


def test_probe_classifies_all_url_errors_as_network_error():
    probe = _load_probe_module()

    def opener(_request, _timeout):
        raise urllib.error.URLError("timed out")

    result = probe.probe_entry(_entry(), timeout=1.0, opener=opener)

    assert result["classification"] == "network_error"


def test_candidate_url_respects_v1_base_url():
    probe = _load_probe_module()

    assert (
        probe.candidate_url("https://api.example/v1", "/v1/messages")
        == "https://api.example/v1/messages"
    )
    assert (
        probe.candidate_url("https://api.example/v1", "/anthropic/v1/messages")
        == "https://api.example/anthropic/v1/messages"
    )
