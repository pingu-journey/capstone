"""HTTP auth failures stop immediately; transient service failures stay bounded."""

import httpx
import pytest
import requests
from huggingface_hub.errors import HfHubHTTPError

from tools.services import ConfigurationError, TransientServiceError, check_service_error


def http_error(family, status):
    url = "https://private.example.org/test?token=secret"
    if family == "requests":
        response = requests.Response()
        response.status_code = status
        response.url = url
        return requests.HTTPError("private secret", response=response)
    response = httpx.Response(status, request=httpx.Request("GET", url))
    if family == "hf":
        return HfHubHTTPError("private secret", response=response)
    return httpx.HTTPStatusError("private secret", request=response.request, response=response)


@pytest.mark.parametrize("family", ["requests", "httpx", "hf"])
@pytest.mark.parametrize("status", [400, 401, 403, 408, 429, 500, 503])
def test_http_status_classification_hides_sensitive_error(family, status):
    expected = ConfigurationError if status in (400, 401, 403) else TransientServiceError
    with pytest.raises(expected) as raised:
        check_service_error(http_error(family, status))
    assert "secret" not in str(raised.value) and "private" not in str(raised.value)
    assert raised.value.__suppress_context__ and raised.value.__cause__ is None


@pytest.mark.parametrize("error", [httpx.ConnectError("secret"), httpx.ReadTimeout("secret"),
    requests.ConnectionError("secret"), requests.Timeout("secret")])
def test_timeout_connection_remain_transient(error):
    with pytest.raises(TransientServiceError):
        check_service_error(error)


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_real_hf_error_worker_auth_propagates_without_retry(tmp_path, monkeypatch, status):
    import json
    from agents import orchestrator, research_worker
    from config import ROOT
    from tools import artifacts
    monkeypatch.setattr(artifacts, "OUTPUT_DIR", tmp_path)
    state = json.loads((ROOT / "tests/fixtures/sample_state.json").read_text())
    state.update(run_id="http_status_test", market_cache={}, research_cache={})
    state["task"] = orchestrator.run(state)["tasks"][0]
    attempts = []
    def collect(*args):
        attempts.append(1)
        raise http_error("hf", status)
    monkeypatch.setattr(research_worker, "collect", collect)
    if status in (401, 403):
        with pytest.raises(ConfigurationError):
            research_worker.run(state)
        assert len(attempts) == 1 and not list((artifacts.run_dir(state) / "workers").glob("*.json"))
    else:
        result = next(iter(research_worker.run(state)["worker_results"].values()))
        assert len(attempts) == result["attempts"] == 2
        assert result["status"] == "failed" and result["error_code"] == "service_unavailable"


def test_missing_hf_token_remains_configuration_error(monkeypatch):
    from rag.embeddings import Embedder
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ConfigurationError, match="HF_TOKEN"):
        Embedder("hf_api").embed_documents(["test"])


@pytest.mark.parametrize("error_type", ["AuthenticationError", "PermissionDeniedError", "BadRequestError"])
def test_openai_provider_fatal_errors_are_safe_configuration_errors(error_type):
    import openai
    response = httpx.Response(401, request=httpx.Request("GET", "https://private.example.org"))
    error = getattr(openai, error_type)("secret", response=response, body=None)
    with pytest.raises(ConfigurationError) as raised:
        check_service_error(error)
    assert "secret" not in str(raised.value)


@pytest.mark.parametrize("error_type", ["InvalidAPIKeyError", "MissingAPIKeyError", "ForbiddenError", "BadRequestError"])
def test_tavily_provider_fatal_errors_are_safe_configuration_errors(error_type):
    from tavily import errors
    error = getattr(errors, error_type)() if error_type == "MissingAPIKeyError" else getattr(errors, error_type)("secret")
    with pytest.raises(ConfigurationError):
        check_service_error(error)
