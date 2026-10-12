import json

import httpx
import pytest

from feedback.ai.client import LLMClient, LLMConfigError, LLMError, MockProvider, OpenAIProvider, ProviderReply, get_llm_client
from tests.fake_openai import chat_completion, error_response, provider


class NullGuard:
    """In-memory guard for pure client tests (the persisted guard is tested in test_llm_budget)."""

    def __init__(self):
        self.acquired, self.finished, self.halts = 0, [], []

    def acquire(self, provider, model, purpose, **admission):
        self.last_admission = admission
        self.acquired += 1
        return self.acquired

    def finish(self, ticket, *, ok, latency_ms, error=None, reply=None):
        self.finished.append((ok, error.code if error else None))

    def halt(self, reason):
        self.halts.append(reason)


class ScriptedProvider:
    """Real-looking (is_mock=False) provider that returns/raises scripted items in order."""
    name, model, is_mock = "scripted", "s-1", False

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def call(self, system, user, schema_name, schema):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ProviderReply(item, 10, 5, 15)


def client(script, retries=2):
    guard = NullGuard()
    return LLMClient(ScriptedProvider(script), max_retries=retries, sleep=lambda s: None, guard=guard), guard


def test_real_provider_without_guard_is_refused():
    with pytest.raises(LLMConfigError):
        LLMClient(ScriptedProvider([]))


def test_retries_transient_errors_and_logs_every_attempt():
    c, guard = client([LLMError("timeout", retryable=True), LLMError("provider_unavailable", retryable=True),
                       '{"a": 1}'])
    resp = c.complete_json("s", "u", "x", {})
    assert resp.data == {"a": 1} and resp.attempts == 3 and resp.input_tokens == 10 and not resp.is_mock
    assert guard.acquired == 3 and guard.finished == [(False, "timeout"), (False, "provider_unavailable"), (True, None)]


def test_retries_are_bounded():
    c, _ = client([LLMError("timeout", retryable=True)] * 5, retries=2)
    with pytest.raises(LLMError) as exc:
        c.complete_json("s", "u", "x", {})
    assert exc.value.code == "timeout" and exc.value.attempts == 3 and c.provider.calls == 3


@pytest.mark.parametrize("code", ["rate_limited", "quota_exhausted", "auth_error", "model_not_found"])
def test_quota_and_auth_errors_never_retry_and_halt(code):
    # even if a provider wrongly marks the error retryable, it is not retried
    c, guard = client([LLMError(code, "x", retryable=True), '{"a": 1}'], retries=5)
    with pytest.raises(LLMError) as exc:
        c.complete_json("s", "u", "x", {})
    assert exc.value.code == code and c.provider.calls == 1 and len(guard.halts) == 1


def test_non_retryable_errors_fail_immediately():
    c, guard = client([LLMError("bad_request"), '{"a": 1}'])
    with pytest.raises(LLMError):
        c.complete_json("s", "u", "x", {})
    assert c.provider.calls == 1 and guard.halts == []


@pytest.mark.parametrize("payload", ["not json", "[1, 2]", ""])
def test_malformed_output_raises_invalid_json(payload):
    c, _ = client([payload])
    with pytest.raises(LLMError) as exc:
        c.complete_json("s", "u", "x", {})
    assert exc.value.code == "invalid_json"


def test_mock_is_default_and_labelled():
    c = get_llm_client()
    assert c.is_mock and c.provider_name == "mock" and c.guard is None


def test_openai_requires_key_no_silent_fallback(settings):
    settings.LLM = {**settings.LLM, "PROVIDER": "openai", "OPENAI_API_KEY": ""}
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        get_llm_client()


@pytest.mark.parametrize("name", ["gemini", "magic"])
def test_other_providers_are_rejected(settings, name):
    # historical Gemini results are stored data only; no other provider can be called
    settings.LLM = {**settings.LLM, "PROVIDER": name, "OPENAI_API_KEY": "sk-x"}
    with pytest.raises(LLMConfigError, match="Unsupported"):
        get_llm_client()


def test_openai_client_gets_persisted_guard_and_default_model(settings, db):
    settings.LLM = {**settings.LLM, "PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}
    c = get_llm_client()
    assert not c.is_mock and c.guard is not None and c.provider_name == "openai"
    assert c.model == settings.LLM["OPENAI_MODEL"] == "gpt-4o-mini"


# ---------------- OpenAI provider (real SDK over httpx.MockTransport; no network) ----------------
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


def test_openai_request_shape_and_usage():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json=chat_completion('{"a": 1}', prompt_tokens=42, completion_tokens=7))

    reply = provider(handler, max_output_tokens=256).call("SYS", "USER", "message_extraction", SCHEMA)
    assert json.loads(reply.text) == {"a": 1}
    assert (reply.input_tokens, reply.output_tokens, reply.total_tokens) == (42, 7, 49)
    assert seen["url"].endswith("/chat/completions") and "sk-test-key-123" not in seen["url"]
    assert seen["auth"] == "Bearer sk-test-key-123"  # key only in the Authorization header
    body = seen["body"]
    assert body["model"] == "gpt-4o-mini" and body["max_completion_tokens"] == 256 and body["temperature"] == 0
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert body["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "message_extraction", "strict": True, "schema": SCHEMA}}


@pytest.mark.parametrize("response,code,retryable", [
    (error_response(401, "invalid_api_key", "Incorrect API key provided: sk-proj-abcd****wxyz.", "invalid_request_error"),
     "auth_error", False),
    (error_response(403, "unsupported_country_region_territory", "Country not supported.", "request_forbidden"),
     "auth_error", False),
    (error_response(429, "rate_limit_exceeded", "Rate limit reached for gpt-4o-mini.", "requests"), "rate_limited", False),
    (error_response(429, "insufficient_quota", "You exceeded your current quota.", "insufficient_quota"),
     "quota_exhausted", False),
    (error_response(404, "model_not_found", "The model `gpt-x` does not exist."), "model_not_found", False),
    (error_response(400, "invalid_json_schema", "Invalid schema for response_format."), "bad_request", False),
    (error_response(500, None, "The server had an error.", "server_error"), "provider_unavailable", True),
    (error_response(503, None, "Overloaded.", "server_error"), "provider_unavailable", True),
])
def test_openai_error_mapping(response, code, retryable):
    with pytest.raises(LLMError) as exc:
        provider(lambda r: response).call("s", "u", "x", SCHEMA)
    assert exc.value.code == code and exc.value.retryable is retryable
    assert exc.value.http_status == response.status_code
    assert "sk-proj-abcd" not in exc.value.detail  # keys are never echoed into logs/UI


def test_openai_timeout_and_connection_errors():
    def slow(r):
        raise httpx.ReadTimeout("slow", request=r)

    def down(r):
        raise httpx.ConnectError("down", request=r)

    for handler, code in ((slow, "timeout"), (down, "connection_error")):
        with pytest.raises(LLMError) as exc:
            provider(handler).call("s", "u", "x", SCHEMA)
        assert exc.value.code == code and exc.value.retryable


@pytest.mark.parametrize("body,code", [
    ({**chat_completion("{}"), "choices": []}, "empty_response"),
    (chat_completion('{"a": ', finish_reason="length"), "truncated_output"),
    (chat_completion("", finish_reason="content_filter"), "refused"),
    (chat_completion(None, refusal="I can't help with that."), "refused"),
])
def test_openai_unusable_responses(body, code):
    with pytest.raises(LLMError) as exc:
        provider(lambda r: httpx.Response(200, json=body)).call("s", "u", "x", SCHEMA)
    assert exc.value.code == code


def test_openai_invalid_json_content_is_rejected_by_client():
    c = LLMClient(provider(lambda r: httpx.Response(200, json=chat_completion("not json"))), guard=NullGuard())
    with pytest.raises(LLMError) as exc:
        c.complete_json("s", "u", "x", SCHEMA)
    assert exc.value.code == "invalid_json"


def test_openai_requires_key():
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        OpenAIProvider(api_key="", model="m", timeout=1, max_output_tokens=10)


def test_mock_provider_responder_can_simulate_failure():
    c = LLMClient(MockProvider(responder=lambda s, u: LLMError("provider_unavailable", retryable=True)),
                  max_retries=1, sleep=lambda s: None)
    with pytest.raises(LLMError):
        c.complete_json("s", "u", "message_extraction", {})
