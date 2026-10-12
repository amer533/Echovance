import json

import httpx
import openai
import pytest

from feedback.ai.client import (GeminiProvider, LLMClient, LLMConfigError, LLMError, MockProvider, OpenAIProvider,
                                ProviderReply, get_llm_client)

REQ = httpx.Request("POST", "https://api.openai.test/v1/chat/completions")


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


@pytest.mark.parametrize("code", ["rate_limited", "quota_exhausted", "auth_error", "model_not_found",
                                  "free_tier_unavailable"])
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


def test_openai_disabled_without_explicit_flag(settings):
    settings.LLM = {**settings.LLM, "PROVIDER": "openai", "OPENAI_API_KEY": "sk-would-be-used", "OPENAI_ENABLED": False}
    with pytest.raises(LLMConfigError, match="disabled"):
        get_llm_client()


def test_gemini_requires_key_no_silent_fallback(settings):
    settings.LLM = {**settings.LLM, "PROVIDER": "gemini", "GEMINI_API_KEY": ""}
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        get_llm_client()


def test_unknown_provider_rejected(settings):
    settings.LLM = {**settings.LLM, "PROVIDER": "magic"}
    with pytest.raises(LLMConfigError):
        get_llm_client()


def test_gemini_client_gets_persisted_guard(settings, db):
    settings.LLM = {**settings.LLM, "PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}
    c = get_llm_client()
    assert not c.is_mock and c.guard is not None and c.model == settings.LLM["GEMINI_MODEL"]


# ---------------- Gemini provider (HTTP mocked with httpx.MockTransport; no network) ----------------
def gemini(handler, **kw):
    return GeminiProvider(api_key="test-key-123", model="gemini-test", timeout=5, max_output_tokens=256,
                          transport=httpx.MockTransport(handler), **kw)


def ok_body(text='{"ok": true}', finish="STOP", usage=True):
    body = {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}]}
    if usage:
        body["usageMetadata"] = {"promptTokenCount": 120, "candidatesTokenCount": 30, "totalTokenCount": 150}
    return body


SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"a": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                           "properties": {"x": {"type": "string", "enum": ["p"]}},
                                                           "required": ["x"]}}},
          "required": ["a"]}


def test_gemini_request_shape_and_usage():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    reply = gemini(handler).call("SYS", "USER", "message_extraction", SCHEMA)
    assert json.loads(reply.text) == {"ok": True}
    assert (reply.input_tokens, reply.output_tokens, reply.total_tokens) == (120, 30, 150)
    assert seen["url"].endswith("/models/gemini-test:generateContent")
    assert "key=" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "test-key-123"
    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
    gen = body["generationConfig"]
    assert gen["responseMimeType"] == "application/json" and gen["temperature"] == 0.0
    assert "additionalProperties" not in json.dumps(gen["responseSchema"])
    assert gen["responseSchema"]["propertyOrdering"] == ["a"]
    assert gen["responseSchema"]["properties"]["a"]["items"]["properties"]["x"]["enum"] == ["p"]


def test_gemini_skips_thought_parts():
    body = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "{}"}]},
                            "finishReason": "STOP"}]}
    assert gemini(lambda r: httpx.Response(200, json=body)).call("s", "u", "x", {}).text == "{}"


def err(status, gstatus, message):
    return lambda r: httpx.Response(status, json={"error": {"code": status, "status": gstatus, "message": message}})


@pytest.mark.parametrize("handler,code,retryable", [
    (err(429, "RESOURCE_EXHAUSTED", "Quota exceeded for metric generate_content_free_tier_requests"), "quota_exhausted", False),
    (err(429, "RESOURCE_EXHAUSTED", "Too many requests"), "rate_limited", False),
    (err(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."), "auth_error", False),
    (err(403, "PERMISSION_DENIED", "Permission denied"), "auth_error", False),
    (err(404, "NOT_FOUND", "models/gemini-test is not found"), "model_not_found", False),
    (err(400, "FAILED_PRECONDITION", "User location is not supported for the API use."), "free_tier_unavailable", False),
    (err(400, "INVALID_ARGUMENT", "Invalid JSON payload"), "bad_request", False),
    (err(503, "UNAVAILABLE", "overloaded"), "provider_unavailable", True),
])
def test_gemini_error_mapping(handler, code, retryable):
    with pytest.raises(LLMError) as exc:
        gemini(handler).call("s", "u", "x", {})
    assert exc.value.code == code and exc.value.retryable is retryable
    assert "test-key-123" not in exc.value.detail


def test_gemini_timeout_and_connection_errors():
    def timeout(r):
        raise httpx.ReadTimeout("slow", request=r)

    def broken(r):
        raise httpx.ConnectError("down", request=r)

    for handler, code in ((timeout, "timeout"), (broken, "connection_error")):
        with pytest.raises(LLMError) as exc:
            gemini(handler).call("s", "u", "x", {})
        assert exc.value.code == code and exc.value.retryable


@pytest.mark.parametrize("body,code", [
    (ok_body(finish="MAX_TOKENS"), "truncated_output"),
    (ok_body(finish="SAFETY"), "refused"),
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "refused"),
    ({"candidates": []}, "empty_response"),
])
def test_gemini_unusable_responses(body, code):
    with pytest.raises(LLMError) as exc:
        gemini(lambda r: httpx.Response(200, json=body)).call("s", "u", "x", {})
    assert exc.value.code == code


def test_gemini_requires_key():
    with pytest.raises(LLMConfigError):
        GeminiProvider(api_key="", model="m", timeout=1, max_output_tokens=10)


# ---------------- OpenAI provider (kept for future use; never called for real) ----------------
class FakeCompletions:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def openai_provider(outcome):
    p = OpenAIProvider(api_key="sk-test", model="gpt-test", timeout=5, max_output_tokens=100)
    fake = FakeCompletions(outcome)
    p._client = type("C", (), {"chat": type("Ch", (), {"completions": fake})()})()
    return p, fake


def completion(content='{"ok": true}', finish="stop", refusal=None):
    msg = type("M", (), {"content": content, "refusal": refusal})()
    choice = type("Ch", (), {"message": msg, "finish_reason": finish})()
    usage = type("U", (), {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49})()
    return type("R", (), {"choices": [choice], "usage": usage})()


def test_openai_provider_sends_strict_schema():
    p, fake = openai_provider(completion())
    reply = p.call("sys", "user", "message_extraction", {"type": "object"})
    assert json.loads(reply.text) == {"ok": True} and reply.total_tokens == 49
    assert fake.kwargs["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("exc,code,retryable", [
    (openai.APITimeoutError(request=REQ), "timeout", True),
    (openai.RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None), "rate_limited", False),
    (openai.AuthenticationError("bad key sk-secret", response=httpx.Response(401, request=REQ), body=None), "auth_error", False),
])
def test_openai_errors_are_mapped(exc, code, retryable):
    p, _ = openai_provider(exc)
    with pytest.raises(LLMError) as err_info:
        p.call("s", "u", "x", {})
    assert err_info.value.code == code and err_info.value.retryable is retryable
    assert "sk-secret" not in err_info.value.detail


def test_mock_provider_responder_can_simulate_failure():
    c = LLMClient(MockProvider(responder=lambda s, u: LLMError("provider_unavailable", retryable=True)),
                  max_retries=1, sleep=lambda s: None)
    with pytest.raises(LLMError):
        c.complete_json("s", "u", "message_extraction", {})
