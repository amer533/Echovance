import json

import pytest
from django.core.management import CommandError, call_command

from feedback.ai.client import LLMClient, LLMError, ProviderReply
from feedback.ai.extraction import locate_evidence
from feedback.ai.privacy import sanitize_for_llm
from feedback.models import CustomerMessage, LLMRequestLog, MessageAnalysis
from feedback.services import llm_budget
from feedback.services.analysis import AnalysisService
from tests import factories

pytestmark = pytest.mark.django_db


class FakeRealProvider:
    """Behaves like a real provider (is_mock=False) without any network access."""
    name, model, is_mock = "gemini", "fake-flash", False

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.prompts = []

    def call(self, system, user, schema_name, schema):
        self.prompts.append(user)
        if self.outcomes:
            o = self.outcomes.pop(0)
            if isinstance(o, Exception):
                raise o
            return ProviderReply(o, 100, 20, 120)
        return ProviderReply(json.dumps({"message_type": "question", "is_complaint": False,
                                         "overall_sentiment": "neutral", "aspects": [], "needs_review": False,
                                         "review_reason": ""}), 100, 20, 120)


def real_service(provider=None):
    provider = provider or FakeRealProvider()
    return AnalysisService(LLMClient(provider, max_retries=3, sleep=lambda s: None,
                                     guard=llm_budget.RequestBudget())), provider


def test_initial_budget_is_three_of_twenty():
    s = llm_budget.status()
    assert (s["used"], s["authorized"], s["hard_limit"], s["remaining"], s["halted"]) == (0, 3, 20, 3, False)


def test_real_run_never_exceeds_budget_and_logs_usage():
    b = factories.batch()
    for i in range(6):
        factories.message(b, text=f"question {i}?")
    svc, provider = real_service()
    run = svc.run(limit=6)
    assert run.requested == 3 and run.succeeded == 3 and len(provider.prompts) == 3
    s = llm_budget.status()
    assert s["used"] == 3 and s["remaining"] == 0 and s["input_tokens"] == 300 and s["total_tokens"] == 360
    run2 = svc.run(limit=6)
    assert run2.requested == 0 and "No real LLM requests remaining" in run2.stopped_reason
    assert CustomerMessage.objects.filter(analysis_status="pending").count() == 3
    assert LLMRequestLog.objects.count() == 3  # persisted: survives restarts


def test_guard_refuses_beyond_budget_even_when_called_directly():
    guard = llm_budget.RequestBudget()
    for _ in range(3):
        guard.acquire("gemini", "m", "test")
    with pytest.raises(LLMError) as exc:
        guard.acquire("gemini", "m", "test")
    assert exc.value.code == "budget_exhausted"


def test_rate_limit_halts_without_retry_and_stops_run():
    b = factories.batch()
    for i in range(3):
        factories.message(b, text=f"msg {i}")
    provider = FakeRealProvider([LLMError("rate_limited", "429", retryable=True, http_status=429)])
    svc, _ = real_service(provider)
    run = svc.run(limit=3)
    assert len(provider.prompts) == 1  # no retry, no further messages
    assert run.failed == 1 and run.stopped_reason.startswith("Stopped: rate_limited")
    s = llm_budget.status()
    assert s["halted"] and s["used"] == 1 and s["errors"] == {"rate_limited": 1}
    # halted: next run sends nothing
    run2 = svc.run(limit=3)
    assert run2.requested == 0 and len(provider.prompts) == 1
    llm_budget.clear_halt()
    assert llm_budget.remaining() == 2


def test_transient_retries_consume_budget():
    factories.message(text="msg")
    provider = FakeRealProvider([LLMError("timeout", retryable=True), LLMError("timeout", retryable=True),
                                 LLMError("timeout", retryable=True)])
    svc, _ = real_service(provider)
    run = svc.run(limit=1)
    # 3 attempts admitted, 4th refused by the budget -> message not marked failed by a call never sent
    assert len(provider.prompts) == 3 and LLMRequestLog.objects.count() == 3
    assert run.stopped_reason.startswith("Stopped: budget_exhausted")
    assert MessageAnalysis.objects.count() == 0


def test_lifetime_ceiling_is_configurable_and_enforced(settings):
    # Budget v2 replaced the fixed DB constraint (<= 20) with LLM_LIFETIME_REQUEST_CEILING.
    llm_budget.status()
    with pytest.raises(ValueError):
        llm_budget.authorize(21)  # default ceiling 20
    settings.LLM = {**settings.LLM, "LIFETIME_REQUEST_CEILING": 25}
    assert llm_budget.authorize(25).authorized_requests == 25
    settings.LLM = {**settings.LLM, "LIFETIME_REQUEST_CEILING": 2}
    assert llm_budget.status()["remaining"] == 2  # ceiling caps even a larger authorisation


def test_budget_command_requires_confirmation():
    with pytest.raises(CommandError, match="--confirm"):
        call_command("llm_budget", "--authorize", "10")
    call_command("llm_budget", "--authorize", "10", "--confirm")
    assert llm_budget.status()["authorized"] == 10
    with pytest.raises(CommandError):
        call_command("llm_budget", "--authorize", "50", "--confirm")


def test_run_analysis_command_requires_allow_real(settings):
    settings.LLM = {**settings.LLM, "PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"}
    with pytest.raises(CommandError, match="--allow-real"):
        call_command("run_analysis", "--limit", "1")


def test_web_and_api_cap_real_analysis_to_one_message(client, api, analyst, settings, monkeypatch):
    b = factories.batch()
    for i in range(4):
        factories.message(b, text=f"q{i}?")
    provider = FakeRealProvider()
    monkeypatch.setattr("feedback.services.analysis.get_llm_client",
                        lambda: LLMClient(provider, guard=llm_budget.RequestBudget()))
    client.force_login(analyst)
    client.post("/analysis/run/", {"limit": 10})
    api.force_authenticate(analyst)
    r = api.post("/api/analyses/run/", {"limit": 10}, format="json")
    assert r.json()["requested"] == 1 and len(provider.prompts) == 2


# ---------------- outbound privacy ----------------
def test_outbound_prompt_is_redacted():
    factories.message(text="@SpotifyCares @115888 charged twice! mail a.b@example.com https://t.co/AbC123 call 555 123 4567")
    svc, provider = real_service()
    svc.run(limit=1)
    sent = provider.prompts[0]
    for secret in ("115888", "example.com", "t.co/AbC123", "555 123 4567", "SpotifyCares"):
        assert secret not in sent
    assert "@user" in sent and "[link]" in sent and "[email]" in sent


def test_evidence_with_placeholders_maps_back_to_stored_text():
    text = "@115888 my app crashes, see https://t.co/xyz  every   time"
    sanitized = sanitize_for_llm(text)
    assert sanitized == "@user my app crashes, see [link]  every   time"
    span = locate_evidence("@user my app crashes, see [link] every time", text)
    assert span == (0, len(text)) and text[span[0]:span[1]] == text
    assert locate_evidence("app crashes daily", text) is None
