"""Latency accounting: queue wait (budget admission / rate limiter), provider time and total are separate.

Regression for the pilot finding: MessageAnalysis.latency_ms used to include rate-limiter sleep and
was reported as model latency. A deterministic fake clock makes every value exact.
"""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from feedback.ai.client import GeminiProvider, LLMClient, LLMError, ProviderReply
from feedback.models import LLMRequestLog, MessageAnalysis
from feedback.services import llm_budget
from feedback.services.analysis import AnalysisService
from tests import factories


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.dt = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):  # perf_counter replacement
        return self.t

    def advance(self, seconds):
        self.t += seconds
        self.dt += timedelta(seconds=seconds)

    def now(self):  # timezone.now replacement for the budget
        return self.dt


class WaitingGuard:
    """Guard that 'sleeps' wait_s on admission (like the rate limiter)."""

    def __init__(self, clock, wait_s):
        self.clock, self.wait_s = clock, wait_s

    def acquire(self, *a, **k):
        self.clock.advance(self.wait_s)
        return 1

    def finish(self, *a, **k):
        self.finished = k

    def halt(self, reason):
        pass


class TimedProvider:
    name, model, is_mock = "timed", "t-1", False

    def __init__(self, clock, seconds, script=None):
        self.clock, self.seconds, self.script = clock, seconds, list(script or [])

    def call(self, *a):
        self.clock.advance(self.seconds)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
        return ProviderReply('{"ok": true}', 10, 2, 12)


def test_wait_provider_and_total_are_separate():
    clock = FakeClock()
    guard = WaitingGuard(clock, 6.0)
    c = LLMClient(TimedProvider(clock, 1.2), guard=guard, clock=clock)
    r = c.complete_json("s", "u", "x", {})
    assert (r.queue_wait_ms, r.provider_latency_ms, r.total_ms) == (6000, 1200, 7200)
    assert r.latency_ms == r.total_ms  # legacy field keeps its old meaning (total)
    assert guard.finished["latency_ms"] == 1200  # request log gets provider-only time


def test_retry_backoff_counts_in_total_not_provider():
    clock = FakeClock()
    provider = TimedProvider(clock, 0.5, script=[LLMError("timeout", retryable=True), None])
    c = LLMClient(provider, guard=WaitingGuard(clock, 1.0), clock=clock, max_retries=1,
                  backoff_seconds=2.0, sleep=clock.advance)
    r = c.complete_json("s", "u", "x", {})
    assert r.attempts == 2
    assert (r.queue_wait_ms, r.provider_latency_ms, r.total_ms) == (2000, 1000, 5000)  # 2 waits, 2 calls, 2s backoff


def test_failure_carries_timing():
    clock = FakeClock()
    provider = TimedProvider(clock, 0.3, script=[LLMError("bad_request", "x")])
    c = LLMClient(provider, guard=WaitingGuard(clock, 4.0), clock=clock)
    with pytest.raises(LLMError) as exc:
        c.complete_json("s", "u", "x", {})
    assert exc.value.timing == {"queue_wait_ms": 4000, "provider_latency_ms": 300, "total_ms": 4300}


def test_mock_provider_has_no_queue_wait():
    from feedback.ai.client import MockProvider
    clock = FakeClock()
    r = LLMClient(MockProvider(responder=lambda s, u: {"a": 1}), clock=clock).complete_json("s", "u", "x", {})
    assert r.queue_wait_ms == 0 and r.provider_latency_ms == 0


@pytest.mark.django_db
def test_rate_limited_job_records_true_provider_latency(settings):
    """End-to-end: real RequestBudget at 10 rpm + Gemini provider over mocked HTTP."""
    settings.ECHOVANCE_AI_MODE = "real"
    settings.LLM = {**settings.LLM, "PROVIDER": "gemini", "GEMINI_TIER": "free", "RATE_LIMIT_RPM": 10}
    llm_budget.authorize(5)
    clock = FakeClock()
    b = factories.batch()
    for i in range(2):
        factories.message(b, text=f"question {i}?", created=datetime(2017, 10, 9, tzinfo=timezone.utc) + timedelta(minutes=i))

    def handler(request):
        clock.advance(1.5)  # the provider takes 1.5 s
        out = {"message_type": "question", "is_complaint": False, "overall_sentiment": "neutral", "aspects": [],
               "needs_review": False, "review_reason": ""}
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(out)}]},
                                                         "finishReason": "STOP"}],
                                         "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 50}})

    provider = GeminiProvider(api_key="k", model="gemini-test", timeout=5, max_output_tokens=256,
                              transport=httpx.MockTransport(handler))
    guard = llm_budget.RequestBudget(sleep=clock.advance, now=clock.now)
    AnalysisService(LLMClient(provider, guard=guard, clock=clock)).run(limit=5)
    first, second = MessageAnalysis.objects.order_by("message__created_at")
    assert (first.provider_latency_ms, first.queue_wait_ms) == (1500, 0)
    # 10 rpm -> 6 s spacing; 1.5 s already elapsed during the first call -> 4.5 s rate-limit wait
    assert (second.provider_latency_ms, second.queue_wait_ms, second.total_ms) == (1500, 4500, 6000)
    assert second.latency_ms == second.total_ms
    logs = list(LLMRequestLog.objects.order_by("id"))
    assert [lg.latency_ms for lg in logs] == [1500, 1500] and [lg.queue_wait_ms for lg in logs] == [0, 4500]
