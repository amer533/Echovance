"""Regression tests for the real-AI upgrade.

The REAL OpenAIProvider code path (openai SDK) is exercised over an httpx.MockTransport (no network): request
building, structured-output parsing, usage accounting, budget admission, privacy gate, promotion of
mock analyses, jobs, provenance, stale alerts, hybrid grouping, action-v2, corrections, evaluation.
"""
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest
from django.core.exceptions import PermissionDenied

from feedback.ai.client import LLMClient, LLMConfigError, MockProvider, get_llm_client
from feedback.ai.privacy import redact
from feedback.models import (ActionItem, AnalysisJob, CustomerMessage, EvaluationLabel, IssueMention,
                             LLMRequestLog, MessageAnalysis)
from feedback.services import actions as action_service
from feedback.services import corrections, evaluation, grouping, jobs, llm_budget, stats, trends
from feedback.services.analysis import AnalysisService
from tests import factories
from tests.fake_openai import FakeOpenAI
from tests.fake_openai import provider as openai_provider

pytestmark = pytest.mark.django_db
T0 = datetime(2017, 10, 4, tzinfo=timezone.utc)
CHARGE = "billing_subscription.unexpected_charge"
STUDENT = "billing_subscription.student_plan"


@pytest.fixture
def real_mode(settings):
    settings.ECHOVANCE_AI_MODE = "real"
    settings.LLM = {**settings.LLM, "PROVIDER": "openai", "OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-4o-mini",
                    "RATE_LIMIT_RPM": 60000, "MAX_RETRIES": 0}
    return settings


def openai_client(server: FakeOpenAI, job=None) -> LLMClient:
    return LLMClient(openai_provider(server), max_retries=0, guard=llm_budget.RequestBudget(job=job),
                     sleep=lambda s: None)


def authorize(n):
    llm_budget.authorize(n)


# ------------------------------------------------------------------ mode / no fallback
def test_real_mode_refuses_mock_provider(settings):
    settings.ECHOVANCE_AI_MODE = "real"
    settings.LLM = {**settings.LLM, "PROVIDER": "mock"}
    with pytest.raises(LLMConfigError, match="disabled in real AI mode"):
        get_llm_client()


def test_real_mode_missing_key_fails_closed(real_mode):
    real_mode.LLM = {**real_mode.LLM, "OPENAI_API_KEY": ""}
    with pytest.raises(LLMConfigError):
        get_llm_client()


def test_real_pipeline_with_mocked_http(real_mode):
    authorize(3)
    b = factories.batch()
    m1 = factories.message(b, text="@SpotifyCares I was charged twice this month")
    factories.message(b, text="@SpotifyCares how do I change my email?")
    server = FakeOpenAI()
    run = AnalysisService(openai_client(server)).run(limit=10)
    assert run.succeeded == 2 and len(server.calls) == 2
    a = m1.analyses.get()
    assert (a.source, a.provider, a.model_name, a.is_current, a.input_tokens) == ("llm", "openai", "gpt-4o-mini", True, 1000)
    mention = IssueMention.objects.get(message=m1)
    assert mention.source == "llm" and m1.text[mention.evidence_start:mention.evidence_end] == "charged twice"
    log = LLMRequestLog.objects.order_by("id").first()
    assert log.status == "succeeded" and log.input_tokens == 1000 and log.output_tokens == 100
    assert log.cost_usd == Decimal("0.000210")  # gpt-4o-mini: (1000 * $0.15 + 100 * $0.60) / 1M
    assert log.request_fingerprint and log.estimated_input_tokens > 0
    assert stats.provenance()["real_analysed"] == 2


def test_provider_failure_never_falls_back_to_mock(real_mode):
    authorize(3)
    m = factories.message(text="I was charged twice")
    server = FakeOpenAI(fail_with=(503, None, "overloaded"))
    run = AnalysisService(openai_client(server)).run(limit=5)
    m.refresh_from_db()
    assert run.failed == 1 and m.analysis_status == "failed"
    assert not MessageAnalysis.objects.filter(source="mock").exists()
    assert not IssueMention.objects.exists() and not m.analyses.filter(is_current=True).exists()
    assert stats.provenance()["real_failed"] == 1


# ------------------------------------------------------------------ promotion mock -> real
def _mock_analyse(*texts):
    b = factories.batch()
    msgs = [factories.message(b, text=t, created=T0 + timedelta(hours=i)) for i, t in enumerate(texts)]
    AnalysisService(LLMClient(MockProvider(), max_retries=0)).run(limit=50)
    return msgs


def test_promote_mock_supersedes_atomically_and_keeps_history(real_mode, settings):
    settings.ECHOVANCE_AI_MODE = "demo"
    (m,) = _mock_analyse("@SpotifyCares I was charged twice for premium, please refund")
    mock_analysis = m.analyses.get(is_current=True)
    assert mock_analysis.source == "mock" and m.mentions.filter(is_active=True, source="mock").exists()
    settings.ECHOVANCE_AI_MODE = "real"
    # default 'pending' selection would never pick an analysed message (the original gap)
    assert AnalysisService(openai_client(FakeOpenAI())).select_messages(10) == []
    authorize(3)
    run = AnalysisService(openai_client(FakeOpenAI())).run(limit=10, selection="promote_mock")
    assert run.succeeded == 1
    m.refresh_from_db()
    current = m.analyses.get(is_current=True)
    assert current.source == "llm" and MessageAnalysis.objects.filter(pk=mock_analysis.pk, is_current=False).exists()
    assert not IssueMention.objects.filter(analysis=mock_analysis, is_active=True).exists()
    assert IssueMention.objects.filter(analysis=mock_analysis).exists()  # history kept
    assert list(IssueMention.objects.filter(message=m, is_active=True).values_list("source", flat=True)) == ["llm"]


def test_failed_promotion_keeps_mock_out_of_real_metrics(real_mode, settings):
    settings.ECHOVANCE_AI_MODE = "demo"
    (m,) = _mock_analyse("I was charged twice")
    settings.ECHOVANCE_AI_MODE = "real"
    authorize(3)
    AnalysisService(openai_client(FakeOpenAI(fail_with=(500, None, "boom")))).run(limit=5, selection="promote_mock")
    assert m.analyses.get(is_current=True).source == "mock"
    failed = m.analyses.get(source="llm")
    assert failed.status == "failed" and not failed.is_current
    assert stats.overview("real")["analysed_in_scope"] == 0 and stats.provenance()["mock_only"] == 1


# ------------------------------------------------------------------ real-only analytics
def test_dashboard_never_mixes_mock_and_real(settings):
    settings.ECHOVANCE_AI_MODE = "demo"
    _mock_analyse("charged twice again", "charged twice please refund")
    factories.analysed_message(factories.batch(), T0, (STUDENT,), "student plan broken")  # real (llm)
    real, demo = stats.overview("real"), stats.overview("demo")
    assert real["analysed_in_scope"] == 1 and demo["analysed_in_scope"] == 2
    assert {c["code"] for c in real["categories"]} == {"billing_subscription"}
    assert [i.key for i in stats.issues_with_counts("real")] == [STUDENT]
    assert CHARGE in [i.key for i in stats.issues_with_counts("demo")]
    p = stats.provenance()
    assert (p["real_analysed"], p["mock_only"], p["total"]) == (1, 2, 3)


def test_api_scope_parameter(api, viewer, settings):
    settings.ECHOVANCE_AI_MODE = "demo"
    _mock_analyse("charged twice")
    api.force_authenticate(viewer)
    assert api.get("/api/dashboard/summary/").json()["analysed_in_scope"] == 0
    assert api.get("/api/dashboard/summary/", {"scope": "demo"}).json()["analysed_in_scope"] == 1
    assert api.get("/api/dashboard/summary/", {"scope": "everything"}).json()["scope"] == "real"


# ------------------------------------------------------------------ budget v2
def test_job_limits_requests_tokens_and_cost(real_mode, manager):
    authorize(10)
    b = factories.batch()
    for i in range(5):
        factories.message(b, text=f"question {i}?")
    job = jobs.create_job(user=manager, selection="pending", max_messages=5, max_requests=2,
                          client=openai_client(FakeOpenAI()))
    jobs.authorize(job, manager)
    done = jobs.run_job(job.pk, client=openai_client(FakeOpenAI(), job=job))
    assert done.status == "stopped" and "job_request_limit" in done.stop_reason
    assert done.requests_used == 2 and LLMRequestLog.objects.filter(job=job).count() == 2
    assert CustomerMessage.objects.filter(analysis_status="pending").count() == 3  # untouched, resumable


def test_job_token_limit(real_mode, manager):
    authorize(10)
    factories.message(text="question?")
    job = jobs.create_job(user=manager, selection="pending", max_messages=1, max_input_tokens=10,
                          client=openai_client(FakeOpenAI()))
    jobs.authorize(job, manager)
    done = jobs.run_job(job.pk, client=openai_client(FakeOpenAI(), job=job))
    assert "job_token_limit" in done.stop_reason and LLMRequestLog.objects.count() == 0


def test_default_openai_pricing_admits_requests_within_the_cost_ceiling(real_mode):
    """gpt-4o-mini prices + the default $1.00 monthly ceiling must not silently block OpenAI requests."""
    authorize(10)
    assert llm_budget.prices("openai") == (Decimal("0.15"), Decimal("0.60"))
    assert llm_budget.estimate_cost("openai", 2000, 1024) == Decimal("0.0009144")  # (2000*0.15 + 1024*0.60) / 1M
    llm_budget.RequestBudget().acquire("openai", "gpt-4o-mini", "t", estimated_input_tokens=2000)
    assert LLMRequestLog.objects.count() == 1


def test_missing_prices_are_refused_and_monthly_ceiling_enforced(real_mode):
    authorize(10)
    real_mode.LLM = {**real_mode.LLM, "OPENAI_PRICE_INPUT_PER_MTOK": "", "OPENAI_PRICE_OUTPUT_PER_MTOK": ""}
    guard = llm_budget.RequestBudget()
    with pytest.raises(llm_budget.BudgetError) as exc:
        guard.acquire("openai", "m", "t", estimated_input_tokens=100)
    assert exc.value.code == "pricing_unknown"
    real_mode.LLM = {**real_mode.LLM, "OPENAI_PRICE_INPUT_PER_MTOK": "0.15", "OPENAI_PRICE_OUTPUT_PER_MTOK": "0.60",
                     "MONTHLY_COST_CEILING_USD": "0.0001"}
    with pytest.raises(llm_budget.BudgetError) as exc:  # 100 in + 1024 out tokens = $0.000629 > $0.0001
        guard.acquire("openai", "m", "t", estimated_input_tokens=100)
    assert exc.value.code == "cost_ceiling_reached" and LLMRequestLog.objects.count() == 0


def test_rate_limit_waits_between_requests(real_mode):
    authorize(5)
    real_mode.LLM = {**real_mode.LLM, "RATE_LIMIT_RPM": 6}
    waits, clock = [], [datetime(2026, 1, 1, tzinfo=timezone.utc)]

    def sleep(sec):
        waits.append(sec)
        clock[0] += timedelta(seconds=sec)

    guard = llm_budget.RequestBudget(sleep=sleep, now=lambda: clock[0])
    guard.acquire("openai", "m", "t")
    guard.acquire("openai", "m", "t")
    assert waits and abs(sum(waits) - 10.0) < 0.01  # 6 rpm -> 10 s spacing


def test_authorisation_cannot_exceed_global_budget(real_mode, manager):
    authorize(2)
    job = jobs.create_job(user=manager, selection="pending", max_messages=5, client=openai_client(FakeOpenAI()))
    with pytest.raises(action_service.ActionError, match="global real-request budget"):
        jobs.authorize(job, manager)
    job.refresh_from_db()
    assert job.status == "draft"


def test_interrupted_requests_are_closed_but_still_counted(real_mode):
    authorize(5)
    t = llm_budget.RequestBudget().acquire("openai", "m", "t")
    LLMRequestLog.objects.filter(pk=t).update(created_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert llm_budget.close_interrupted() == 1
    assert LLMRequestLog.objects.get(pk=t).status == "interrupted" and llm_budget.status()["used"] == 1


# ------------------------------------------------------------------ jobs: permissions, resume, cancel
def test_job_requires_manager_authorisation(real_mode, analyst, manager):
    job = jobs.create_job(user=analyst, selection="pending", max_messages=1, client=openai_client(FakeOpenAI()))
    with pytest.raises(PermissionDenied):
        jobs.authorize(job, analyst)
    assert jobs.run_job(job.pk, client=openai_client(FakeOpenAI())).status == "draft"  # draft never runs
    with pytest.raises(PermissionDenied):
        jobs.create_job(user=factories.user("v2", "Viewers"), selection="pending", max_messages=1,
                        client=openai_client(FakeOpenAI()))


def test_paused_job_resumes_without_reprocessing(real_mode, manager):
    authorize(10)
    b = factories.batch()
    msgs = [factories.message(b, text=f"question {i}?", created=T0 + timedelta(minutes=i)) for i in range(4)]
    job = jobs.create_job(user=manager, selection="pending", max_messages=4, client=openai_client(FakeOpenAI()))
    jobs.authorize(job, manager)

    def pause_after_two(n):
        if n == 2:
            AnalysisJob.objects.filter(pk=job.pk).update(status="paused")
        return None

    server = FakeOpenAI(fail_with=pause_after_two)
    first = jobs.run_job(job.pk, client=openai_client(server, job=job))
    assert first.status == "paused" and first.processed == 2
    jobs.resume(first, manager)
    server2 = FakeOpenAI()
    done = jobs.run_job(job.pk, client=openai_client(server2, job=job))
    assert done.status == "completed" and done.processed == 4 and len(server2.calls) == 2
    assert MessageAnalysis.objects.filter(message__in=msgs).count() == 4  # nothing analysed twice


def test_cancelled_job_stops(real_mode, manager):
    job = jobs.create_job(user=manager, selection="pending", max_messages=1, client=openai_client(FakeOpenAI()))
    jobs.cancel(job, manager)
    assert jobs.run_job(job.pk, client=openai_client(FakeOpenAI())).status == "cancelled"
    with pytest.raises(action_service.ActionError):
        jobs.authorize(job, manager)


def test_job_never_retries_a_failed_message_within_itself(real_mode, manager):
    authorize(10)
    factories.message(text="question?")
    job = jobs.create_job(user=manager, selection="pending", max_messages=5, client=openai_client(FakeOpenAI()))
    jobs.authorize(job, manager)
    server = FakeOpenAI(fail_with=(400, "invalid_request", "bad"))
    done = jobs.run_job(job.pk, client=openai_client(server, job=job))
    assert len(server.calls) == 1 and done.failed == 1 and done.status == "completed"


def test_process_jobs_command_runs_only_authorised(real_mode, manager, monkeypatch):
    from django.core.management import call_command
    authorize(3)
    factories.message(text="question?")
    draft = jobs.create_job(user=manager, selection="pending", max_messages=1, client=openai_client(FakeOpenAI()))
    server = FakeOpenAI()
    monkeypatch.setattr("feedback.services.jobs.get_llm_client", lambda: openai_client(server))
    call_command("process_jobs")
    assert not server.calls and AnalysisJob.objects.get(pk=draft.pk).status == "draft"
    jobs.authorize(draft, manager)
    call_command("process_jobs")
    assert len(server.calls) == 1 and AnalysisJob.objects.get(pk=draft.pk).status == "completed"


# ------------------------------------------------------------------ dedup
def test_identical_inputs_are_reused_not_resent(real_mode):
    authorize(5)
    b = factories.batch()
    m1 = factories.message(b, text="@111 I was charged twice", created=T0)
    m2 = factories.message(b, text="@222 I was charged twice", created=T0 + timedelta(hours=1))  # same after redaction
    server = FakeOpenAI()
    run = AnalysisService(openai_client(server)).run(limit=5)
    assert run.succeeded == 2 and len(server.calls) == 1
    a2 = m2.analyses.get()
    assert a2.reused_from == m1.analyses.get() and a2.input_tokens == 0
    mention = IssueMention.objects.get(message=m2)
    assert m2.text[mention.evidence_start:mention.evidence_end] == "charged twice"


# ------------------------------------------------------------------ privacy gate
def test_restricted_policy_never_sends_internal_or_confidential_data(real_mode):
    authorize(5)
    real_mode.LLM = {**real_mode.LLM, "ALLOWED_DATA_CLASSES": {"openai:paid": ["public"]}}  # LLM_PAID_ALLOWED_DATA_CLASSES=public
    internal = factories.message(factories.batch("internal"), text="I was charged twice")
    confidential = factories.message(factories.batch("confidential"), text="I was charged twice too")
    server = FakeOpenAI()
    run = AnalysisService(openai_client(server)).run(limit=5)
    assert server.calls == [] and LLMRequestLog.objects.count() == 0 and run.failed == 2
    for m in (internal, confidential):
        m.refresh_from_db()
        assert m.analysis_status == "privacy_blocked"
        assert m.analyses.get().error_code == "privacy_blocked"
    assert stats.provenance()["privacy_blocked"] == 2


def test_openai_policy_allows_internal_but_never_confidential(real_mode):
    authorize(5)
    real_mode.LLM = {**real_mode.LLM, "OPENAI_PRICE_INPUT_PER_MTOK": "0.1", "OPENAI_PRICE_OUTPUT_PER_MTOK": "0.4",
                     "MONTHLY_COST_CEILING_USD": "5", "ALLOWED_DATA_CLASSES": {"openai:paid": ["public", "internal"]}}
    factories.message(factories.batch("internal"), text="question one?")
    factories.message(factories.batch("confidential"), text="question two?")
    server = FakeOpenAI()
    AnalysisService(openai_client(server)).run(limit=5)
    assert len(server.calls) == 1
    assert LLMRequestLog.objects.get().cost_usd == Decimal("0.000140")  # 1000*0.1 + 100*0.4 per 1M


def test_outbound_prompt_redacts_identifiers(real_mode):
    authorize(5)
    factories.message(text="My name is Jane Doe, order AB12CD34EF, card 4111 1111 1111 1111. charged twice! Thanks, Chad")
    server = FakeOpenAI()
    AnalysisService(openai_client(server)).run(limit=1)
    sent = json.dumps(server.calls[0])
    for secret in ("Jane Doe", "AB12CD34EF", "4111", "Chad"):
        assert secret not in sent
    assert MessageAnalysis.objects.get().redaction == {"name": 2, "card": 1, "id": 1}


def test_residual_risk_blocks_sending(real_mode):
    authorize(5)
    factories.message(text="my passport number is X")
    server = FakeOpenAI()
    AnalysisService(openai_client(server)).run(limit=1)
    assert server.calls == [] and "passport" not in redact("my passport number is X").counts


# ------------------------------------------------------------------ stale alerts / revisions
RULES = {"RULE_VERSION": "test-v2", "WINDOW_DAYS": 7, "BASELINE_WINDOWS": 3, "MIN_WINDOW_ANALYZED_MESSAGES": 20,
         "MIN_CURRENT_MENTIONS": 8, "MIN_ABSOLUTE_INCREASE": 5, "MIN_GROWTH_RATIO": 1.5, "MIN_SHARE_RATIO": 1.3,
         "MIN_EMERGING_MENTIONS": 6, "EVIDENCE_LIMIT": 5}
END = datetime(2017, 11, 1, tzinfo=timezone.utc)


def _surge():
    b = factories.batch()
    start = T0
    for counts in [{CHARGE: 4}, {CHARGE: 5}, {CHARGE: 3}, {CHARGE: 14}]:
        factories.fill_window(b, start, 7, 30, counts)
        start += timedelta(days=7)


def test_reanalysis_marks_alert_stale_and_creates_revision(manager):
    _surge()
    alert = trends.TrendService(RULES).detect(END).created[0]
    from feedback.services import review
    review.review_alert(alert, manager, "acknowledged", "looking into it")
    action = action_service.create_manual(user=manager, issue=alert.issue, alert=alert, title="t", description="d")
    # a human correction removes one contributing mention
    m = IssueMention.objects.filter(issue=alert.issue, message_created_at__gte=alert.window_start, is_active=True).first()
    corrections.correct_analysis(m.message, manager, message_type="question", aspects=[])
    alert.refresh_from_db()
    assert alert.is_stale and "changed" in alert.stale_reason
    assert alert.status == "acknowledged" and alert.review_note == "looking into it" and alert.current_count == 14
    assert ActionItem.objects.get(pk=action.pk).alert_id == alert.pk  # linked action preserved
    res = trends.TrendService(RULES).detect(END)
    new = res.created[0]
    alert.refresh_from_db()
    assert new.revision == 2 and new.current_count == 13 and alert.superseded_by_id == new.pk
    again = trends.TrendService(RULES).detect(END)
    assert again.created == [] and again.existing == [new]


def test_scopes_have_separate_alerts(settings):
    _surge()
    assert trends.TrendService(RULES, scope="demo").detect(END).sufficient_data is False
    real = trends.TrendService(RULES, scope="real").detect(END)
    assert real.created and real.created[0].data_scope == "real"


# ------------------------------------------------------------------ hybrid grouping
def _real_mention(issue_key, summary, evidence, when=T0):
    text = f"{evidence} {summary}"
    m = factories.analysed_message(factories.batch(), when, (), text=text)
    a = m.analyses.get()
    cat, sub = issue_key.split(".")
    return IssueMention.objects.create(issue=grouping.get_or_create_issue(cat, sub), analysis=a, message=m,
                                       message_created_at=when, sentiment="negative", summary=summary,
                                       evidence_text=evidence, evidence_start=0, evidence_end=len(evidence), source="llm")


def test_similar_complaint_filed_under_wrong_subtype_gets_suggestion_not_merge():
    for i in range(3):
        _real_mention(STUDENT, f"Student discount verification fails {i}", "student verification keeps failing unidays")
    odd = _real_mention(CHARGE, "Student discount verification failed", "student verification keeps failing on unidays")
    created = grouping.suggest_for_mentions([odd.pk])
    assert len(created) == 1 and created[0].suggested_issue.key == STUDENT
    odd.refresh_from_db()
    assert odd.issue.key == CHARGE  # nothing merged automatically


def test_dissimilar_and_cross_category_mentions_are_never_suggested():
    for i in range(3):
        _real_mention("playback_performance.offline_downloads", "Offline songs disappear", f"offline downloads vanish {i}")
    m = _real_mention(CHARGE, "Duplicate charge", "charged twice this month")
    assert grouping.suggest_for_mentions([m.pk]) == []  # different category: never compared
    for i in range(3):
        _real_mention(STUDENT, "Student discount", f"student discount unidays {i}")
    m2 = _real_mention(CHARGE, "Duplicate charge", "charged twice on my card")
    assert grouping.suggest_for_mentions([m2.pk]) == []  # same category but not similar


def test_accept_and_reject_suggestions(analyst, viewer):
    for i in range(3):
        _real_mention(STUDENT, f"Student discount verification fails {i}", "student verification keeps failing unidays")
    odd = _real_mention(CHARGE, "Student discount verification failed", "student verification keeps failing on unidays")
    s = grouping.suggest_for_mentions([odd.pk])[0]
    with pytest.raises(PermissionDenied):
        grouping.accept_suggestion(s, viewer)
    grouping.accept_suggestion(s, analyst, "subtype mistake")
    odd.refresh_from_db()
    s.refresh_from_db()
    assert odd.issue.key == STUDENT and odd.original_issue.key == CHARGE and odd.regrouped_by == analyst
    assert s.status == "accepted" and s.review_note == "subtype mistake"
    with pytest.raises(action_service.TransitionError):
        grouping.reject_suggestion(s, analyst)


# ------------------------------------------------------------------ actions: real AI, human approval
def test_real_action_proposal_cites_real_evidence_only(real_mode, manager, settings):
    authorize(5)
    issue = grouping.get_or_create_issue("billing_subscription", "unexpected_charge")
    settings.ECHOVANCE_AI_MODE = "demo"
    _mock_analyse("I was charged twice for premium")  # mock evidence for the same issue
    settings.ECHOVANCE_AI_MODE = "real"
    with pytest.raises(action_service.SuggestionError, match="real AI"):
        action_service.suggest_with_ai(user=manager, issue=issue, client=openai_client(FakeOpenAI()))
    real = factories.analysed_message(factories.batch(), T0, (CHARGE,), "charged twice!! fix")
    server = FakeOpenAI()
    action = action_service.suggest_with_ai(user=manager, issue=issue, client=openai_client(server))
    assert action.status == "proposed" and action.origin == "ai" and not action.ai_is_mock
    assert action.steps == ["Pull billing logs", "Reproduce"] and action.limitations and action.uncertainty == "medium"
    assert list(action.evidence.values_list("message_id", flat=True)) == [real.pk]
    assert action.decided_by is None and action.assignee is None
    assert "charged tw" in json.dumps(server.calls[0]) and LLMRequestLog.objects.filter(purpose="action_suggestion").count() == 1


# ------------------------------------------------------------------ human corrections
def test_human_correction_requires_verbatim_evidence(analyst, viewer):
    m = factories.message(text="Premium stopped working after I paid")
    with pytest.raises(PermissionDenied):
        corrections.correct_analysis(m, viewer, message_type="complaint", aspects=[])
    with pytest.raises(action_service.ActionError, match="Evidence must be copied"):
        corrections.correct_analysis(m, analyst, message_type="complaint", aspects=[
            {"category": "billing_subscription", "subtype": "premium_not_activated", "evidence": "never paid"}])
    a = corrections.correct_analysis(m, analyst, message_type="complaint", aspects=[
        {"category": "billing_subscription", "subtype": "premium_not_activated", "evidence": "Premium stopped working"}])
    assert a.source == "human" and a.is_current and a.corrected_by == analyst
    assert stats.provenance()["human_corrected"] == 1 and stats.overview("real")["analysed_in_scope"] == 1


# ------------------------------------------------------------------ evaluation workflow
def test_blind_labelling_gold_and_kappa(analyst, manager, monkeypatch):
    msgs = [factories.message(text=f"msg {i} charged twice") for i in range(3)]
    monkeypatch.setattr(evaluation, "eval_sample_ids", lambda: [m.source_message_id for m in msgs])
    assert evaluation.next_for_labelling(analyst) == msgs[0]
    evaluation.save_label(analyst, msgs[0], "complaint", [CHARGE])
    evaluation.save_label(manager, msgs[0], "complaint", [CHARGE])
    evaluation.save_label(analyst, msgs[1], "complaint", [CHARGE])
    evaluation.save_label(manager, msgs[1], "question", [])
    with pytest.raises(PermissionDenied):
        evaluation.save_label(analyst, msgs[1], "question", [], role="adjudicator")
    gold, agreement = evaluation.gold_from_db(min_annotators=2)
    assert [g.tweet_id for g in gold] == [msgs[0].source_message_id]
    assert agreement["unresolved_disagreements"] == 1 and agreement["double_labelled"] == 2
    evaluation.save_label(manager, msgs[1], "complaint", [CHARGE], role="adjudicator")
    gold, _ = evaluation.gold_from_db(min_annotators=2)
    assert len(gold) == 2
    other = factories.message(text="not in the sample")
    with pytest.raises(ValueError):
        evaluation.save_label(analyst, other, "question", [])


def test_evaluation_uses_only_real_llm_predictions(real_mode, settings):
    settings.ECHOVANCE_AI_MODE = "demo"
    (m,) = _mock_analyse("I was charged twice")
    gold = [evaluation.GoldItem(m.source_message_id, "complaint", {CHARGE})]
    assert evaluation.evaluate(gold)["status"] == "refused"
    settings.ECHOVANCE_AI_MODE = "real"
    authorize(3)
    AnalysisService(openai_client(FakeOpenAI())).run(limit=5, selection="promote_mock")
    r = evaluation.evaluate(gold)
    assert r["status"] == "measured" and r["complaint_detection"]["f1"] == 1.0
    assert r["real_usage"]["requests"] == 1 and r["real_usage"]["mean_input_tokens"] == 1000


def test_labelling_page_is_blind(client, analyst, monkeypatch, settings):
    settings.ECHOVANCE_AI_MODE = "demo"
    (m,) = _mock_analyse("I was charged twice for premium")
    monkeypatch.setattr(evaluation, "eval_sample_ids", lambda: [m.source_message_id])
    client.force_login(analyst)
    html = client.get("/evaluation/label/").content.decode()
    assert "charged twice for premium" in html
    assert "[MOCK]" not in html and "Extracted issues" not in html and "keyword rule" not in html  # blind
    client.post("/evaluation/label/", {"message": m.pk, "message_type": "complaint", "issue_keys": [CHARGE]})
    assert EvaluationLabel.objects.get().issue_keys == [CHARGE]


@pytest.mark.parametrize("message", ["abc", "", "999999"])
def test_labelling_rejects_unknown_message_with_404(client, analyst, message):
    client.force_login(analyst)
    assert client.post("/evaluation/label/", {"message": message, "message_type": "complaint"}).status_code == 404


def test_create_job_command_makes_seeded_eval_draft_without_sending(real_mode, analyst, monkeypatch):
    from django.core.management import call_command
    msgs = [factories.message(text=f"eval {i}?") for i in range(5)]
    monkeypatch.setattr("feedback.management.commands.create_job.eval_sample_ids",
                        lambda: [m.source_message_id for m in msgs])
    call_command("create_job", "--user", analyst.username, "--eval-sample", "3")
    job = AnalysisJob.objects.get()
    assert job.status == "draft" and job.selection == "message_ids" and len(job.message_ids) == 3
    assert job.provider == "openai" and LLMRequestLog.objects.count() == 0
    call_command("create_job", "--user", analyst.username, "--eval-sample", "3")
    assert AnalysisJob.objects.order_by("id").last().message_ids == job.message_ids  # seeded -> reproducible


# ------------------------------------------------------------------ pilot support: dev split + stop_on_error
def test_dev_sample_selects_only_unused_dev_messages_and_never_reads_eval(real_mode, analyst, monkeypatch):
    from django.core.management import call_command
    msgs = [factories.message(text=f"dev {i}?", created=T0 + timedelta(minutes=i)) for i in range(6)]
    used = msgs[0]
    MessageAnalysis.objects.create(message=used, provider="openai", model_name="m", prompt_version="v", is_mock=False,
                                   source="llm", status="succeeded", is_current=True)
    monkeypatch.setattr("feedback.services.jobs.dev_sample_ids", lambda: [m.source_message_id for m in msgs])

    def eval_must_not_be_read():
        raise AssertionError("held-out evaluation sample was accessed")

    monkeypatch.setattr("feedback.services.evaluation.eval_sample_ids", eval_must_not_be_read)
    monkeypatch.setattr("feedback.management.commands.create_job.eval_sample_ids", eval_must_not_be_read)
    call_command("create_job", "--user", analyst.username, "--dev-sample", "4", "--max-requests", "4", "--stop-on-error")
    job = AnalysisJob.objects.get()
    assert len(job.message_ids) == 4 and used.pk not in job.message_ids
    assert set(job.message_ids) <= {m.pk for m in msgs} and job.stop_on_error and job.max_requests == 4
    call_command("create_job", "--user", analyst.username, "--dev-sample", "4")
    assert sorted(AnalysisJob.objects.order_by("id").last().message_ids) == sorted(job.message_ids)  # seeded
    with pytest.raises(Exception, match="Only 5 unused"):
        call_command("create_job", "--user", analyst.username, "--dev-sample", "6")


def test_stop_on_error_halts_at_first_provider_error(real_mode, manager):
    authorize(10)
    b = factories.batch()
    msgs = [factories.message(b, text=f"question {i}?", created=T0 + timedelta(minutes=i)) for i in range(4)]
    job = jobs.create_job(user=manager, selection="message_ids", message_ids=[m.pk for m in msgs], max_messages=4,
                          stop_on_error=True, client=openai_client(FakeOpenAI()))
    jobs.authorize(job, manager)
    server = FakeOpenAI(fail_with=lambda n: (400, "invalid_request", "bad") if n == 2 else None)
    done = jobs.run_job(job.pk, client=openai_client(server, job=job))
    assert done.status == "stopped" and "stop_on_error" in done.stop_reason
    assert len(server.calls) == 2 and done.processed == 2 and done.failed == 1


def test_stop_on_error_halts_at_first_privacy_block(real_mode, manager):
    authorize(10)
    confidential = factories.message(factories.batch("confidential"), text="question a?", created=T0)
    public = factories.message(factories.batch("public"), text="question b?", created=T0 + timedelta(minutes=1))
    job = jobs.create_job(user=manager, selection="message_ids", message_ids=[confidential.pk, public.pk],
                          max_messages=2, stop_on_error=True, client=openai_client(FakeOpenAI()))
    jobs.authorize(job, manager)
    server = FakeOpenAI()
    done = jobs.run_job(job.pk, client=openai_client(server, job=job))
    assert done.status == "stopped" and "privacy_blocked" in done.stop_reason and server.calls == []


def test_default_job_limits_afford_the_longest_message_at_openai_prices(real_mode, manager):
    """Default job limits must cover the worst-case request admission (paid pricing exposed a gap)."""
    authorize(5)
    long = factories.message(text="charged twice " + "and still waiting for help " * 60)  # > LLM_MAX_INPUT_CHARS
    job = jobs.create_job(user=manager, selection="message_ids", message_ids=[long.pk], max_messages=1,
                          client=openai_client(FakeOpenAI()))
    assert job.max_cost_usd == Decimal("0.0011")  # $0.00107565 worst case, rounded UP to 4 decimals
    jobs.authorize(job, manager)
    server = FakeOpenAI()
    done = jobs.run_job(job.pk, client=openai_client(server, job=job))
    assert done.status == "completed" and done.processed == 1 and len(server.calls) == 1
    assert done.needs_review == 1  # truncated to LLM_MAX_INPUT_CHARS -> flagged for review, by design
    assert done.cost_usd == Decimal("0.000210") and done.cost_usd <= done.max_cost_usd


def test_job_request_cap_holds_even_with_more_messages(real_mode, manager):
    authorize(13)
    b = factories.batch()
    msgs = [factories.message(b, text=f"question {i}?", created=T0 + timedelta(minutes=i)) for i in range(12)]
    job = jobs.create_job(user=manager, selection="message_ids", message_ids=[m.pk for m in msgs], max_messages=12,
                          max_requests=10, stop_on_error=True, client=openai_client(FakeOpenAI()))
    jobs.authorize(job, manager)
    server = FakeOpenAI()
    done = jobs.run_job(job.pk, client=openai_client(server, job=job))
    assert len(server.calls) == 10 and done.requests_used == 10 and "job_request_limit" in done.stop_reason
