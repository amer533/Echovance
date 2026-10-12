"""extract-v2 behaviour (offline): symptoms vs independent issues, unconfirmed causes, uncertain subtypes,
ambiguous and sarcastic complaints, split/leakage guards, and preservation of extract-v1 results.

All messages here are INVENTED (they mirror patterns seen in the development-split pilot without copying
any dataset text). Model outputs are simulated; the application-side validator is what is under test.
"""
import pytest

from feedback.ai import extraction
from feedback.ai.client import LLMClient, MockProvider
from feedback.ai.extraction import ExtractionSchemaError, validate_extraction
from feedback.models import IssueMention, MessageAnalysis
from feedback.services import evaluation, split_guard
from feedback.services.analysis import AnalysisService
from tests import factories


def asp(cat, sub, ev, basis="explicit", sent="negative", symptoms=(), summary="The customer reports it."):
    return {"category": cat, "subtype": sub, "sentiment": sent, "summary": summary, "evidence": ev,
            "evidence_basis": basis, "related_symptoms": list(symptoms)}


def out(aspects, mtype="complaint", review=False, reason=""):
    return {"message_type": mtype, "is_complaint": mtype == "complaint", "overall_sentiment": "negative",
            "aspects": aspects, "needs_review": review, "review_reason": reason}


# ------------------------------------------------------------------ prompt content
def test_prompt_states_the_new_rules_and_schema():
    p = extraction.SYSTEM_PROMPT
    for phrase in ("INDEPENDENT problem", "related_symptoms", "evidence_basis", "Never present a cause as confirmed",
                   "service_outage: only when the customer says", "ui_change: only when the customer says",
                   "library_data_loss: only when the customer says", "Sarcasm", "Do NOT output inferred aspects",
                   "\"other\" subtype and set needs_review true"):
        assert phrase in p, phrase
    item = extraction.OUTPUT_SCHEMA["properties"]["aspects"]["items"]
    assert {"evidence_basis", "related_symptoms"} <= set(item["required"])
    assert item["properties"]["evidence_basis"]["enum"] == ["explicit", "hedged", "inferred"]
    assert extraction.PROMPT_VERSION == "extract-v2"


# ------------------------------------------------------------------ symptoms vs independent issues
SITE = "is the site down? It won't let me change my payment card or invite my sister"


def test_one_incident_with_symptoms_is_one_aspect():
    r = validate_extraction(out([asp("playback_performance", "service_outage", "is the site down?", basis="hedged",
                                     symptoms=["cannot change payment card", "cannot invite family member"])]), SITE)
    assert len(r.aspects) == 1 and r.aspects[0].related_symptoms == ["cannot change payment card",
                                                                     "cannot invite family member"]
    assert "aspect_0_unconfirmed_cause_service_outage" in r.review_reasons  # a question is not a confirmed outage


def test_independent_issues_stay_separate():
    text = "I was charged twice this month. Separately, the app crashes when I open a playlist."
    r = validate_extraction(out([asp("billing_subscription", "unexpected_charge", "I was charged twice"),
                                 asp("application_errors", "app_crash", "the app crashes when I open a playlist")]), text)
    assert [a.subtype for a in r.aspects] == ["unexpected_charge", "app_crash"] and not r.needs_review


# ------------------------------------------------------------------ unconfirmed causes
@pytest.mark.parametrize("sub,ev,text", [
    ("service_outage", "I think the web player is down", "I think the web player is down, nothing plays"),
    ("ui_change", "did you remove the queue button?", "did you remove the queue button? can't see it"),
    ("library_data_loss", "maybe my playlists got deleted?", "maybe my playlists got deleted? they look empty"),
])
def test_hedged_cause_needs_human_review(sub, ev, text):
    cat = "application_errors" if sub != "service_outage" else "playback_performance"
    r = validate_extraction(out([asp(cat, sub, ev, basis="hedged")]), text)
    assert f"aspect_0_unconfirmed_cause_{sub}" in r.review_reasons and r.needs_review


def test_explicit_customer_statement_is_accepted():
    text = "The web player is down for me and plays nothing"
    r = validate_extraction(out([asp("playback_performance", "service_outage", "The web player is down for me")]), text)
    assert not r.needs_review and r.aspects[0].evidence_basis == "explicit"


def test_inferred_aspect_is_dropped_never_counted():
    text = "my songs keep stopping halfway"
    r = validate_extraction(out([asp("playback_performance", "playback_interruption", "my songs keep stopping"),
                                 asp("playback_performance", "service_outage", "keep stopping", basis="inferred")]), text)
    assert [a.subtype for a in r.aspects] == ["playback_interruption"]
    assert "aspect_1_inferred_not_stated_by_customer" in r.review_reasons


# ------------------------------------------------------------------ ambiguous / uncertain / sarcasm
def test_cannot_find_setting_is_a_question_without_ui_change():
    r = validate_extraction(out([], mtype="question"), "where do I find the equaliser on my phone?")
    assert r.message_type == "question" and r.aspects == [] and not r.needs_review


def test_uncertain_subtype_goes_to_review():
    text = "something is off with my account lately"
    r = validate_extraction(out([asp("account_access", "other", "something is off with my account")],
                                review=True, reason="subtype unclear"), text)
    assert "aspect_0_uncertain_subtype" in r.review_reasons and r.needs_review


def test_sarcastic_complaint_with_explicit_evidence_is_counted():
    text = "Love how my downloaded songs vanish every single week. Great job."
    r = validate_extraction(out([asp("playback_performance", "offline_downloads", "my downloaded songs vanish every single week")]), text)
    assert r.message_type == "complaint" and not r.needs_review


def test_v2_requires_evidence_basis_but_v1_payloads_still_validate():
    legacy = {"category": "billing_subscription", "subtype": "unexpected_charge", "sentiment": "negative",
              "summary": "s", "evidence": "charged twice"}
    with pytest.raises(ExtractionSchemaError):
        validate_extraction(out([legacy]), "charged twice")
    assert validate_extraction(out([legacy]), "charged twice", prompt_version="extract-v1").aspects


# ------------------------------------------------------------------ pipeline: review-flagged outputs are not counted
@pytest.mark.django_db
def test_hedged_outage_analysis_is_not_counted():
    m = factories.message(text="I think the web player is down, nothing plays at all")
    payload = out([asp("playback_performance", "service_outage", "I think the web player is down", basis="hedged")])
    AnalysisService(LLMClient(MockProvider(responder=lambda s, u: payload))).run(limit=1)
    a = m.analyses.get()
    assert a.status == "needs_review" and not IssueMention.objects.filter(message=m).exists()
    assert a.validated_output["aspects"][0]["evidence_basis"] == "hedged"


@pytest.mark.django_db
def test_v2_reanalysis_preserves_v1_result():
    """Existing extract-v1 analyses are kept; a v2 run adds a new row and supersedes atomically."""
    m = factories.message(text="I was charged twice for premium")
    v1 = MessageAnalysis.objects.create(message=m, provider="mock", model_name="mock-keyword-v2",
                                        prompt_version="extract-v1", is_mock=True, source="mock", status="succeeded",
                                        is_current=True, message_type="complaint", is_complaint=True)
    payload = out([asp("billing_subscription", "unexpected_charge", "charged twice")])
    AnalysisService(LLMClient(MockProvider(responder=lambda s, u: payload))).run(limit=5, message_ids=[m.pk])
    assert MessageAnalysis.objects.filter(pk=v1.pk, is_current=False, prompt_version="extract-v1").exists()
    assert m.analyses.get(is_current=True).prompt_version == "extract-v2"


# ------------------------------------------------------------------ split / leakage guards
def test_integrity_report_detects_overlap_and_prompt_leakage(monkeypatch):
    rows = {"dev_examples.csv": [{"tweet_id": "1", "text": "alpha bravo charlie delta echo foxtrot golf hotel india"}],
            "eval_sample.csv": [{"tweet_id": "1", "text": "zulu yankee"}, {"tweet_id": "2", "text": "unrelated words"}]}
    monkeypatch.setattr(split_guard, "_rows", lambda name: rows[name])
    rep = split_guard.integrity_report({"p": "intro alpha bravo charlie delta echo foxtrot golf hotel outro"})
    assert rep["overlap_ids"] == 1 and rep["prompt_leakage"]["p"]["dev_split_texts"] == 1 and rep["ok"] is False


def test_real_splits_are_disjoint_and_prompts_are_clean():
    if not split_guard.dev_ids() or not split_guard.eval_ids():
        pytest.skip("local dataset splits not present")
    rep = split_guard.integrity_report(split_guard.current_prompts())
    assert rep["ok"], rep


@pytest.mark.django_db
def test_dev_split_message_can_never_become_gold(analyst, monkeypatch):
    m = factories.message(text="some reviewed development message")
    monkeypatch.setattr(evaluation, "eval_sample_ids", lambda: [m.source_message_id])  # even if wrongly listed
    monkeypatch.setattr(split_guard, "dev_ids", lambda: {m.source_message_id})
    with pytest.raises(ValueError, match="Development-split"):
        evaluation.save_label(analyst, m, "question", [])
    kept, dropped = evaluation._drop_dev_split([evaluation.GoldItem(m.source_message_id, "question", set()),
                                               evaluation.GoldItem(999, "question", set())])
    assert dropped == 1 and [g.tweet_id for g in kept] == [999]
