"""Importing a snapshot of genuine real-LLM analyses: validated, attributed, idempotent, never billed."""
import json
from pathlib import Path

import pytest
from django.conf import settings
from django.core.management import CommandError, call_command

from feedback.models import AnalysisRun, CustomerMessage, IssueMention, LLMRequestLog, MessageAnalysis
from feedback.services import llm_budget, stats
from feedback.services.analysis import AnalysisService
from feedback.services.snapshots import SnapshotError, import_llm_snapshot
from tests import factories

pytestmark = pytest.mark.django_db
TEXT = "@user I was charged twice for premium this month, please fix it"


def snapshot(*items, provider="gemini"):
    return {"kind": "echovance.llm_analysis_snapshot", "provider": provider, "model_name": "gemini-x",
            "prompt_version": "extract-v1", "analyses": list(items)}


def item(msg, evidence="charged twice for premium", **kw):
    return {"tweet_id": msg.source_message_id, "message_type": "complaint", "is_complaint": True,
            "overall_sentiment": "negative", "status": "succeeded", "review_reasons": [],
            "aspects": [{"category": "billing_subscription", "subtype": "unexpected_charge", "sentiment": "negative",
                         "summary": "Charged twice.", "evidence": evidence}],
            "analysed_at": "2026-10-11T15:46:34+00:00", "attempts": 1, "latency_ms": 1422,
            "input_tokens": 1075, "output_tokens": 127, **kw}


def test_snapshot_becomes_current_real_analysis_without_using_budget():
    msg = factories.message(text=TEXT)
    budget_before = llm_budget.status()
    r = import_llm_snapshot(snapshot(item(msg)))
    assert (r.imported, r.rejected, r.missing_messages) == (1, [], [])
    a = MessageAnalysis.objects.get(message=msg)
    assert a.source == "llm" and not a.is_mock and a.is_current and a.status == "succeeded"
    assert (a.provider, a.model_name, a.prompt_version) == ("gemini", "gemini-x", "extract-v1")
    assert a.created_at.isoformat().startswith("2026-10-11T15:46:34") and a.input_tokens == 1075
    m = IssueMention.objects.get(analysis=a)
    assert m.source == "llm" and msg.text[m.evidence_start:m.evidence_end] == "charged twice for premium"
    assert "no API request" in a.run.stopped_reason and not a.run.is_mock
    assert LLMRequestLog.objects.count() == 0 and llm_budget.status()["used"] == budget_before["used"] == 0
    assert CustomerMessage.objects.get(pk=msg.pk).analysis_status == "completed"
    assert stats.provenance()["real_analysed"] == 1
    again = import_llm_snapshot(snapshot(item(msg)))
    assert (again.imported, again.already_present, again.run) == (0, 1, None)
    assert AnalysisRun.objects.count() == 1


def test_snapshot_supersedes_mock_analysis_and_keeps_history():
    msg = factories.message(text=TEXT)
    AnalysisService().run(limit=10)  # conftest: mock provider in demo mode
    mock = MessageAnalysis.objects.get(message=msg)
    assert mock.source == "mock" and mock.is_current
    import_llm_snapshot(snapshot(item(msg)))
    mock.refresh_from_db()
    assert not mock.is_current and not mock.mentions.filter(is_active=True).exists()
    assert MessageAnalysis.objects.get(message=msg, is_current=True).source == "llm"


def test_results_that_do_not_validate_are_rejected_not_adjusted():
    msg = factories.message(text=TEXT)
    other = factories.message(text="how do I change my profile picture?")
    r = import_llm_snapshot(snapshot(item(msg, evidence="charged three times"),
                                     item(other, message_type="question", is_complaint=False, aspects=[]),
                                     {**item(msg), "tweet_id": 1}))
    assert [t for t, _ in r.rejected] == [msg.source_message_id] and r.missing_messages == [1]
    assert r.imported == 1 and not MessageAnalysis.objects.filter(message=msg).exists()


def test_status_must_match_what_the_validator_derives():
    msg = factories.message(text=TEXT)
    r = import_llm_snapshot(snapshot(item(msg, status="needs_review")))
    assert r.imported == 0 and "differs from recorded" in r.rejected[0][1]


@pytest.mark.parametrize("bad", [{"kind": "other"}, snapshot(provider="mock"), snapshot(provider="human")])
def test_only_real_provider_snapshots_are_accepted(bad):
    with pytest.raises(SnapshotError):
        import_llm_snapshot(bad)


def test_command_reports_partial_import(tmp_path):
    msg = factories.message(text=TEXT)
    path = tmp_path / "s.json"
    path.write_text(json.dumps(snapshot(item(msg), {**item(msg), "tweet_id": 1})), encoding="utf-8")
    with pytest.raises(CommandError, match="partially"):
        call_command("import_llm_snapshot", str(path))
    assert MessageAnalysis.objects.filter(source="llm").count() == 1


PUBLIC = Path(settings.BASE_DIR) / "data" / "spotifycares_public"


@pytest.mark.skipif(not (PUBLIC / "gemini_snapshot.json").exists(), reason="public dataset not present")
def test_shipped_public_dataset_and_gemini_snapshot_are_consistent():
    """The published (redacted) subset imports cleanly and all genuine Gemini results validate against it."""
    call_command("import_messages", str(PUBLIC / "spotifycares_subset.csv"), "--data-classification", "public")
    assert CustomerMessage.objects.filter(author_role="customer", is_thread_root=True).count() == 4000
    assert CustomerMessage.objects.filter(author_role="agent").count() == 4000
    data = json.loads((PUBLIC / "gemini_snapshot.json").read_text(encoding="utf-8"))
    r = import_llm_snapshot(data)
    assert (r.imported, r.rejected, r.missing_messages) == (len(data["analyses"]), [], [])
    assert stats.provenance()["real_analysed"] == len(data["analyses"]) == 13
    assert LLMRequestLog.objects.count() == 0
