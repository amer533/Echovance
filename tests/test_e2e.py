"""End-to-end main workflow through the public API with a synthetic CSV and the mock provider."""
from datetime import datetime, timedelta, timezone

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from feedback.models import ActionItem, CustomerMessage, IssueMention
from tests.factories import csv_bytes, row

pytestmark = [pytest.mark.django_db, pytest.mark.e2e]

COMPLAINTS = {
    "billing": "@SpotifyCares I was charged twice for premium this month",
    "offline": "@SpotifyCares my offline downloads disappeared again, why?",
    "question": "@SpotifyCares how do I change my profile picture?",
}


def synthetic_csv() -> bytes:
    """4 weeks of baseline + 1 surge week for billing complaints, plus agent replies (SYNTHETIC)."""
    rows, tid = [], 1
    start = datetime(2017, 10, 4, tzinfo=timezone.utc)
    for week in range(4):
        n_billing = 12 if week == 3 else 2
        plan = ["billing"] * n_billing + ["offline"] * 3 + ["question"] * (25 - n_billing - 3)
        for i, kind in enumerate(plan):
            created = start + timedelta(days=7 * week, hours=6 * i)
            rows.append(row(tid, author=str(5000 + tid), created=created, text=COMPLAINTS[kind], response=str(tid + 1)))
            rows.append(row(tid + 1, author="SpotifyCares", inbound="False", created=created + timedelta(minutes=5),
                            text="@x Sorry to hear that, please DM us", in_reply=str(tid)))
            tid += 2
    return csv_bytes(rows)


def test_main_workflow(api, settings):
    from tests import factories
    settings.TREND_RULES = {**settings.TREND_RULES, "MIN_WINDOW_ANALYZED_MESSAGES": 20}
    analyst, manager, agent = factories.user("a", "Analysts"), factories.user("m", "Managers"), factories.user("ag")

    # 1. import (twice: second is fully duplicate)
    api.force_authenticate(analyst)
    data = synthetic_csv()
    first = api.post("/api/imports/", {"file": SimpleUploadedFile("s.csv", data), "brand_account": "SpotifyCares"},
                     format="multipart").json()
    assert first["accepted_count"] == 200 and first["customer_root_count"] == 100 and first["agent_count"] == 100
    second = api.post("/api/imports/", {"file": SimpleUploadedFile("s.csv", data), "brand_account": "SpotifyCares"},
                      format="multipart").json()
    assert second["accepted_count"] == 0 and second["duplicate_count"] == 200

    # 2. analyse in bounded batches until nothing is pending
    while CustomerMessage.objects.filter(analysis_status="pending").exists():
        assert api.post("/api/analyses/run/", {"limit": 25}, format="json").status_code == 201
    assert CustomerMessage.objects.filter(author_role="agent", analyses__isnull=False).count() == 0
    # mock-mode run: its data lives in the DEMO scope and is invisible in the default real scope
    assert api.get("/api/issues/").json()["count"] == 0
    issues = {i["key"]: i for i in api.get("/api/issues/", {"scope": "demo"}).json()["results"]}
    assert issues["billing_subscription.unexpected_charge"]["mention_count"] == 18
    assert "playback_performance.offline_downloads" in issues

    # 3. every mention is traceable to a stored message and timestamp
    for m in IssueMention.objects.select_related("message"):
        assert m.message.text[m.evidence_start:m.evidence_end] == m.evidence_text
        assert m.message_created_at == m.message.created_at

    # 4. detect trends: billing surges in the last week, offline is flat
    det = api.post("/api/alerts/detect/", {"scope": "demo"}, format="json").json()["windows"][0]
    assert det["sufficient_data"] and len(det["alerts_created"]) == 1
    alert = api.get(f"/api/alerts/{det['alerts_created'][0]}/").json()
    assert alert["issue_key"] == "billing_subscription.unexpected_charge" and alert["rule"] == "surge"
    assert alert["data_scope"] == "demo" and alert["is_stale"] is False
    assert alert["current_count"] == 12 and alert["baseline_mean"] == 2

    # 5. manager reviews alert, gets AI proposal, approves, assigns; agent completes
    api.force_authenticate(manager)
    api.post(f"/api/alerts/{alert['id']}/review/", {"status": "acknowledged", "note": "investigating"}, format="json")
    act = api.post("/api/actions/suggest/", {"issue": alert["issue"], "alert": alert["id"]}, format="json").json()
    assert act["status"] == "proposed" and act["origin"] == "ai"
    api.patch(f"/api/actions/{act['id']}/", {"status": "approved", "note": "agreed"}, format="json")
    api.patch(f"/api/actions/{act['id']}/", {"assignee": agent.pk}, format="json")
    api.force_authenticate(agent)
    api.patch(f"/api/actions/{act['id']}/", {"status": "in_progress"}, format="json")
    done = api.patch(f"/api/actions/{act['id']}/", {"status": "completed", "note": "help article updated"},
                     format="json").json()
    assert done["status"] == "completed"
    assert [e["event"] for e in done["events"]] == ["ai_suggested", "status_changed", "assigned", "status_changed",
                                                    "status_changed"]
    # completing the action does not resolve the issue
    assert api.get(f"/api/issues/{alert['issue']}/").json()["status"] == "open"
    assert ActionItem.objects.get().decided_by.username == "m"
