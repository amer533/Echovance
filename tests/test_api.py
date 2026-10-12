from datetime import datetime, timedelta, timezone

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from feedback.models import ActionItem, CustomerMessage, IssueAlert
from feedback.services.grouping import get_or_create_issue
from tests import factories
from tests.factories import csv_bytes, row

pytestmark = pytest.mark.django_db
KEY = "billing_subscription.unexpected_charge"


def upload(api, data, name="d.csv"):
    return api.post("/api/imports/", {"file": SimpleUploadedFile(name, data, "text/csv"),
                                      "brand_account": "SpotifyCares"}, format="multipart")


@pytest.mark.parametrize("method,url", [
    ("get", "/api/messages/"), ("get", "/api/issues/"), ("get", "/api/alerts/"), ("get", "/api/actions/"),
    ("get", "/api/imports/"), ("get", "/api/dashboard/summary/"), ("post", "/api/analyses/run/"),
    ("post", "/api/actions/"), ("post", "/api/alerts/detect/"), ("post", "/api/imports/"),
])
def test_anonymous_requests_are_rejected(api, method, url):
    assert getattr(api, method)(url).status_code in (401, 403)


def test_import_endpoint_contract(api, analyst):
    api.force_authenticate(analyst)
    r = upload(api, csv_bytes([row(1), row(2, author="SpotifyCares", inbound="False", in_reply="1")]))
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "completed" and body["accepted_count"] == 2 and body["agent_count"] == 1
    assert api.get(f"/api/imports/{body['id']}/").json()["id"] == body["id"]
    again = upload(api, csv_bytes([row(1)]))
    assert again.status_code == 201 and again.json()["duplicate_count"] == 1
    bad = upload(api, b"foo,bar\n1,2\n")
    assert bad.status_code == 400 and bad.json()["status"] == "failed"


def test_viewer_cannot_import_or_run_pipeline(api, viewer):
    api.force_authenticate(viewer)
    assert upload(api, csv_bytes([row(1)])).status_code == 403
    assert api.post("/api/analyses/run/", {"limit": 1}, format="json").status_code == 403
    assert api.post("/api/alerts/detect/", {}, format="json").status_code == 403
    assert api.get("/api/messages/").status_code == 200  # read access is fine


def test_messages_list_filters_and_pagination(api, viewer):
    api.force_authenticate(viewer)
    b = factories.batch()
    for i in range(30):
        factories.message(b, text=f"msg {i}")
    factories.message(b, role="agent", root=False, text="agent reply")
    r = api.get("/api/messages/", {"author_role": "customer"}).json()
    assert r["count"] == 30 and len(r["results"]) == 25 and r["next"]
    assert api.get("/api/messages/", {"author_role": "agent"}).json()["count"] == 1
    assert api.get("/api/messages/", {"q": "msg 1"}).json()["count"] == 11


def test_run_analysis_endpoint_uses_mock_and_caps_batch(api, analyst):
    api.force_authenticate(analyst)
    factories.message(text="I was charged twice for premium, please refund")
    r = api.post("/api/analyses/run/", {"limit": 5}, format="json")
    assert r.status_code == 201 and r.json()["is_mock"] is True and r.json()["succeeded"] == 1
    assert api.post("/api/analyses/run/", {"limit": 500}, format="json").status_code == 400
    msg = CustomerMessage.objects.get()
    detail = api.get(f"/api/messages/{msg.pk}/").json()
    assert detail["analyses"][0]["is_mock"] is True and detail["analyses"][0]["mentions"]


def test_run_analysis_reports_missing_credentials_and_disabled_provider(api, analyst, settings):
    api.force_authenticate(analyst)
    settings.LLM = {**settings.LLM, "PROVIDER": "gemini", "GEMINI_API_KEY": ""}
    r = api.post("/api/analyses/run/", {"limit": 1}, format="json")
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
    settings.LLM = {**settings.LLM, "PROVIDER": "openai", "OPENAI_API_KEY": "sk-x", "OPENAI_ENABLED": False}
    r = api.post("/api/analyses/run/", {"limit": 1}, format="json")
    assert r.status_code == 503 and "disabled" in r.json()["detail"]


def _seed_alertable():
    b = factories.batch()
    start = datetime(2017, 10, 4, tzinfo=timezone.utc)
    for counts in [{KEY: 3}, {KEY: 4}, {KEY: 3}, {KEY: 30}]:
        factories.fill_window(b, start, 7, 120, counts)
        start += timedelta(days=7)


def test_issue_and_alert_endpoints(api, manager):
    api.force_authenticate(manager)
    _seed_alertable()
    issues = api.get("/api/issues/").json()
    assert issues["count"] == 1 and issues["results"][0]["mention_count"] == 40
    iid = issues["results"][0]["id"]
    detail = api.get(f"/api/issues/{iid}/").json()
    assert sum(w["count"] for w in detail["weekly_trend"]) == 40 and len(detail["recent_mentions"]) == 20
    det = api.post("/api/alerts/detect/", {}, format="json").json()
    assert det["windows"][0]["alerts_created"] and det["windows"][0]["sufficient_data"]
    det2 = api.post("/api/alerts/detect/", {}, format="json").json()
    assert det2["windows"][0]["alerts_created"] == [] and det2["windows"][0]["alerts_existing"]
    alert_id = det["windows"][0]["alerts_created"][0]
    a = api.get(f"/api/alerts/{alert_id}/").json()
    assert a["rule"] == "surge" and len(a["evidence"]) == 10 and a["status"] == "open"
    assert api.post(f"/api/alerts/{alert_id}/review/", {"status": "dismissed"}, format="json").status_code == 400
    r = api.post(f"/api/alerts/{alert_id}/review/", {"status": "acknowledged", "note": "ok"}, format="json")
    assert r.status_code == 200 and r.json()["reviewed_by"]["username"] == "mgr"
    assert api.patch(f"/api/issues/{iid}/", {"status": "monitoring"}, format="json").json()["status"] == "monitoring"


def test_viewer_cannot_review_alerts_or_change_issue(api, viewer):
    _seed_alertable()
    from feedback.services.trends import TrendService
    alert = TrendService().detect(datetime(2017, 11, 1, tzinfo=timezone.utc)).created[0]
    api.force_authenticate(viewer)
    assert api.post(f"/api/alerts/{alert.pk}/review/", {"status": "acknowledged"}, format="json").status_code == 403
    assert api.patch(f"/api/issues/{alert.issue_id}/", {"status": "resolved"}, format="json").status_code == 403


def test_action_endpoints_enforce_workflow(api, manager, viewer):
    issue = get_or_create_issue("billing_subscription", "unexpected_charge")
    factories.analysed_message(factories.batch(), datetime(2017, 10, 9, tzinfo=timezone.utc), (KEY,), "charged twice!")
    api.force_authenticate(manager)
    r = api.post("/api/actions/", {"issue": issue.pk, "title": "Audit", "description": "Check"}, format="json")
    assert r.status_code == 201 and r.json()["status"] == "proposed"
    aid = r.json()["id"]
    assert api.patch(f"/api/actions/{aid}/", {"status": "completed"}, format="json").status_code == 409
    assert api.patch(f"/api/actions/{aid}/", {"status": "approved", "assignee": viewer.pk},
                     format="json").status_code == 400  # one change per request
    assert api.patch(f"/api/actions/{aid}/", {"status": "approved", "note": "go"}, format="json").status_code == 200
    assert api.patch(f"/api/actions/{aid}/", {"assignee": viewer.pk, "due_date": "2030-01-31"},
                     format="json").json()["assignee"]["username"] == "view"
    api.force_authenticate(viewer)
    assert api.patch(f"/api/actions/{aid}/", {"status": "in_progress"}, format="json").status_code == 200
    done = api.patch(f"/api/actions/{aid}/", {"status": "completed", "note": "done"}, format="json").json()
    assert done["status"] == "completed" and [e["to_status"] for e in done["events"]][-1] == "completed"
    assert api.post("/api/actions/", {"issue": issue.pk, "title": "x", "description": "y"},
                    format="json").status_code == 403
    assert api.put(f"/api/actions/{aid}/", {}, format="json").status_code == 405


def test_non_manager_action_creation_is_forbidden_before_validation(api, analyst):
    """Permission is checked before the body, so a 403 reveals nothing about which issues exist."""
    api.force_authenticate(analyst)
    assert api.post("/api/actions/", {"issue": 999999}, format="json").status_code == 403
    assert api.post("/api/actions/suggest/", {"issue": 999999}, format="json").status_code == 403


def test_ai_suggest_endpoint(api, manager):
    issue = get_or_create_issue("billing_subscription", "unexpected_charge")
    factories.analysed_message(factories.batch(), datetime(2017, 10, 9, tzinfo=timezone.utc), (KEY,), "charged twice!")
    api.force_authenticate(manager)
    r = api.post("/api/actions/suggest/", {"issue": issue.pk}, format="json")
    assert r.status_code == 201 and r.json()["status"] == "proposed" and r.json()["ai_is_mock"] is True
    assert ActionItem.objects.get().origin == "ai"


def test_dashboard_summary_is_computed_from_data(api, viewer):
    api.force_authenticate(viewer)
    empty = api.get("/api/dashboard/summary/").json()
    assert empty["customer_roots"] == 0 and empty["weekly_trend"] == []
    _seed_alertable()
    body = api.get("/api/dashboard/summary/").json()
    assert body["customer_roots"] == 480 and body["current_analyses_real"] == 480
    assert body["current_analyses_mock"] == 0 and body["provenance"]["real_analysed"] == 480
    assert body["categories"][0] == {"code": "billing_subscription", "label": "Billing / Subscription", "count": 40}
    assert IssueAlert.objects.count() == 0  # reading never creates alerts
