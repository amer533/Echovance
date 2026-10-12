from datetime import datetime, timedelta, timezone

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from feedback.models import ActionItem, CustomerMessage, IssueAlert
from feedback.services.trends import TrendService
from tests import factories
from tests.factories import csv_bytes, row

pytestmark = pytest.mark.django_db
KEY = "billing_subscription.unexpected_charge"


@pytest.fixture
def seeded():
    b = factories.batch()
    start = datetime(2017, 10, 4, tzinfo=timezone.utc)
    for counts in [{KEY: 3}, {KEY: 4}, {KEY: 3}, {KEY: 30}]:
        factories.fill_window(b, start, 7, 120, counts)
        start += timedelta(days=7)
    alert = TrendService().detect(datetime(2017, 11, 1, tzinfo=timezone.utc)).created[0]
    return alert


def test_pages_require_login(client):
    r = client.get(reverse("dashboard"))
    assert r.status_code == 302 and reverse("login") in r["Location"]


def test_login_flow(client, manager):
    r = client.post(reverse("login"), {"username": "mgr", "password": "pw-Test-12345"})
    assert r.status_code == 302
    assert client.get(reverse("dashboard")).status_code == 200


def test_empty_state_pages_render(client, viewer):
    client.force_login(viewer)
    for name in ["dashboard", "imports", "messages", "issues", "alerts", "actions"]:
        r = client.get(reverse(name))
        assert r.status_code == 200, name
    assert b"No data imported yet" in client.get(reverse("dashboard")).content
    assert b"Demo mode (mock AI)" in client.get(reverse("dashboard")).content


def test_all_pages_render_with_data(client, manager, seeded):
    client.force_login(manager)
    msg = CustomerMessage.objects.first()
    action = ActionItem.objects.create(issue=seeded.issue, alert=seeded, title="t", description="d", origin="manual",
                                       created_by=manager)
    urls = [reverse("dashboard"), reverse("messages"), reverse("messages") + f"?issue={seeded.issue_id}&status=completed",
            reverse("message-detail", args=[msg.pk]), reverse("issues"), reverse("issue-detail", args=[seeded.issue_id]),
            reverse("alerts") + "?status=all", reverse("alert-detail", args=[seeded.pk]), reverse("actions"),
            reverse("action-detail", args=[action.pk]), reverse("import-detail", args=[msg.import_batch_id])]
    for u in urls:
        r = client.get(u)
        assert r.status_code == 200, u
    dash = client.get(reverse("dashboard")).content.decode()
    assert "Billing / Subscription" in dash and "480" in dash


def test_import_form_round_trip(client, analyst):
    client.force_login(analyst)
    f = SimpleUploadedFile("d.csv", csv_bytes([row(1), row(2)]), "text/csv")
    r = client.post(reverse("imports"), {"file": f, "brand_account": "SpotifyCares"}, follow=True)
    assert r.status_code == 200 and b"Imported 2 new rows" in r.content
    assert CustomerMessage.objects.count() == 2


def test_viewer_cannot_post_pipeline_forms(client, viewer):
    client.force_login(viewer)
    f = SimpleUploadedFile("d.csv", csv_bytes([row(1)]), "text/csv")
    assert client.post(reverse("imports"), {"file": f, "brand_account": "SpotifyCares"}).status_code == 403
    assert client.post(reverse("run-analysis"), {"limit": 1}).status_code == 403
    assert client.post(reverse("detect-alerts"), {}).status_code == 403


def test_analysis_and_detection_buttons_work(client, analyst):
    client.force_login(analyst)
    factories.message(text="I was charged twice for premium")
    r = client.post(reverse("run-analysis"), {"limit": 5}, follow=True)
    assert b"MOCK provider (not real AI)" in r.content
    assert CustomerMessage.objects.get().analysis_status == "completed"
    r = client.post(reverse("detect-alerts"), {"all_windows": "on"}, follow=True)
    assert r.status_code == 200 and b"Evaluated" in r.content


def test_action_buttons_follow_state_machine(client, manager, viewer, seeded):
    client.force_login(manager)
    r = client.post(reverse("action-suggest", args=[seeded.issue_id]), {"alert": seeded.pk}, follow=True)
    act = ActionItem.objects.get()
    assert act.status == "proposed" and act.alert == seeded and b"Approve" in r.content
    post = lambda data: client.post(reverse("action-update", args=[act.pk]), data, follow=True)  # noqa: E731
    post({"kind": "transition", "status": "approved", "note": ""})
    post({"kind": "assign", "assignee": viewer.pk, "due_date": ""})
    client.force_login(viewer)
    post({"kind": "transition", "status": "in_progress"})
    r = post({"kind": "transition", "status": "completed", "note": "done"})
    act.refresh_from_db()
    assert act.status == "completed" and b"This action is closed" in r.content
    r = post({"kind": "transition", "status": "approved"})
    assert b"Cannot move" in r.content


def test_alert_review_form(client, manager, seeded):
    client.force_login(manager)
    client.post(reverse("alert-review", args=[seeded.pk]), {"status": "acknowledged", "note": ""})
    assert IssueAlert.objects.get().status == "acknowledged"


def test_evidence_is_highlighted_and_escaped(client, viewer):
    client.force_login(viewer)
    m = factories.analysed_message(factories.batch(), datetime(2017, 10, 9, tzinfo=timezone.utc), (KEY,),
                                   text="<b>charged</b> twice")
    html = client.get(reverse("message-detail", args=[m.pk])).content.decode()
    assert '<mark class="evidence">&lt;b&gt;charged</mark>' in html and "<b>charged</b>" not in html


# ---------------- regression: action decisions through the web form ----------------
# Bug: the Approve button's status value was dropped by app.js before submission, so the view
# answered "status: This field is required". The browser path is covered in
# test_browser_actions.py; these tests pin the server-side contract the browser relies on.
@pytest.fixture
def proposed_action(manager):
    from feedback.services import actions as action_service
    from feedback.services.grouping import get_or_create_issue
    issue = get_or_create_issue("billing_subscription", "unexpected_charge")
    return action_service.create_manual(user=manager, issue=issue, title="Audit charges", description="Check logs")


def _update(client, action, data):
    return client.post(reverse("action-update", args=[action.pk]), data, follow=True)


def test_detail_renders_one_named_submit_button_per_allowed_transition(client, manager, proposed_action):
    client.force_login(manager)
    html = client.get(reverse("action-detail", args=[proposed_action.pk])).content.decode()
    form = html.split('name="kind" value="transition"', 1)[1].split("</form>", 1)[0]
    assert 'type="submit" name="status" value="approved"' in form
    assert 'type="submit" name="status" value="rejected"' in form
    assert 'value="completed"' not in form and 'value="in_progress"' not in form


def test_approve_via_form(client, manager, proposed_action):
    client.force_login(manager)
    r = _update(client, proposed_action, {"kind": "transition", "status": "approved", "note": "ok"})
    proposed_action.refresh_from_db()
    assert b"Status updated." in r.content and proposed_action.status == "approved"
    assert proposed_action.decided_by == manager and proposed_action.decision_note == "ok"
    assert proposed_action.events.last().to_status == "approved"


def test_missing_status_is_rejected_without_side_effects(client, manager, proposed_action):
    client.force_login(manager)
    r = _update(client, proposed_action, {"kind": "transition", "note": "x"})
    proposed_action.refresh_from_db()
    assert b"status: This field is required." in r.content
    assert proposed_action.status == "proposed" and proposed_action.events.count() == 1


def test_reject_via_form_requires_reason(client, manager, proposed_action):
    client.force_login(manager)
    r = _update(client, proposed_action, {"kind": "transition", "status": "rejected", "note": ""})
    proposed_action.refresh_from_db()
    assert b"A reason is required to reject an action." in r.content and proposed_action.status == "proposed"
    _update(client, proposed_action, {"kind": "transition", "status": "rejected", "note": "duplicate"})
    proposed_action.refresh_from_db()
    assert proposed_action.status == "rejected" and proposed_action.decision_note == "duplicate"


def test_edit_via_form_only_while_proposed(client, manager, proposed_action):
    client.force_login(manager)
    _update(client, proposed_action, {"kind": "edit", "title": "Audit duplicate charges", "description": "Check logs"})
    proposed_action.refresh_from_db()
    assert proposed_action.title == "Audit duplicate charges" and proposed_action.events.last().event == "edited"
    _update(client, proposed_action, {"kind": "transition", "status": "approved"})
    r = _update(client, proposed_action, {"kind": "edit", "title": "Changed after approval", "description": "x"})
    proposed_action.refresh_from_db()
    assert b"Only proposed actions can be edited." in r.content and proposed_action.title == "Audit duplicate charges"


@pytest.mark.parametrize("target", ["completed", "in_progress", "proposed", "bogus"])
def test_invalid_transitions_via_form_change_nothing(client, manager, proposed_action, target):
    client.force_login(manager)
    r = _update(client, proposed_action, {"kind": "transition", "status": target})
    proposed_action.refresh_from_db()
    assert proposed_action.status == "proposed" and proposed_action.events.count() == 1
    assert (b"Cannot move" in r.content) or (b"Select a valid choice" in r.content) or (b"Assign" in r.content)


def test_viewer_cannot_approve_via_form(client, viewer, proposed_action):
    client.force_login(viewer)
    r = client.post(reverse("action-update", args=[proposed_action.pk]), {"kind": "transition", "status": "approved"})
    proposed_action.refresh_from_db()
    assert r.status_code == 403 and proposed_action.status == "proposed"


def test_app_js_preserves_clicked_button_value():
    """Static guard: the submit handler must not drop the clicked button's name/value."""
    from pathlib import Path
    js = (Path(__file__).resolve().parent.parent / "feedback/static/feedback/app.js").read_text(encoding="utf-8")
    assert "event.submitter" in js and "mirror.name = submitter.name" in js and "mirror.value = submitter.value" in js
