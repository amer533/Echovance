from datetime import date, datetime, timezone

import pytest
from django.core.exceptions import PermissionDenied

from feedback.ai.client import LLMClient, LLMError, MockProvider
from feedback.models import ActionItem, Issue
from feedback.services import actions as svc
from feedback.services import review
from feedback.services.grouping import get_or_create_issue
from tests import factories

pytestmark = pytest.mark.django_db
S = ActionItem.Status


@pytest.fixture
def issue(db):
    i = get_or_create_issue("billing_subscription", "unexpected_charge")
    factories.analysed_message(factories.batch(), datetime(2017, 10, 10, tzinfo=timezone.utc), (i.key,),
                               text="I was charged twice")
    return i


@pytest.fixture
def proposed(issue, manager):
    return svc.create_manual(user=manager, issue=issue, title="Audit duplicate charges", description="Check logs")


def test_manual_action_starts_proposed_with_audit(proposed, manager):
    assert proposed.status == S.PROPOSED and proposed.origin == "manual"
    e = proposed.events.get()
    assert (e.event, e.to_status, e.actor) == ("created", S.PROPOSED, manager)


def test_full_happy_path(proposed, manager, viewer):
    a = svc.transition(proposed, S.APPROVED, manager, "looks right")
    assert a.decided_by == manager and a.decided_at and a.decision_note == "looks right"
    a = svc.assign(a, manager, viewer, date(2030, 1, 1))
    a = svc.transition(a, S.IN_PROGRESS, viewer)            # assignee may start
    a = svc.transition(a, S.COMPLETED, viewer, "fixed copy")  # and complete
    assert a.status == S.COMPLETED and a.completed_at and a.completion_note == "fixed copy"
    assert [e.event for e in a.events.all()] == ["created", "status_changed", "assigned", "status_changed",
                                                 "status_changed"]
    assert Issue.objects.get(pk=a.issue_id).status == Issue.Status.OPEN  # completion != resolution


@pytest.mark.parametrize("path,target", [
    ([], S.IN_PROGRESS), ([], S.COMPLETED), ([S.APPROVED], S.COMPLETED), ([S.APPROVED], S.PROPOSED),
])
def test_invalid_transitions_rejected(proposed, manager, path, target):
    for step in path:
        proposed = svc.transition(proposed, step, manager)
    with pytest.raises(svc.TransitionError):
        svc.transition(proposed, target, manager)


def test_terminal_states(proposed, manager):
    a = svc.transition(proposed, S.REJECTED, manager, "not needed")
    for target in S.values:
        with pytest.raises(svc.TransitionError):
            svc.transition(a, target, manager, "x")


def test_reject_requires_reason(proposed, manager):
    with pytest.raises(svc.ActionError):
        svc.transition(proposed, S.REJECTED, manager, "  ")


def test_start_requires_assignee(proposed, manager):
    a = svc.transition(proposed, S.APPROVED, manager)
    with pytest.raises(svc.ActionError, match="Assign"):
        svc.transition(a, S.IN_PROGRESS, manager)


def test_non_managers_cannot_approve_reject_assign_or_create(proposed, analyst, viewer, issue):
    for user in (analyst, viewer):
        with pytest.raises(PermissionDenied):
            svc.transition(proposed, S.APPROVED, user)
        with pytest.raises(PermissionDenied):
            svc.create_manual(user=user, issue=issue, title="t", description="d")
        with pytest.raises(PermissionDenied):
            svc.suggest_with_ai(user=user, issue=issue)


def test_only_assignee_or_manager_can_progress(proposed, manager, viewer, analyst):
    a = svc.assign(svc.transition(proposed, S.APPROVED, manager), manager, viewer)
    with pytest.raises(PermissionDenied):
        svc.assign(a, analyst, analyst)
    with pytest.raises(PermissionDenied):
        svc.transition(a, S.IN_PROGRESS, analyst)
    assert svc.transition(a, S.IN_PROGRESS, manager).status == S.IN_PROGRESS


def test_cannot_assign_unapproved_action(proposed, manager, viewer):
    with pytest.raises(svc.TransitionError):
        svc.assign(proposed, manager, viewer)


def test_edit_only_while_proposed(proposed, manager):
    a = svc.edit_proposal(proposed, manager, title="New title")
    assert a.title == "New title" and a.events.last().event == "edited"
    a = svc.transition(a, S.APPROVED, manager)
    with pytest.raises(svc.TransitionError):
        svc.edit_proposal(a, manager, title="Sneaky change")


def test_allowed_transitions_reflect_role(proposed, manager, viewer):
    assert svc.allowed_transitions(proposed, manager) == [S.APPROVED, S.REJECTED]
    assert svc.allowed_transitions(proposed, viewer) == []


def _client(responder):
    return LLMClient(MockProvider(responder=responder), max_retries=0, sleep=lambda s: None)


def test_ai_suggestion_is_always_proposed(issue, manager):
    a = svc.suggest_with_ai(user=manager, issue=issue, client=_client(lambda s, u: {
        "title": "Review billing retries", "steps": ["Reproduce the double charge."], "evidence_refs": [1],
        "rationale": "r", "limitations": "small sample", "uncertainty": "medium", "needs_more_information": False}))
    assert a.status == S.PROPOSED and a.origin == "ai" and a.ai_is_mock and a.decided_by is None
    assert a.events.get().event == "ai_suggested"


def test_ai_suggestion_prompt_contains_evidence_as_data(issue, manager):
    seen = {}

    def responder(schema, prompt):
        seen["p"] = prompt
        return {"title": "t", "steps": ["d"], "evidence_refs": [1], "rationale": "r", "limitations": "l",
                "uncertainty": "high", "needs_more_information": True}

    a = svc.suggest_with_ai(user=manager, issue=issue, client=_client(responder))
    assert "<customer_evidence>" in seen["p"] and "I was char" in seen["p"]
    assert a.rationale.startswith("[AI flagged")


@pytest.mark.parametrize("out", [LLMError("timeout", retryable=True),
                                 {"title": "", "steps": [], "evidence_refs": [], "rationale": "", "limitations": "",
                                  "uncertainty": "low", "needs_more_information": False},
                                 {"title": "x"},
                                 {"title": "x", "steps": ["s"], "evidence_refs": [99], "rationale": "r",
                                  "limitations": "l", "uncertainty": "low", "needs_more_information": False}])
def test_ai_failure_creates_nothing(issue, manager, out):
    with pytest.raises(svc.SuggestionError):
        svc.suggest_with_ai(user=manager, issue=issue, client=_client(lambda s, u: out))
    assert ActionItem.objects.count() == 0


def test_ai_needs_evidence(manager):
    empty = get_or_create_issue("content_catalog", "missing_content")
    with pytest.raises(svc.SuggestionError):
        svc.suggest_with_ai(user=manager, issue=empty, client=_client(lambda s, u: {}))


def test_alert_review_and_issue_status_are_manager_only(manager, viewer, issue):
    from feedback.models import IssueAlert
    t = datetime(2017, 10, 1, tzinfo=timezone.utc)
    alert = IssueAlert.objects.create(issue=issue, rule="surge", rule_version="v", window_start=t,
                                      window_end=t.replace(day=8), baseline_start=t, baseline_end=t,
                                      current_count=1, baseline_total=0, baseline_mean=0, current_share=0,
                                      baseline_share=0, current_window_volume=1, thresholds={}, explanation="e")
    with pytest.raises(PermissionDenied):
        review.review_alert(alert, viewer, "acknowledged")
    with pytest.raises(svc.ActionError):
        review.review_alert(alert, manager, "dismissed", "")
    a = review.review_alert(alert, manager, "acknowledged", "on it")
    assert a.reviewed_by == manager and a.reviewed_at
    with pytest.raises(PermissionDenied):
        review.set_issue_status(issue, viewer, "resolved")
    assert review.set_issue_status(issue, manager, "resolved").status_changed_by == manager
