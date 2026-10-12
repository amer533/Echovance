"""Manager review of alerts and manual issue status decisions."""
from __future__ import annotations

from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.utils import timezone

from feedback.models import Issue, IssueAlert

from .actions import ActionError, is_manager


@transaction.atomic
def review_alert(alert: IssueAlert, user, status: str, note: str = "") -> IssueAlert:
    if not is_manager(user):
        raise PermissionDenied("Only managers can review alerts.")
    if status not in (IssueAlert.Status.ACKNOWLEDGED, IssueAlert.Status.DISMISSED):
        raise ActionError("Alert review status must be 'acknowledged' or 'dismissed'.")
    if status == IssueAlert.Status.DISMISSED and not note.strip():
        raise ActionError("A note is required to dismiss an alert.")
    alert = IssueAlert.objects.select_for_update().get(pk=alert.pk)
    alert.status = status
    alert.reviewed_by = user
    alert.reviewed_at = timezone.now()
    alert.review_note = note.strip()[:500]
    alert.save()
    return alert


@transaction.atomic
def set_issue_status(issue: Issue, user, status: str) -> Issue:
    if not is_manager(user):
        raise PermissionDenied("Only managers can change issue status.")
    if status not in Issue.Status.values:
        raise ActionError("Unknown issue status.")
    issue = Issue.objects.select_for_update().get(pk=issue.pk)
    issue.status = status
    issue.status_changed_by = user
    issue.status_changed_at = timezone.now()
    issue.save()
    return issue
