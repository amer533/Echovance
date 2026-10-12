from django.conf import settings
from django.db import models

from .alerts import IssueAlert
from .issues import Issue, IssueMention


class ActionItem(models.Model):
    """An operational action proposed for an issue.

    State machine (enforced in feedback.services.actions.ActionService):
        proposed -> approved | rejected
        approved -> in_progress | rejected
        in_progress -> completed
    AI suggestions are always created as `proposed`; completing an action never resolves an issue.
    """

    class Status(models.TextChoices):
        PROPOSED = "proposed", "Proposed (awaiting review)"
        APPROVED = "approved", "Approved"
        IN_PROGRESS = "in_progress", "In progress"
        COMPLETED = "completed", "Completed"
        REJECTED = "rejected", "Rejected"

    class Origin(models.TextChoices):
        AI = "ai", "AI suggestion"
        MANUAL = "manual", "Created by user"

    issue = models.ForeignKey(Issue, on_delete=models.PROTECT, related_name="actions")
    alert = models.ForeignKey(IssueAlert, null=True, blank=True, on_delete=models.SET_NULL,
                              related_name="actions")
    title = models.CharField(max_length=160)
    description = models.TextField(max_length=2000)
    rationale = models.TextField(max_length=1000, blank=True)
    steps = models.JSONField(default=list, blank=True, help_text="Concrete investigation/improvement steps")
    limitations = models.TextField(max_length=1000, blank=True)
    uncertainty = models.CharField(max_length=8, blank=True, help_text="AI-stated: low | medium | high")
    evidence = models.ManyToManyField(IssueMention, blank=True, related_name="actions",
                                      help_text="Customer evidence the proposal cites")
    origin = models.CharField(max_length=8, choices=Origin.choices)
    ai_provider = models.CharField(max_length=32, blank=True)
    ai_model = models.CharField(max_length=100, blank=True)
    ai_prompt_version = models.CharField(max_length=32, blank=True)
    ai_is_mock = models.BooleanField(default=False)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PROPOSED, db_index=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="created_actions")
    assignee = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                 on_delete=models.SET_NULL, related_name="assigned_actions")
    due_date = models.DateField(null=True, blank=True)
    decided_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="decided_actions")
    decided_at = models.DateTimeField(null=True, blank=True)
    decision_note = models.CharField(max_length=500, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    completion_note = models.CharField(max_length=1000, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        permissions = [("manage_actions", "Can approve, reject and assign actions")]
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(status__in=["in_progress", "completed"]) | models.Q(assignee__isnull=False),
                name="active_action_has_assignee"),
            models.CheckConstraint(
                condition=models.Q(status="proposed") | models.Q(decided_by__isnull=False),
                name="decided_action_has_decider"),
        ]

    def __str__(self):
        return self.title


class ActionEvent(models.Model):
    """Append-only audit trail for action changes."""

    action = models.ForeignKey(ActionItem, on_delete=models.CASCADE, related_name="events")
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="+")
    event = models.CharField(max_length=32)
    from_status = models.CharField(max_length=16, blank=True)
    to_status = models.CharField(max_length=16, blank=True)
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
