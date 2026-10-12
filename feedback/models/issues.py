from django.conf import settings
from django.db import models

from .analysis import MessageAnalysis
from .messages import CustomerMessage


class Issue(models.Model):
    """A recurring customer problem, identified by a stable taxonomy key (category.subtype)."""

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        MONITORING = "monitoring", "Monitoring"
        RESOLVED = "resolved", "Resolved (manager decision)"

    key = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=50, db_index=True)
    subtype = models.CharField(max_length=50)
    title = models.CharField(max_length=200)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.OPEN)
    status_changed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                          on_delete=models.SET_NULL, related_name="+")
    status_changed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["key"]

    def __str__(self):
        return self.title


class IssueMention(models.Model):
    """One complaint aspect of one message that supports an issue, with verified evidence."""

    issue = models.ForeignKey(Issue, on_delete=models.PROTECT, related_name="mentions")
    analysis = models.ForeignKey(MessageAnalysis, on_delete=models.CASCADE, related_name="mentions")
    message = models.ForeignKey(CustomerMessage, on_delete=models.CASCADE, related_name="mentions")
    # Denormalised from the message so trend windows are an indexed range scan.
    message_created_at = models.DateTimeField()
    is_active = models.BooleanField(default=True, help_text="False once superseded by a newer analysis")
    source = models.CharField(max_length=8, default="llm", db_index=True,
                              help_text="Copied from the analysis: llm | mock | human")
    original_issue = models.ForeignKey(Issue, null=True, blank=True, on_delete=models.PROTECT, related_name="+",
                                       help_text="Set when a reviewer moved this mention to another issue")
    regrouped_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                     on_delete=models.SET_NULL, related_name="+")
    regrouped_at = models.DateTimeField(null=True, blank=True)
    embedding = models.JSONField(null=True, blank=True, help_text="Cached embedding (optional similarity method)")
    embedding_model = models.CharField(max_length=100, blank=True)
    sentiment = models.CharField(max_length=16)
    summary = models.CharField(max_length=300)
    evidence_text = models.CharField(max_length=500)
    evidence_start = models.PositiveIntegerField()
    evidence_end = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-message_created_at"]
        constraints = [
            models.UniqueConstraint(fields=["analysis", "issue"], name="uniq_mention_per_analysis_issue"),
            models.UniqueConstraint(fields=["message", "issue"], condition=models.Q(is_active=True),
                                    name="uniq_active_mention_per_message_issue"),
            models.CheckConstraint(condition=models.Q(evidence_end__gt=models.F("evidence_start")),
                                   name="evidence_span_nonempty"),
        ]
        indexes = [models.Index(fields=["issue", "is_active", "message_created_at"],
                                name="mention_issue_time_idx"),
                   models.Index(fields=["source", "is_active", "message_created_at"],
                                name="mention_source_time_idx")]


class GroupingSuggestion(models.Model):
    """Similarity evidence that a mention may belong to another issue of the SAME category.

    Never applied automatically; a reviewer accepts (mention moves, history kept) or rejects.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Awaiting review"
        ACCEPTED = "accepted", "Accepted (mention moved)"
        REJECTED = "rejected", "Rejected"
        OBSOLETE = "obsolete", "Obsolete (mention superseded)"

    mention = models.ForeignKey(IssueMention, on_delete=models.CASCADE, related_name="grouping_suggestions")
    current_issue = models.ForeignKey(Issue, on_delete=models.CASCADE, related_name="+")
    suggested_issue = models.ForeignKey(Issue, on_delete=models.CASCADE, related_name="+")
    method = models.CharField(max_length=32)
    similarity_current = models.FloatField()
    similarity_suggested = models.FloatField()
    nearest_mentions = models.JSONField(default=list, help_text="Ids of the most similar mentions in the suggested issue")
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING, db_index=True)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                    on_delete=models.SET_NULL, related_name="+")
    reviewed_at = models.DateTimeField(null=True, blank=True)
    review_note = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["mention", "suggested_issue"], name="uniq_suggestion_per_mention_issue"),
            models.CheckConstraint(condition=~models.Q(current_issue=models.F("suggested_issue")),
                                   name="suggestion_targets_other_issue"),
        ]
