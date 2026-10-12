from django.db import models

from .imports import ImportBatch


class CustomerMessage(models.Model):
    """A single source tweet: customer message or brand (agent) reply.

    Only customer thread roots are complaint candidates; follow-ups and agent replies are kept
    for traceability/context and are never counted as independent complaints.
    """

    class AuthorRole(models.TextChoices):
        CUSTOMER = "customer", "Customer"
        AGENT = "agent", "Support agent"

    class AnalysisStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        COMPLETED = "completed", "Analysed"
        NEEDS_REVIEW = "needs_review", "Needs review"
        FAILED = "failed", "Failed"
        PRIVACY_BLOCKED = "privacy_blocked", "Blocked by privacy gate"
        NOT_APPLICABLE = "not_applicable", "Not applicable"

    source_message_id = models.BigIntegerField(unique=True, help_text="Original tweet_id")
    import_batch = models.ForeignKey(ImportBatch, on_delete=models.PROTECT, related_name="messages")
    author_role = models.CharField(max_length=16, choices=AuthorRole.choices)
    # Pseudonymous: SHA-256 prefix of the (already anonymised) source author id.
    author_hash = models.CharField(max_length=16, db_index=True)
    brand_account = models.CharField(max_length=64)
    created_at = models.DateTimeField(db_index=True, help_text="Original message timestamp (UTC)")
    text = models.TextField(help_text="Message text after PII redaction")
    in_response_to_id = models.BigIntegerField(null=True, blank=True, db_index=True)
    response_ids = models.CharField(max_length=255, blank=True)
    is_thread_root = models.BooleanField()
    analysis_status = models.CharField(max_length=16, choices=AnalysisStatus.choices, db_index=True)
    imported_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["author_role", "is_thread_root", "created_at"], name="msg_root_time_idx"),
            models.Index(fields=["analysis_status", "created_at"], name="msg_status_time_idx"),
        ]
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(author_role="agent") | models.Q(analysis_status="not_applicable"),
                name="agent_messages_not_analysed"),
        ]

    @property
    def is_complaint_candidate(self) -> bool:
        return self.author_role == self.AuthorRole.CUSTOMER and self.is_thread_root

    def __str__(self):
        return f"{self.author_role} {self.source_message_id}"
