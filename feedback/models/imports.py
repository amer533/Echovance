from django.conf import settings
from django.db import models


class ImportBatch(models.Model):
    """One CSV import attempt (web upload or management command)."""

    class Status(models.TextChoices):
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        COMPLETED_WITH_ERRORS = "completed_with_errors", "Completed with row errors"
        FAILED = "failed", "Failed"

    class Source(models.TextChoices):
        UPLOAD = "upload", "Web/API upload"
        COMMAND = "command", "Management command"

    class DataClass(models.TextChoices):
        PUBLIC = "public", "Public (already published data)"
        INTERNAL = "internal", "Internal (company data, not confidential)"
        CONFIDENTIAL = "confidential", "Confidential (never sent to external AI)"

    source = models.CharField(max_length=16, choices=Source.choices)
    original_filename = models.CharField(max_length=255)
    file_sha256 = models.CharField(max_length=64, db_index=True)
    file_size_bytes = models.PositiveBigIntegerField()
    brand_account = models.CharField(
        max_length=64, help_text="Brand support handle used to separate agent replies from customers.")
    data_classification = models.CharField(
        max_length=16, choices=DataClass.choices, default=DataClass.INTERNAL,
        help_text="Privacy gate input: which external AI providers/tiers may receive this data.")
    status = models.CharField(max_length=32, choices=Status.choices, default=Status.PROCESSING)
    total_rows = models.PositiveIntegerField(default=0)
    accepted_count = models.PositiveIntegerField(default=0)
    duplicate_count = models.PositiveIntegerField(default=0)
    rejected_count = models.PositiveIntegerField(default=0)
    customer_root_count = models.PositiveIntegerField(default=0)
    agent_count = models.PositiveIntegerField(default=0)
    # Row-level errors: [{"row": 12, "field": "created_at", "code": "invalid_date"}] — never raw text.
    errors = models.JSONField(default=list, blank=True)
    failure_reason = models.CharField(max_length=255, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="imports")
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]
        permissions = [("run_pipeline", "Can import data, run analysis and trend detection")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(accepted_count__lte=models.F("total_rows")),
                name="import_accepted_lte_total"),
        ]

    def __str__(self):
        return f"Import #{self.pk} {self.original_filename} ({self.status})"
