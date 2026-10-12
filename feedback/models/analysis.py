from django.conf import settings
from django.db import models

from .messages import CustomerMessage


class AnalysisRun(models.Model):
    """One invocation of the analysis pipeline (web request or management command)."""

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    provider = models.CharField(max_length=32)
    model_name = models.CharField(max_length=100)
    prompt_version = models.CharField(max_length=32)
    is_mock = models.BooleanField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.RUNNING)
    requested = models.PositiveIntegerField(default=0)
    succeeded = models.PositiveIntegerField(default=0)
    needs_review = models.PositiveIntegerField(default=0)
    failed = models.PositiveIntegerField(default=0)
    skipped = models.PositiveIntegerField(default=0)
    stopped_reason = models.CharField(max_length=200, blank=True,
                                      help_text="Why a run stopped early (budget, quota, halt).")
    started_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name="analysis_runs")
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]


class AnalysisJob(models.Model):
    """An explicitly authorised, bounded, resumable batch of analysis work (see services/jobs.py)."""

    class Selection(models.TextChoices):
        PENDING = "pending", "New / unprocessed conversations"
        PROMOTE_MOCK = "promote_mock", "Re-analyse mock-analysed conversations with the real LLM"
        RETRY_FAILED = "retry_failed", "Retry failed real analyses"
        MESSAGE_IDS = "message_ids", "Specific conversations"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft (awaiting authorisation)"
        AUTHORIZED = "authorized", "Authorised (queued)"
        RUNNING = "running", "Running"
        PAUSED = "paused", "Paused"
        CANCELLED = "cancelled", "Cancelled"
        COMPLETED = "completed", "Completed"
        STOPPED = "stopped", "Stopped (limit or provider error)"

    selection = models.CharField(max_length=16, choices=Selection.choices)
    message_ids = models.JSONField(default=list, blank=True)
    provider = models.CharField(max_length=32)
    model_name = models.CharField(max_length=100)
    prompt_version = models.CharField(max_length=32)
    is_mock = models.BooleanField(default=False)
    max_messages = models.PositiveIntegerField()
    max_requests = models.PositiveIntegerField()
    max_input_tokens = models.PositiveIntegerField()
    max_output_tokens = models.PositiveIntegerField()
    max_cost_usd = models.DecimalField(max_digits=10, decimal_places=4)
    stop_on_error = models.BooleanField(default=False, help_text="Stop the whole job at the first failed, "
                                        "privacy-blocked or provider-error message (pilot / verification runs)")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.DRAFT, db_index=True)
    processed = models.PositiveIntegerField(default=0)
    succeeded = models.PositiveIntegerField(default=0)
    needs_review = models.PositiveIntegerField(default=0)
    failed = models.PositiveIntegerField(default=0)
    reused = models.PositiveIntegerField(default=0)
    privacy_blocked = models.PositiveIntegerField(default=0)
    requests_used = models.PositiveIntegerField(default=0)
    input_tokens_used = models.PositiveIntegerField(default=0)
    output_tokens_used = models.PositiveIntegerField(default=0)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    stop_reason = models.CharField(max_length=300, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="+")
    authorized_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                      on_delete=models.PROTECT, related_name="+")
    authorized_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            # Only draft (never authorised) or cancelled jobs may lack an authoriser; anything that can
            # run or has run must name the manager who authorised it.
            models.CheckConstraint(condition=models.Q(status__in=["draft", "cancelled"]) |
                                   models.Q(authorized_by__isnull=False),
                                   name="job_authorized_before_running"),
        ]

    def __str__(self):
        return f"Job #{self.pk} {self.selection} ({self.status})"


class MessageAnalysis(models.Model):
    """Structured LLM extraction result for one customer message.

    Unique per (message, provider, model, prompt_version) so re-running analysis is idempotent.
    Only one analysis per message is `is_current`; issue counts use mentions of current analyses.
    """

    class Status(models.TextChoices):
        SUCCEEDED = "succeeded", "Succeeded"
        NEEDS_REVIEW = "needs_review", "Needs human review"
        FAILED = "failed", "Failed"

    class Source(models.TextChoices):
        LLM = "llm", "Real LLM"
        MOCK = "mock", "Mock (demo / test double)"
        HUMAN = "human", "Human correction"

    class MessageType(models.TextChoices):
        COMPLAINT = "complaint", "Complaint"
        FEATURE_REQUEST = "feature_request", "Feature request"
        QUESTION = "question", "Question"
        PRAISE = "praise", "Praise"
        OTHER = "other", "Other / unclear"

    message = models.ForeignKey(CustomerMessage, on_delete=models.CASCADE, related_name="analyses")
    run = models.ForeignKey(AnalysisRun, null=True, blank=True, on_delete=models.SET_NULL,
                            related_name="analyses")
    provider = models.CharField(max_length=32)
    model_name = models.CharField(max_length=100)
    prompt_version = models.CharField(max_length=32)
    is_mock = models.BooleanField(help_text="True when produced by the mock test provider, not a real LLM")
    source = models.CharField(max_length=8, choices=Source.choices, default=Source.LLM, db_index=True)
    job = models.ForeignKey(AnalysisJob, null=True, blank=True, on_delete=models.SET_NULL, related_name="analyses")
    input_fingerprint = models.CharField(max_length=64, blank=True, db_index=True,
                                         help_text="SHA-256 of the redacted prompt (request de-duplication)")
    reused_from = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="+",
                                    help_text="Validated result reused from an identical earlier request")
    redaction = models.JSONField(default=dict, blank=True, help_text="Counts of redacted entity types")
    corrected_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                     on_delete=models.SET_NULL, related_name="+")
    status = models.CharField(max_length=16, choices=Status.choices)
    is_current = models.BooleanField(default=False)
    message_type = models.CharField(max_length=24, choices=MessageType.choices, blank=True)
    is_complaint = models.BooleanField(null=True)
    review_reasons = models.JSONField(default=list, blank=True)
    validated_output = models.JSONField(null=True, blank=True)
    error_code = models.CharField(max_length=64, blank=True)
    error_detail = models.CharField(max_length=500, blank=True)
    attempts = models.PositiveSmallIntegerField(default=0)
    latency_ms = models.PositiveIntegerField(null=True, blank=True,
                                             help_text="Legacy: total wall time of the call INCLUDING budget/rate-limit "
                                                       "wait (same as total_ms). Use provider_latency_ms for model speed.")
    provider_latency_ms = models.PositiveIntegerField(null=True, blank=True,
                                                      help_text="Time inside provider calls only (all attempts)")
    queue_wait_ms = models.PositiveIntegerField(null=True, blank=True,
                                                help_text="Budget admission incl. rate-limiter sleep (all attempts)")
    total_ms = models.PositiveIntegerField(null=True, blank=True,
                                           help_text="Wait + provider + retry backoff + parsing")
    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["message", "provider", "model_name", "prompt_version"],
                                    name="uniq_analysis_per_model_prompt"),
            models.UniqueConstraint(fields=["message"], condition=models.Q(is_current=True),
                                    name="one_current_analysis_per_message"),
        ]
        indexes = [models.Index(fields=["status", "is_current"], name="analysis_status_idx")]

    def __str__(self):
        return f"Analysis {self.pk} of message {self.message_id} ({self.status})"
