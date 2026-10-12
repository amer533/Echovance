from django.db import models

# Ceilings are configuration (settings.LLM), enforced by services/llm_budget.py. The former fixed
# DB constraint (authorised <= 20) was replaced by LLM_LIFETIME_REQUEST_CEILING (default 20).


class LLMBudget(models.Model):
    """Singleton (pk=1) holding the authorised number of real LLM requests and the halt switch."""

    authorized_requests = models.PositiveSmallIntegerField(
        default=3, help_text="Total real requests currently authorised (cumulative, never resets).")
    halted = models.BooleanField(default=False, help_text="True after a quota/rate-limit/config error.")
    halt_reason = models.CharField(max_length=300, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    last_request_at = models.DateTimeField(null=True, blank=True, help_text="For rate limiting")

    class Meta:
        constraints = [
            models.CheckConstraint(condition=models.Q(pk=1), name="llm_budget_singleton"),
        ]

    def __str__(self):
        return f"LLM budget: {self.authorized_requests} authorised"


class LLMRequestLog(models.Model):
    """One row per real outbound LLM request attempt (created BEFORE the request is sent)."""

    class Status(models.TextChoices):
        IN_FLIGHT = "in_flight", "In flight"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        INTERRUPTED = "interrupted", "Interrupted (process stopped mid-request)"

    job = models.ForeignKey("feedback.AnalysisJob", null=True, blank=True, on_delete=models.SET_NULL,
                            related_name="requests")
    request_fingerprint = models.CharField(max_length=64, blank=True, db_index=True)
    estimated_input_tokens = models.PositiveIntegerField(null=True, blank=True)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)
    provider = models.CharField(max_length=32)
    model_name = models.CharField(max_length=100)
    purpose = models.CharField(max_length=40)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.IN_FLIGHT, db_index=True)
    error_code = models.CharField(max_length=64, blank=True)
    http_status = models.PositiveSmallIntegerField(null=True, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True, help_text="Provider call time only")
    queue_wait_ms = models.PositiveIntegerField(null=True, blank=True,
                                                help_text="Time spent waiting for admission (rate limiter) before sending")
    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    total_tokens = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
