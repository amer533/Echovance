from datetime import timedelta

from django.conf import settings
from django.db import models

from .issues import Issue, IssueMention


class IssueAlert(models.Model):
    """A deterministic trend-rule hit for one issue and one current/baseline window pair."""

    class Rule(models.TextChoices):
        SURGE = "surge", "Increase versus baseline"
        EMERGING = "emerging", "Emerging (no baseline mentions)"

    class Scope(models.TextChoices):
        REAL = "real", "Real AI + human data"
        DEMO = "demo", "Mock demo data"
        LEGACY_MIXED = "legacy_mixed", "Legacy (computed over mixed mock + real data)"

    class Status(models.TextChoices):
        OPEN = "open", "Awaiting review"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        DISMISSED = "dismissed", "Dismissed"

    issue = models.ForeignKey(Issue, on_delete=models.CASCADE, related_name="alerts")
    rule = models.CharField(max_length=16, choices=Rule.choices)
    rule_version = models.CharField(max_length=32)
    data_scope = models.CharField(max_length=16, choices=Scope.choices, default=Scope.REAL, db_index=True)
    revision = models.PositiveSmallIntegerField(default=1)
    input_fingerprint = models.CharField(max_length=64, blank=True,
                                         help_text="Hash of contributing mention ids at creation time")
    is_stale = models.BooleanField(default=False, db_index=True)
    stale_reason = models.CharField(max_length=300, blank=True)
    stale_at = models.DateTimeField(null=True, blank=True)
    superseded_by = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    window_start = models.DateTimeField()
    window_end = models.DateTimeField()
    baseline_start = models.DateTimeField()
    baseline_end = models.DateTimeField()
    current_count = models.PositiveIntegerField()
    baseline_total = models.PositiveIntegerField()
    baseline_mean = models.FloatField()
    growth_ratio = models.FloatField(null=True, blank=True, help_text="Null when baseline mean is 0")
    current_share = models.FloatField(help_text="Mentions / analysed root messages in current window")
    baseline_share = models.FloatField()
    current_window_volume = models.PositiveIntegerField()
    baseline_window_volumes = models.JSONField(default=list)
    thresholds = models.JSONField(help_text="Threshold values applied, for auditability")
    explanation = models.TextField()
    evidence = models.ManyToManyField(IssueMention, related_name="alerts", blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.OPEN, db_index=True)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                                    on_delete=models.SET_NULL, related_name="reviewed_alerts")
    reviewed_at = models.DateTimeField(null=True, blank=True)
    review_note = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-window_end", "-current_count"]
        constraints = [
            models.UniqueConstraint(fields=["issue", "window_start", "window_end", "rule_version", "data_scope",
                                            "revision"], name="uniq_alert_issue_window_rule_scope_rev"),
            models.UniqueConstraint(fields=["issue", "window_start", "window_end", "rule_version", "data_scope"],
                                    condition=models.Q(is_stale=False), name="one_fresh_alert_per_window"),
            models.CheckConstraint(condition=models.Q(window_end__gt=models.F("window_start")),
                                   name="alert_window_valid"),
        ]

    def __str__(self):
        return f"{self.get_rule_display()} – {self.issue.key} ({self.window_start:%Y-%m-%d})"

    @property
    def window_last_day(self):
        """Last day INSIDE the half-open window (window_end itself is excluded)."""
        return (self.window_end - timedelta(days=1)).date()

    @property
    def current_share_pct(self) -> float:
        return round(self.current_share * 100, 1)

    @property
    def baseline_share_pct(self) -> float:
        return round(self.baseline_share * 100, 1)
