from django.conf import settings
from django.db import models

from .messages import CustomerMessage


class EvaluationLabel(models.Model):
    """An independent human label for a held-out evaluation message (blind: no model output shown).

    role=annotator: one independent label per person; role=adjudicator: resolves disagreements.
    """

    class Role(models.TextChoices):
        ANNOTATOR = "annotator", "Independent annotator"
        ADJUDICATOR = "adjudicator", "Adjudicator"

    message = models.ForeignKey(CustomerMessage, on_delete=models.CASCADE, related_name="evaluation_labels")
    annotator = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="+")
    role = models.CharField(max_length=12, choices=Role.choices, default=Role.ANNOTATOR)
    message_type = models.CharField(max_length=24)
    issue_keys = models.JSONField(default=list, blank=True)
    notes = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["message_id", "created_at"]
        constraints = [
            models.UniqueConstraint(fields=["message", "annotator", "role"], name="uniq_label_per_annotator_role"),
        ]
