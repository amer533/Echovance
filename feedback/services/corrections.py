"""Human correction of an analysis (source=human).

A correction is a new MessageAnalysis that becomes current (the previous analysis and its mentions
stay in history, inactive). Evidence must still quote the stored message verbatim; the same
validator as for LLM output is used, so a person cannot attach unsupported evidence either.
Human-corrected analyses count in the "real" data scope and are labelled as human everywhere.
They are NOT used as AI-quality evidence (evaluation compares against LLM predictions only).
"""
from __future__ import annotations

from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.utils import timezone

from feedback.ai import extraction
from feedback.models import CustomerMessage, IssueMention, MessageAnalysis

from .actions import ActionError
from .grouping import get_or_create_issue


def correct_analysis(message: CustomerMessage, user, *, message_type: str, aspects: list[dict],
                     note: str = "") -> MessageAnalysis:
    if not (user and user.is_authenticated and user.has_perm("feedback.run_pipeline")):
        raise PermissionDenied("Correcting analyses needs the run_pipeline permission.")
    if not message.is_complaint_candidate:
        raise ActionError("Only customer conversation roots can be analysed.")
    data = {"message_type": message_type, "is_complaint": message_type == "complaint",
            "overall_sentiment": "negative" if message_type == "complaint" else "neutral",
            "aspects": [{"category": a["category"], "subtype": a["subtype"], "sentiment": a.get("sentiment", "negative"),
                         "summary": a.get("summary") or "Corrected by a reviewer.", "evidence": a["evidence"],
                         "evidence_basis": "explicit", "related_symptoms": []}
                        for a in aspects],
            "needs_review": False, "review_reason": ""}
    try:
        validated = extraction.validate_extraction(data, message.text)
    except extraction.ExtractionSchemaError as exc:
        raise ActionError(f"Invalid correction: {exc}") from None
    # A reviewer may deliberately choose an "other" subtype; that flag only guards model uncertainty.
    validated.review_reasons = [r for r in validated.review_reasons if not r.endswith("_uncertain_subtype")]
    if validated.review_reasons:
        raise ActionError("Correction rejected: " + "; ".join(validated.review_reasons) +
                          ". Evidence must be copied exactly from the message.")
    now = timezone.now()
    with transaction.atomic():
        msg = CustomerMessage.objects.select_for_update().get(pk=message.pk)
        old = msg.analyses.filter(is_current=True)
        IssueMention.objects.filter(analysis__in=old, is_active=True).update(is_active=False)
        old.update(is_current=False)
        analysis = MessageAnalysis.objects.create(
            message=msg, provider="human", model_name=f"user-{user.pk}", prompt_version=f"correction-{now:%Y%m%d%H%M%S%f}",
            is_mock=False, source=MessageAnalysis.Source.HUMAN, status=MessageAnalysis.Status.SUCCEEDED,
            is_current=True, message_type=validated.message_type, is_complaint=validated.is_complaint,
            review_reasons=[f"human_correction: {note[:200]}"] if note else [],
            validated_output=validated.as_dict(), corrected_by=user, attempts=0)
        for a in validated.aspects:
            IssueMention.objects.create(issue=get_or_create_issue(a.category, a.subtype), analysis=analysis,
                                        message=msg, message_created_at=msg.created_at, sentiment=a.sentiment,
                                        summary=a.summary, evidence_text=a.evidence[:500], evidence_start=a.start,
                                        evidence_end=a.end, source="human")
        msg.analysis_status = CustomerMessage.AnalysisStatus.COMPLETED
        msg.save(update_fields=["analysis_status"])
    from .trends import refresh_alert_staleness
    refresh_alert_staleness()
    return analysis
