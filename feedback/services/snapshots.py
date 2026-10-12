"""Load a snapshot of genuine real-LLM analyses (e.g. data/spotifycares_public/gemini_snapshot.json).

This lets a fresh database show the real-provider results produced during development WITHOUT a new API
request. Nothing is generated or edited here:
  * only real providers are accepted (never mock or human output);
  * every result goes through the same validator as live LLM output, against the stored message text;
    a result whose evidence is not found verbatim, or whose validated status differs from the recorded
    one, is rejected instead of being adjusted;
  * no LLM request-log or budget entry is written: the requests were made (and counted) elsewhere, so the
    budget of this database is untouched;
  * the analyses are attached to one AnalysisRun whose note states that it is an imported snapshot.
Loading is idempotent (unique per message/provider/model/prompt). Like a successful live analysis, an
imported result becomes the message's current analysis; a previous current analysis (e.g. mock) is
kept in history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from django.db import transaction
from django.utils import timezone

from feedback.ai import extraction
from feedback.models import AnalysisRun, CustomerMessage, IssueMention, MessageAnalysis

from .grouping import get_or_create_issue

SNAPSHOT_KIND = "echovance.llm_analysis_snapshot"
REAL_PROVIDERS = {"gemini", "openai"}


class SnapshotError(ValueError):
    pass


@dataclass
class SnapshotResult:
    run: AnalysisRun | None = None
    imported: int = 0
    already_present: int = 0
    missing_messages: list[int] = field(default_factory=list)
    rejected: list[tuple[int, str]] = field(default_factory=list)


def _validator_input(item: dict) -> dict:
    return {"message_type": item["message_type"], "is_complaint": item["is_complaint"],
            "overall_sentiment": item.get("overall_sentiment", "neutral"),
            "aspects": [{**{k: a[k] for k in ("category", "subtype", "sentiment", "summary", "evidence")},
                         "evidence_basis": a.get("evidence_basis", "explicit"),
                         "related_symptoms": a.get("related_symptoms", [])} for a in item["aspects"]],
            "needs_review": any(r.startswith("model_flagged") for r in item.get("review_reasons", [])),
            "review_reason": ""}


def import_llm_snapshot(snapshot: dict) -> SnapshotResult:
    if snapshot.get("kind") != SNAPSHOT_KIND:
        raise SnapshotError(f"Not an analysis snapshot (expected kind '{SNAPSHOT_KIND}').")
    provider, model, prompt_version = snapshot["provider"], snapshot["model_name"], snapshot["prompt_version"]
    if provider not in REAL_PROVIDERS:
        raise SnapshotError(f"Only real-provider snapshots can be imported, not '{provider}'.")
    S = MessageAnalysis.Status
    key = dict(provider=provider, model_name=model, prompt_version=prompt_version)
    result = SnapshotResult()
    for item in snapshot["analyses"]:
        tweet_id = int(item["tweet_id"])
        msg = CustomerMessage.objects.filter(source_message_id=tweet_id, author_role=CustomerMessage.AuthorRole.CUSTOMER,
                                             is_thread_root=True).first()
        if msg is None:
            result.missing_messages.append(tweet_id)
            continue
        if MessageAnalysis.objects.filter(message=msg, **key).exists():
            result.already_present += 1
            continue
        try:
            validated = extraction.validate_extraction(_validator_input(item), msg.text, prompt_version=prompt_version)
        except extraction.ExtractionSchemaError as exc:
            result.rejected.append((tweet_id, str(exc)))
            continue
        status = S.NEEDS_REVIEW if validated.needs_review else S.SUCCEEDED
        if status != item.get("status"):
            result.rejected.append((tweet_id, f"validated status {status} differs from recorded {item.get('status')}: "
                                              + "; ".join(validated.review_reasons)))
            continue
        if result.run is None:
            result.run = AnalysisRun.objects.create(
                **key, is_mock=False, status=AnalysisRun.Status.COMPLETED,
                stopped_reason=f"Imported snapshot of genuine {provider} results; no API request was made.")
        with transaction.atomic():
            msg = CustomerMessage.objects.select_for_update().get(pk=msg.pk)
            old = msg.analyses.filter(is_current=True)
            IssueMention.objects.filter(analysis__in=old, is_active=True).update(is_active=False)
            old.update(is_current=False)
            analysis = MessageAnalysis.objects.create(
                message=msg, run=result.run, **key, is_mock=False, source=MessageAnalysis.Source.LLM, status=status,
                is_current=True, message_type=validated.message_type, is_complaint=validated.is_complaint,
                review_reasons=validated.review_reasons, validated_output=validated.as_dict(),
                attempts=item.get("attempts") or 0, latency_ms=item.get("latency_ms"),
                input_tokens=item.get("input_tokens"), output_tokens=item.get("output_tokens"))
            if item.get("analysed_at"):  # keep the original analysis time, not the import time
                MessageAnalysis.objects.filter(pk=analysis.pk).update(
                    created_at=datetime.fromisoformat(item["analysed_at"]))
            if status == S.SUCCEEDED:
                for a in validated.aspects:
                    IssueMention.objects.create(
                        issue=get_or_create_issue(a.category, a.subtype), analysis=analysis, message=msg,
                        message_created_at=msg.created_at, sentiment=a.sentiment, summary=a.summary,
                        evidence_text=a.evidence[:500], evidence_start=a.start, evidence_end=a.end,
                        source=MessageAnalysis.Source.LLM)
                msg.analysis_status = CustomerMessage.AnalysisStatus.COMPLETED
                result.run.succeeded += 1
            else:
                msg.analysis_status = CustomerMessage.AnalysisStatus.NEEDS_REVIEW
                result.run.needs_review += 1
            msg.save(update_fields=["analysis_status"])
        result.imported += 1
    if result.run is not None:
        result.run.requested = result.imported
        result.run.finished_at = timezone.now()
        result.run.save()
        from .trends import refresh_alert_staleness
        refresh_alert_staleness()
    return result
