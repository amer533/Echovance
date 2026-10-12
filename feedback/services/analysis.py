"""Analysis orchestration: select eligible messages, gate privacy, call the LLM, validate, persist.

Idempotency: an analysis is unique per (message, provider, model, prompt_version). Re-running skips
messages already analysed with the same configuration; failed analyses are retried by updating the
same row. Network calls happen outside DB transactions; persistence of one message (analysis +
mentions + status + supersession of the previous current analysis) is atomic.

Selections:
  pending       new/unprocessed conversations (+ failed ones when retry_failed)
  promote_mock  conversations whose CURRENT analysis is mock -> re-analysed by the real provider;
                on success the real analysis supersedes the mock one (history kept, mentions inactive)
  retry_failed  conversations whose last attempt with this provider/model/prompt failed
  message_ids   explicit conversations
A failed real attempt is stored as status=failed and never becomes the current analysis.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from django.conf import settings
from django.db import transaction
from django.db.models import Exists, OuterRef
from django.utils import timezone

from feedback.ai import extraction
from feedback.ai.client import HALT_CODES, LLMClient, LLMError, LLMResponse, get_llm_client
from feedback.models import AnalysisJob, AnalysisRun, CustomerMessage, IssueMention, MessageAnalysis

from . import privacy_gate
from .grouping import get_or_create_issue
from .llm_budget import NOT_SENT_CODES

logger = logging.getLogger(__name__)

# Errors that end a run immediately (no further messages are attempted).
STOP_CODES = HALT_CODES | NOT_SENT_CODES - {"input_too_large"}
SELECTIONS = ("pending", "promote_mock", "retry_failed", "message_ids")


@dataclass
class _Outcome:
    message: CustomerMessage
    response: LLMResponse | None = None
    error: LLMError | None = None
    truncated: bool = False
    fingerprint: str = ""
    redaction: dict | None = None
    blocked_reason: str = ""
    reused: MessageAnalysis | None = None


class AnalysisService:
    def __init__(self, client: LLMClient | None = None, job: AnalysisJob | None = None):
        self.client = client or get_llm_client()
        self.job = job
        self.source = MessageAnalysis.Source.MOCK if self.client.is_mock else MessageAnalysis.Source.LLM
        self.new_mention_ids: list[int] = []

    # ---------- selection ----------
    def _key(self):
        return dict(provider=self.client.provider_name, model_name=self.client.model,
                    prompt_version=extraction.PROMPT_VERSION)

    def _already_done(self):
        return MessageAnalysis.objects.filter(message=OuterRef("pk"), **self._key()).exclude(
            status=MessageAnalysis.Status.FAILED)

    def _failed_here(self):
        return MessageAnalysis.objects.filter(message=OuterRef("pk"), status=MessageAnalysis.Status.FAILED,
                                              **self._key())

    def select_messages(self, limit: int, message_ids: list[int] | None = None, retry_failed: bool = True,
                        selection: str | None = None):
        selection = selection or ("message_ids" if message_ids else "pending")
        if selection not in SELECTIONS:
            raise ValueError(f"Unknown selection '{selection}'")
        S = CustomerMessage.AnalysisStatus
        qs = CustomerMessage.objects.filter(author_role=CustomerMessage.AuthorRole.CUSTOMER, is_thread_root=True)
        if selection == "message_ids":
            qs = qs.filter(pk__in=message_ids or [])
        elif selection == "pending":
            statuses = [S.PENDING] + ([S.FAILED] if retry_failed else [])
            qs = qs.filter(analysis_status__in=statuses)
        elif selection == "promote_mock":
            qs = qs.filter(Exists(MessageAnalysis.objects.filter(
                message=OuterRef("pk"), is_current=True, source=MessageAnalysis.Source.MOCK)))
        elif selection == "retry_failed":
            qs = qs.filter(Exists(self._failed_here()))
        qs = qs.exclude(Exists(self._already_done()))
        if self.job is not None:
            # Never re-attempt within the same job (failures are retried only by a new, explicit job).
            qs = qs.exclude(Exists(MessageAnalysis.objects.filter(message=OuterRef("pk"), job=self.job)))
        return list(qs.select_related("import_batch").order_by("created_at", "id")[:limit])

    # ---------- execution ----------
    def _reusable(self, fingerprint: str) -> MessageAnalysis | None:
        if self.client.is_mock or not fingerprint:
            return None
        return (MessageAnalysis.objects.filter(input_fingerprint=fingerprint, source=MessageAnalysis.Source.LLM,
                                               reused_from__isnull=True, **self._key())
                .exclude(status=MessageAnalysis.Status.FAILED).exclude(validated_output=None).first())

    def _call(self, message: CustomerMessage) -> _Outcome:
        gate = privacy_gate.check(message, self.client.provider_name, self.client.is_mock)
        prompt, truncated = extraction.build_user_prompt(gate.redaction.text, settings.LLM["MAX_INPUT_CHARS"])
        # Same fingerprint the client logs for the request (system prompt embeds the prompt version).
        fp = LLMClient.request_fingerprint(self.client.provider_name, self.client.model, "message_extraction",
                                           extraction.SYSTEM_PROMPT, prompt)
        out = _Outcome(message, truncated=truncated, fingerprint=fp, redaction=gate.redaction.counts)
        if not gate.allowed:
            out.blocked_reason = gate.reason
            return out
        reusable = self._reusable(fp)
        if reusable is not None:
            out.reused = reusable
            return out
        try:
            out.response = self.client.complete_json(extraction.SYSTEM_PROMPT, prompt, "message_extraction",
                                                     extraction.OUTPUT_SCHEMA, purpose="message_extraction")
        except LLMError as exc:
            out.error = exc
        return out

    def _job_interrupted(self) -> str:
        if self.job is None:
            return ""
        status = AnalysisJob.objects.filter(pk=self.job.pk).values_list("status", flat=True).first()
        if status in (AnalysisJob.Status.CANCELLED, AnalysisJob.Status.PAUSED):
            return f"Job {status} by a user."
        AnalysisJob.objects.filter(pk=self.job.pk).update(heartbeat_at=timezone.now())
        return ""

    def run(self, *, limit: int, message_ids: list[int] | None = None, user=None, retry_failed: bool = True,
            workers: int = 1, selection: str | None = None, stop_on_error: bool = False) -> AnalysisRun:
        run = AnalysisRun.objects.create(provider=self.client.provider_name, model_name=self.client.model,
                                         prompt_version=extraction.PROMPT_VERSION,
                                         is_mock=self.client.is_mock, started_by=user)
        try:
            if not self.client.is_mock:
                # Real providers: sequential, never more messages than the remaining request budget.
                from feedback.services import llm_budget
                workers = 1
                limit = min(limit, llm_budget.remaining())
                if limit == 0:
                    run.stopped_reason = "No real LLM requests remaining (budget used up or halted)."
            messages = self.select_messages(limit, message_ids, retry_failed, selection) if limit > 0 else []
            run.requested = len(messages)
            run.save(update_fields=["requested", "stopped_reason"])
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for outcome in pool.map(self._call, messages):
                        self._persist(run, outcome)
            else:
                for m in messages:
                    reason = self._job_interrupted()
                    if reason:
                        run.stopped_reason = reason
                        break
                    outcome = self._call(m)
                    if outcome.error is not None and outcome.error.code in STOP_CODES:
                        if outcome.error.code not in NOT_SENT_CODES:
                            self._persist(run, outcome)  # a real request was sent and failed
                        run.stopped_reason = f"Stopped: {outcome.error.code}. {outcome.error.detail}"[:200]
                        break
                    failed_before = run.failed
                    self._persist(run, outcome)
                    if stop_on_error and run.failed > failed_before:
                        code = ("privacy_blocked" if outcome.blocked_reason else
                                outcome.error.code if outcome.error else "schema_validation_failed")
                        run.stopped_reason = f"Stopped on first error (stop_on_error): {code}"[:200]
                        break
            run.skipped = max(0, run.requested - run.succeeded - run.needs_review - run.failed)
            run.status = AnalysisRun.Status.COMPLETED
        except Exception:
            logger.exception("Analysis run %s aborted", run.pk)
            run.status = AnalysisRun.Status.FAILED
        run.finished_at = timezone.now()
        run.save()
        self._after_run()
        return run

    def _after_run(self) -> None:
        """Keep downstream derived data honest: grouping suggestions and alert staleness."""
        from . import grouping, trends
        try:
            if self.new_mention_ids:
                grouping.suggest_for_mentions(self.new_mention_ids)
            trends.refresh_alert_staleness()
        except Exception:  # derived data must never break the analysis itself
            logger.exception("Post-run refresh failed")

    # ---------- persistence ----------
    def _validate(self, data: dict, msg: CustomerMessage, outcome: _Outcome, defaults: dict):
        S = MessageAnalysis.Status
        try:
            validated = extraction.validate_extraction(data, msg.text)
        except extraction.ExtractionSchemaError as exc:
            defaults.update(status=S.FAILED, error_code="schema_validation_failed", error_detail=str(exc)[:300])
            return None
        if outcome.truncated:
            validated.review_reasons.append("input_truncated")
        defaults.update(status=S.NEEDS_REVIEW if validated.needs_review else S.SUCCEEDED,
                        message_type=validated.message_type, is_complaint=validated.is_complaint,
                        review_reasons=validated.review_reasons, validated_output=validated.as_dict())
        return validated

    @staticmethod
    def _reuse_payload(prev: MessageAnalysis) -> dict:
        out = prev.validated_output or {}
        model_flag = [r for r in prev.review_reasons if r.startswith("model_flagged")]
        return {"message_type": out.get("message_type", prev.message_type),
                "is_complaint": out.get("is_complaint", prev.is_complaint),
                "overall_sentiment": out.get("overall_sentiment", "neutral"),
                "aspects": [{**{k: a[k] for k in ("category", "subtype", "sentiment", "summary", "evidence")},
                             "evidence_basis": a.get("evidence_basis", "explicit"),
                             "related_symptoms": a.get("related_symptoms", [])}
                            for a in out.get("aspects", [])],
                "needs_review": bool(model_flag),
                "review_reason": model_flag[0].partition(": ")[2] if model_flag else ""}

    def _persist(self, run: AnalysisRun, outcome: _Outcome) -> None:
        msg = outcome.message
        S = MessageAnalysis.Status
        defaults = {"run": run, "job": self.job, "is_mock": self.client.is_mock, "source": self.source,
                    "error_code": "", "error_detail": "", "review_reasons": [], "validated_output": None,
                    "input_fingerprint": outcome.fingerprint, "redaction": outcome.redaction or {},
                    "reused_from": None}
        validated = None
        if outcome.blocked_reason:
            defaults.update(status=S.FAILED, error_code="privacy_blocked", error_detail=outcome.blocked_reason[:300],
                            attempts=0)
        elif outcome.reused is not None:
            defaults.update(reused_from=outcome.reused, attempts=0, latency_ms=0, total_ms=0, provider_latency_ms=0,
                            queue_wait_ms=0, input_tokens=0, output_tokens=0)
            validated = self._validate(self._reuse_payload(outcome.reused), msg, outcome, defaults)
        elif outcome.error is not None:
            defaults.update(status=S.FAILED, error_code=outcome.error.code,
                            error_detail=outcome.error.detail, attempts=outcome.error.attempts)
            if outcome.error.timing:
                t = outcome.error.timing
                defaults.update(latency_ms=t["total_ms"], total_ms=t["total_ms"],
                                provider_latency_ms=t["provider_latency_ms"], queue_wait_ms=t["queue_wait_ms"])
        else:
            resp = outcome.response
            defaults.update(attempts=resp.attempts, latency_ms=resp.total_ms, total_ms=resp.total_ms,
                            provider_latency_ms=resp.provider_latency_ms, queue_wait_ms=resp.queue_wait_ms,
                            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens)
            validated = self._validate(resp.data, msg, outcome, defaults)

        with transaction.atomic():
            msg = CustomerMessage.objects.select_for_update().get(pk=msg.pk)
            analysis, _ = MessageAnalysis.objects.update_or_create(message=msg, **self._key(), defaults=defaults)
            if analysis.status == S.FAILED:
                run.failed += 1
                if not msg.analyses.filter(is_current=True).exists():
                    msg.analysis_status = (CustomerMessage.AnalysisStatus.PRIVACY_BLOCKED if outcome.blocked_reason
                                           else CustomerMessage.AnalysisStatus.FAILED)
                    msg.save(update_fields=["analysis_status"])
            else:
                # Supersede previous current analyses (e.g. mock) and their mentions; history is kept.
                old = msg.analyses.filter(is_current=True).exclude(pk=analysis.pk)
                IssueMention.objects.filter(analysis__in=old, is_active=True).update(is_active=False)
                old.update(is_current=False)
                analysis.is_current = True
                analysis.save(update_fields=["is_current"])
                analysis.mentions.all().delete()
                # Only fully validated analyses contribute issue mentions (counted in trends).
                if analysis.status == S.SUCCEEDED:
                    for a in validated.aspects:
                        mention = IssueMention.objects.create(
                            issue=get_or_create_issue(a.category, a.subtype), analysis=analysis, message=msg,
                            message_created_at=msg.created_at, sentiment=a.sentiment, summary=a.summary,
                            evidence_text=a.evidence[:500], evidence_start=a.start, evidence_end=a.end,
                            source=self.source)
                        self.new_mention_ids.append(mention.pk)
                    run.succeeded += 1
                    msg.analysis_status = CustomerMessage.AnalysisStatus.COMPLETED
                else:
                    run.needs_review += 1
                    msg.analysis_status = CustomerMessage.AnalysisStatus.NEEDS_REVIEW
                msg.save(update_fields=["analysis_status"])
            if self.job is not None:
                from django.db.models import F
                inc = {"processed": F("processed") + 1}
                if outcome.blocked_reason:
                    inc["privacy_blocked"] = F("privacy_blocked") + 1
                elif analysis.status == S.FAILED:
                    inc["failed"] = F("failed") + 1
                elif analysis.status == S.NEEDS_REVIEW:
                    inc["needs_review"] = F("needs_review") + 1
                else:
                    inc["succeeded"] = F("succeeded") + 1
                if outcome.reused is not None:
                    inc["reused"] = F("reused") + 1
                AnalysisJob.objects.filter(pk=self.job.pk).update(**inc)
        run.save(update_fields=["succeeded", "needs_review", "failed"])
