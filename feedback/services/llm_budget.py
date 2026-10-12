"""Persisted request/cost budget for REAL LLM providers (budget v2).

Every real request attempt is admitted here BEFORE it is sent and logged in LLMRequestLog.
Admission (under a row lock on the LLMBudget singleton) requires ALL of:

  * live calls not halted (quota / rate-limit / auth / model errors halt until a human clears it)
  * used < min(authorised total, LLM_LIFETIME_REQUEST_CEILING)      [global, owner-authorised]
  * known prices for the model (OPENAI_PRICE_*; unknown prices refuse the request)
  * estimated input tokens <= LLM_MAX_INPUT_TOKENS_PER_REQUEST
  * month-to-date cost + estimated cost <= LLM_MONTHLY_COST_CEILING_USD
  * the job's own limits (requests, input/output tokens, cost)        [per-run, manager-authorised]
  * rate limit: at least 60 / LLM_RATE_LIMIT_RPM seconds since the previous request

The authorised total only changes through `manage.py llm_budget --authorize N --confirm`.
"""
from __future__ import annotations

import time
from datetime import timedelta
from decimal import Decimal

from django.conf import settings
from django.db import transaction
from django.db.models import Count, F, Q, Sum
from django.utils import timezone

from feedback.ai.client import LLMError, ProviderReply
from feedback.models import AnalysisJob, LLMBudget, LLMRequestLog

MTOK = Decimal(1_000_000)


class BudgetError(LLMError):
    def __init__(self, code: str, detail: str):
        super().__init__(code, detail, retryable=False)


# Codes meaning "no request was sent" — callers must not record a provider failure for these.
NOT_SENT_CODES = {"budget_exhausted", "provider_halted", "pricing_unknown", "input_too_large",
                  "cost_ceiling_reached", "job_request_limit", "job_token_limit", "job_cost_limit"}


def lifetime_ceiling() -> int:
    return settings.LLM["LIFETIME_REQUEST_CEILING"]


def _budget(lock: bool = False) -> LLMBudget:
    initial = min(settings.LLM["INITIAL_AUTHORIZED_REQUESTS"], lifetime_ceiling())
    LLMBudget.objects.get_or_create(pk=1, defaults={"authorized_requests": initial})
    qs = LLMBudget.objects.select_for_update() if lock else LLMBudget.objects
    return qs.get(pk=1)


def provider_tier(provider: str) -> str:
    return "paid"  # OpenAI API usage is always billed


def prices(provider: str) -> tuple[Decimal, Decimal] | None:
    """USD per 1M (input, output) tokens, or None when unknown (requests are then refused)."""
    if provider != "openai":
        return None
    cfg = settings.LLM
    pin, pout = cfg.get("OPENAI_PRICE_INPUT_PER_MTOK", ""), cfg.get("OPENAI_PRICE_OUTPUT_PER_MTOK", "")
    if pin == "" or pout == "":
        return None
    return Decimal(str(pin)), Decimal(str(pout))


def estimate_tokens(*texts: str) -> int:
    """Conservative token estimate (~3 characters per token) used for admission only."""
    return sum(len(t) for t in texts) // 3 + 1


def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    p = prices(provider)
    if p is None:
        return None
    return (Decimal(input_tokens) * p[0] + Decimal(output_tokens) * p[1]) / MTOK


def month_cost() -> Decimal:
    start = timezone.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return LLMRequestLog.objects.filter(created_at__gte=start).aggregate(c=Sum("cost_usd"))["c"] or Decimal(0)


def status() -> dict:
    b = _budget()
    agg = LLMRequestLog.objects.aggregate(
        used=Count("id"), failed=Count("id", filter=Q(status="failed")),
        succeeded=Count("id", filter=Q(status="succeeded")),
        interrupted=Count("id", filter=Q(status="interrupted")),
        input_tokens=Sum("input_tokens"), output_tokens=Sum("output_tokens"), total_tokens=Sum("total_tokens"),
        cost=Sum("cost_usd"))
    limit = min(b.authorized_requests, lifetime_ceiling())
    return {"used": agg["used"], "succeeded": agg["succeeded"], "failed": agg["failed"],
            "interrupted": agg["interrupted"], "authorized": b.authorized_requests,
            "lifetime_ceiling": lifetime_ceiling(), "hard_limit": lifetime_ceiling(),
            "remaining": max(0, limit - agg["used"]), "halted": b.halted, "halt_reason": b.halt_reason,
            "input_tokens": agg["input_tokens"] or 0, "output_tokens": agg["output_tokens"] or 0,
            "total_tokens": agg["total_tokens"] or 0, "cost_usd": str(agg["cost"] or Decimal(0)),
            "month_cost_usd": str(month_cost()),
            "monthly_cost_ceiling_usd": settings.LLM["MONTHLY_COST_CEILING_USD"],
            "rate_limit_rpm": settings.LLM["RATE_LIMIT_RPM"],
            "errors": dict(LLMRequestLog.objects.filter(status="failed").values_list("error_code")
                           .annotate(n=Count("id")))}


def remaining() -> int:
    s = status()
    return 0 if s["halted"] else s["remaining"]


@transaction.atomic
def authorize(total: int) -> LLMBudget:
    if not 0 <= total <= lifetime_ceiling():
        raise ValueError(f"Authorised total must be between 0 and the configured lifetime ceiling "
                         f"{lifetime_ceiling()} (LLM_LIFETIME_REQUEST_CEILING).")
    b = _budget(lock=True)
    b.authorized_requests = total
    b.save()
    return b


@transaction.atomic
def clear_halt() -> LLMBudget:
    b = _budget(lock=True)
    b.halted, b.halt_reason = False, ""
    b.save()
    return b


def close_interrupted(older_than_minutes: int = 10) -> int:
    """Mark requests left 'in_flight' by a crashed process. They keep counting towards the budget."""
    cutoff = timezone.now() - timedelta(minutes=older_than_minutes)
    return LLMRequestLog.objects.filter(status="in_flight", created_at__lt=cutoff).update(
        status="interrupted", error_code="interrupted", finished_at=timezone.now())


class RequestBudget:
    """Guard passed to LLMClient for real providers. Optional `job` adds per-run limits."""

    def __init__(self, job: AnalysisJob | None = None, sleep=time.sleep, now=timezone.now):
        self.job = job
        self._sleep = sleep
        self._now = now

    def _check_job(self, est_in: int, max_out: int, est_cost: Decimal) -> None:
        job = AnalysisJob.objects.select_for_update().get(pk=self.job.pk)
        if job.requests_used >= job.max_requests:
            raise BudgetError("job_request_limit", f"Job #{job.pk} used its {job.max_requests} authorised requests.")
        if job.input_tokens_used + est_in > job.max_input_tokens or job.output_tokens_used + max_out > job.max_output_tokens:
            raise BudgetError("job_token_limit", f"Job #{job.pk} token limit would be exceeded.")
        if job.cost_usd + est_cost > job.max_cost_usd:
            raise BudgetError("job_cost_limit", f"Job #{job.pk} cost limit (${job.max_cost_usd}) would be exceeded.")

    def acquire(self, provider: str, model: str, purpose: str, *, estimated_input_tokens: int = 0,
                fingerprint: str = "") -> int:
        cfg = settings.LLM
        max_out = cfg["MAX_OUTPUT_TOKENS"]
        if estimated_input_tokens > cfg["MAX_INPUT_TOKENS_PER_REQUEST"]:
            raise BudgetError("input_too_large", f"Estimated {estimated_input_tokens} input tokens exceeds the "
                              f"{cfg['MAX_INPUT_TOKENS_PER_REQUEST']} per-request limit.")
        est_cost = estimate_cost(provider, estimated_input_tokens, max_out)
        if est_cost is None:
            raise BudgetError("pricing_unknown", f"No prices configured for {provider}; set "
                              "OPENAI_PRICE_INPUT_PER_MTOK and OPENAI_PRICE_OUTPUT_PER_MTOK in .env.")
        interval = 60.0 / max(1, cfg["RATE_LIMIT_RPM"])
        waited = 0.0
        while True:
            with transaction.atomic():
                b = _budget(lock=True)
                if b.halted:
                    raise BudgetError("provider_halted", f"Live LLM calls are halted ({b.halt_reason}). "
                                      "Run `manage.py llm_budget --clear-halt` after resolving the cause.")
                used = LLMRequestLog.objects.count()
                limit = min(b.authorized_requests, lifetime_ceiling())
                if used >= limit:
                    raise BudgetError("budget_exhausted", f"Real LLM request budget used up ({used}/{limit}). "
                                      "Further real requests need explicit authorisation.")
                ceiling = Decimal(str(cfg["MONTHLY_COST_CEILING_USD"] or "0"))
                if month_cost() + est_cost > ceiling:
                    raise BudgetError("cost_ceiling_reached", f"Monthly cost ceiling ${ceiling} would be exceeded.")
                if self.job is not None:
                    self._check_job(estimated_input_tokens, max_out, est_cost)
                now = self._now()
                wait = 0.0
                if b.last_request_at is not None:
                    wait = interval - (now - b.last_request_at).total_seconds()
                if wait <= 0:
                    log = LLMRequestLog.objects.create(
                        provider=provider, model_name=model, purpose=purpose[:40], job=self.job,
                        request_fingerprint=fingerprint, estimated_input_tokens=estimated_input_tokens,
                        queue_wait_ms=int(round(waited * 1000)))
                    b.last_request_at = now
                    b.save(update_fields=["last_request_at", "updated_at"])
                    if self.job is not None:
                        AnalysisJob.objects.filter(pk=self.job.pk).update(requests_used=F("requests_used") + 1)
                    return log.pk
            pause = min(wait, interval)
            self._sleep(pause)  # rate limit: wait outside the lock, then re-check everything
            waited += pause

    def finish(self, ticket: int, *, ok: bool, latency_ms: int, error: LLMError | None = None,
               reply: ProviderReply | None = None) -> None:
        log = LLMRequestLog.objects.get(pk=ticket)
        fields = {"status": LLMRequestLog.Status.SUCCEEDED if ok else LLMRequestLog.Status.FAILED,
                  "latency_ms": latency_ms, "finished_at": timezone.now()}
        if error is not None:
            fields.update(error_code=error.code, http_status=error.http_status)
        in_t = out_t = 0
        if reply is not None:
            in_t, out_t = reply.input_tokens or 0, reply.output_tokens or 0
            fields.update(input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                          total_tokens=reply.total_tokens)
        cost = estimate_cost(log.provider, in_t, out_t) or Decimal(0)
        fields["cost_usd"] = cost
        LLMRequestLog.objects.filter(pk=ticket).update(**fields)
        if self.job is not None:
            AnalysisJob.objects.filter(pk=self.job.pk).update(
                input_tokens_used=F("input_tokens_used") + in_t, output_tokens_used=F("output_tokens_used") + out_t,
                cost_usd=F("cost_usd") + cost)

    def halt(self, reason: str) -> None:
        with transaction.atomic():
            b = _budget(lock=True)
            b.halted, b.halt_reason = True, reason[:300]
            b.save()
