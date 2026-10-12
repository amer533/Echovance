"""Analysis jobs: explicitly authorised, bounded, resumable batches of LLM work.

Lifecycle: draft --authorize(manager)--> authorized --worker--> running --> completed | stopped
           authorized/running --pause--> paused --resume--> authorized ; any open state --cancel--> cancelled
* Creating a job needs feedback.run_pipeline; authorising it needs feedback.manage_actions.
* A real-provider job can only be authorised when the global remaining request budget covers its
  max_requests (the global total itself is raised only by the owner via `llm_budget --authorize`).
* The worker (`manage.py process_jobs`) processes one message at a time. Every message is atomic, and
  selection skips messages already done, so stopping/crashing and resuming is safe.
* Per-job limits (requests, tokens, cost) are enforced by RequestBudget(job=...) before each request.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.utils import timezone

from feedback.ai import extraction
from feedback.ai.client import LLMClient, get_llm_client
from feedback.models import AnalysisJob

from . import llm_budget
from .actions import ActionError, is_manager
from .analysis import AnalysisService

J = AnalysisJob.Status
OPEN = {J.DRAFT, J.AUTHORIZED, J.RUNNING, J.PAUSED}


def per_message_estimate(provider: str) -> dict:
    """Admission-style estimate for one extraction request (static prompt dominates)."""
    import json
    static = len(extraction.SYSTEM_PROMPT) + len(json.dumps(extraction.OUTPUT_SCHEMA))
    est_in = (static + 400) // 3 + 1  # + a typical redacted tweet and delimiters
    est_out = settings.LLM["MAX_OUTPUT_TOKENS"]
    cost = llm_budget.estimate_cost(provider, est_in, est_out)
    return {"input_tokens": est_in, "max_output_tokens": est_out, "cost_usd": cost}


def estimate(provider: str, n_requests: int) -> dict:
    e = per_message_estimate(provider)
    rpm = max(1, settings.LLM["RATE_LIMIT_RPM"])
    return {"requests": n_requests, "input_tokens": e["input_tokens"] * n_requests,
            "max_output_tokens": e["max_output_tokens"] * n_requests,
            "cost_usd": None if e["cost_usd"] is None else e["cost_usd"] * n_requests,
            "minimum_minutes_at_rate_limit": round(n_requests / rpm, 1)}


def create_job(*, user, selection: str, max_messages: int, message_ids: list[int] | None = None,
               max_requests: int | None = None, max_input_tokens: int | None = None,
               max_output_tokens: int | None = None, max_cost_usd: Decimal | None = None,
               client: LLMClient | None = None, stop_on_error: bool = False) -> AnalysisJob:
    if not (user and user.has_perm("feedback.run_pipeline")):
        raise PermissionDenied("Creating analysis jobs needs the run_pipeline permission.")
    if selection not in AnalysisJob.Selection.values:
        raise ActionError("Unknown selection.")
    if selection == AnalysisJob.Selection.MESSAGE_IDS and not message_ids:
        raise ActionError("Choose at least one conversation.")
    if max_messages < 1:
        raise ActionError("A job must process at least one message.")
    client = client or get_llm_client()  # validates provider configuration (no silent fallback)
    max_requests = max_requests if max_requests is not None else max_messages
    est = estimate(client.provider_name, max_requests)
    if not client.is_mock and est["cost_usd"] is None and max_cost_usd is None:
        raise ActionError("Prices for this provider/tier are not configured; a cost limit cannot be computed.")
    return AnalysisJob.objects.create(
        selection=selection, message_ids=message_ids or [], provider=client.provider_name, model_name=client.model,
        prompt_version=extraction.PROMPT_VERSION, is_mock=client.is_mock, max_messages=max_messages,
        max_requests=max_requests,
        max_input_tokens=max_input_tokens if max_input_tokens is not None else est["input_tokens"],
        max_output_tokens=max_output_tokens if max_output_tokens is not None else est["max_output_tokens"],
        max_cost_usd=max_cost_usd if max_cost_usd is not None else (est["cost_usd"] or Decimal(0)),
        stop_on_error=stop_on_error, created_by=user)


@transaction.atomic
def authorize(job: AnalysisJob, user) -> AnalysisJob:
    if not is_manager(user):
        raise PermissionDenied("Only managers can authorise analysis jobs.")
    job = AnalysisJob.objects.select_for_update().get(pk=job.pk)
    if job.status != J.DRAFT:
        raise ActionError("Only draft jobs can be authorised.")
    if not job.is_mock:
        remaining = llm_budget.remaining()
        if job.max_requests > remaining:
            raise ActionError(f"The global real-request budget has {remaining} request(s) left; this job needs up to "
                              f"{job.max_requests}. The owner must first run `manage.py llm_budget --authorize N "
                              "--confirm`.")
    job.status, job.authorized_by, job.authorized_at = J.AUTHORIZED, user, timezone.now()
    job.save()
    return job


def _set_status(job, user, allowed_from, target, manager_only=False):
    if not (user and (is_manager(user) if manager_only else user.has_perm("feedback.run_pipeline"))):
        raise PermissionDenied("You are not allowed to change this job.")
    with transaction.atomic():
        job = AnalysisJob.objects.select_for_update().get(pk=job.pk)
        if job.status not in allowed_from:
            raise ActionError(f"Cannot change a {job.status} job to {target}.")
        job.status = target
        if target in (J.CANCELLED,):
            job.finished_at = timezone.now()
            job.stop_reason = f"Cancelled by {user.username}."
        job.save()
    return job


def cancel(job, user):
    return _set_status(job, user, OPEN, J.CANCELLED)


def pause(job, user):
    return _set_status(job, user, {J.AUTHORIZED, J.RUNNING}, J.PAUSED)


def resume(job, user):
    """Paused or stale-running (crashed worker) jobs go back to the queue. Manager decision."""
    stale_before = timezone.now() - timedelta(minutes=settings.JOBS["STALE_HEARTBEAT_MINUTES"])
    j = AnalysisJob.objects.get(pk=job.pk)
    allowed = {J.PAUSED, J.STOPPED}
    if j.status == J.RUNNING and (j.heartbeat_at is None or j.heartbeat_at < stale_before):
        allowed.add(J.RUNNING)
    return _set_status(job, user, allowed, J.AUTHORIZED, manager_only=True)


def run_job(job_id: int, *, client: LLMClient | None = None, chunk: int = 25) -> AnalysisJob:
    """Process an authorised job until done, limited, cancelled or paused. Safe to call repeatedly."""
    with transaction.atomic():
        job = AnalysisJob.objects.select_for_update().get(pk=job_id)
        if job.status != J.AUTHORIZED:
            return job
        job.status = J.RUNNING
        job.started_at = job.started_at or timezone.now()
        job.heartbeat_at = timezone.now()
        job.save()
    llm_budget.close_interrupted()
    try:
        client = client or get_llm_client()
    except Exception as exc:
        return _finish(job, J.STOPPED, f"Provider not configured: {exc}")
    if (client.provider_name, client.model, client.is_mock) != (job.provider, job.model_name, job.is_mock):
        return _finish(job, J.STOPPED, "Provider/model configuration changed since the job was authorised.")
    if not client.is_mock:
        client.guard = llm_budget.RequestBudget(job=job)
    service = AnalysisService(client=client, job=job)
    while True:
        job.refresh_from_db()
        if job.status != J.RUNNING:
            return job  # paused or cancelled by a user
        left = job.max_messages - job.processed
        if left <= 0:
            return _finish(job, J.COMPLETED, "Reached max_messages.")
        run = service.run(limit=min(chunk, left), selection=job.selection, message_ids=job.message_ids or None,
                          stop_on_error=job.stop_on_error)
        job.refresh_from_db()
        if job.status in (J.CANCELLED, J.PAUSED):
            return job
        if run.stopped_reason:
            return _finish(job, J.STOPPED, run.stopped_reason)
        if run.status == run.Status.FAILED:
            return _finish(job, J.STOPPED, "Analysis run failed unexpectedly; see logs.")
        if run.requested == 0:
            return _finish(job, J.COMPLETED, "No more eligible conversations.")


def _finish(job, status, reason):
    AnalysisJob.objects.filter(pk=job.pk, status=J.RUNNING).update(
        status=status, stop_reason=reason[:300], finished_at=timezone.now())
    job.refresh_from_db()
    return job


def dev_sample_ids() -> list[int]:
    """tweet_ids of the 50-message DEVELOPMENT split (data/eval/dev_examples.csv).

    Reads only the development file; the held-out evaluation sample is never opened here.
    """
    import csv
    from pathlib import Path
    path = Path(settings.BASE_DIR) / "data" / "eval" / "dev_examples.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [int(r["tweet_id"]) for r in csv.DictReader(f)]


def unused_dev_message_ids(n: int, seed: int) -> list[int]:
    """Seeded random choice of n development-split conversations never analysed by a real LLM."""
    import random
    from feedback.models import CustomerMessage, MessageAnalysis
    used = set(MessageAnalysis.objects.filter(source="llm").values_list("message__source_message_id", flat=True))
    candidates = sorted(set(dev_sample_ids()) - used)
    chosen = random.Random(seed).sample(candidates, min(n, len(candidates)))
    return list(CustomerMessage.objects.filter(source_message_id__in=chosen, author_role="customer",
                                               is_thread_root=True).order_by("created_at").values_list("pk", flat=True))
