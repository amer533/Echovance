from django.core.management.base import BaseCommand, CommandError

from feedback.ai.client import LLMConfigError
from feedback.services import llm_budget
from feedback.services.analysis import AnalysisService


class Command(BaseCommand):
    help = ("Analyse pending customer thread roots (idempotent). With a real provider this requires "
            "--allow-real and never exceeds the remaining persisted request budget.")

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=100)
        parser.add_argument("--workers", type=int, default=1, help="Concurrent calls (mock only; max 16)")
        parser.add_argument("--batch-size", type=int, default=200, help="Messages per AnalysisRun")
        parser.add_argument("--no-retry-failed", action="store_true")
        parser.add_argument("--allow-real", action="store_true",
                            help="Explicit authorisation to send messages to the configured real LLM provider")

    def handle(self, limit, workers, batch_size, no_retry_failed, allow_real, **opts):
        try:
            service = AnalysisService()
        except LLMConfigError as exc:
            raise CommandError(exc.detail)
        client = service.client
        if client.is_mock:
            self.stdout.write("Provider: MOCK (deterministic keyword test double, not real AI)")
        else:
            if not allow_real:
                raise CommandError(f"LLM_PROVIDER={client.provider_name} is a real provider. Bulk real analysis is "
                                   "disabled; pass --allow-real only with explicit authorisation.")
            s = llm_budget.status()
            self.stdout.write(f"Provider: {client.provider_name}/{client.model} | budget used {s['used']}/"
                              f"{min(s['authorized'], s['hard_limit'])}, remaining {llm_budget.remaining()}"
                              + (f" | HALTED: {s['halt_reason']}" if s["halted"] else ""))
            limit = min(limit, llm_budget.remaining())
            workers = 1
        remaining = limit
        while remaining > 0:
            run = service.run(limit=min(batch_size, remaining), retry_failed=not no_retry_failed,
                              workers=max(1, min(workers, 16)))
            self.stdout.write(f"Run #{run.pk}: requested={run.requested} succeeded={run.succeeded} "
                              f"needs_review={run.needs_review} failed={run.failed} status={run.status}"
                              + (f" | {run.stopped_reason}" if run.stopped_reason else ""))
            if run.requested == 0 or run.status == "failed" or run.stopped_reason:
                break
            remaining -= run.requested
        if not client.is_mock:
            self.stdout.write(f"Remaining real requests: {llm_budget.remaining()}")
