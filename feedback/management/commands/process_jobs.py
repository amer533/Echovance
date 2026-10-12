import time

from django.core.management.base import BaseCommand

from feedback.models import AnalysisJob
from feedback.services import jobs


class Command(BaseCommand):
    help = ("Process AUTHORISED analysis jobs (oldest first). Only manager-authorised jobs run; each stops at its "
            "own limits and the global budget. Safe to stop (Ctrl+C) and restart at any time.")

    def add_arguments(self, parser):
        parser.add_argument("--job", type=int, help="Process only this job id")
        parser.add_argument("--loop", action="store_true", help="Keep polling for authorised jobs")
        parser.add_argument("--poll-seconds", type=int, default=10)
        parser.add_argument("--chunk", type=int, default=25, help="Messages per analysis run within a job")

    def handle(self, job, loop, poll_seconds, chunk, **opts):
        while True:
            qs = AnalysisJob.objects.filter(status=AnalysisJob.Status.AUTHORIZED).order_by("created_at")
            if job:
                qs = qs.filter(pk=job)
            pending = list(qs.values_list("pk", flat=True))
            for pk in pending:
                self.stdout.write(f"Job #{pk}: starting")
                done = jobs.run_job(pk, chunk=chunk)
                self.stdout.write(f"Job #{pk}: {done.status} — processed {done.processed}/{done.max_messages}, "
                                  f"requests {done.requests_used}/{done.max_requests}, ok {done.succeeded}, "
                                  f"review {done.needs_review}, failed {done.failed}, reused {done.reused}, "
                                  f"privacy-blocked {done.privacy_blocked}. {done.stop_reason}")
            if not loop:
                if not pending:
                    self.stdout.write("No authorised jobs to process.")
                return
            time.sleep(poll_seconds)
