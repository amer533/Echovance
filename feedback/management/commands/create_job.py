import json
import random

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from feedback.ai.client import LLMConfigError
from feedback.models import CustomerMessage
from feedback.services import jobs
from feedback.services.actions import ActionError
from feedback.services.evaluation import eval_sample_ids


class Command(BaseCommand):
    help = ("Create a DRAFT analysis job (sends nothing). A manager must authorise it in the UI/API before "
            "`process_jobs` runs it.")

    def add_arguments(self, parser):
        parser.add_argument("--user", required=True, help="Username of the creator (needs run_pipeline)")
        parser.add_argument("--selection", choices=["pending", "promote_mock", "retry_failed", "message_ids"],
                            default="message_ids")
        parser.add_argument("--max-messages", type=int, default=20)
        parser.add_argument("--eval-sample", type=int, metavar="N",
                            help="Use N messages from the held-out evaluation sample (seeded random)")
        parser.add_argument("--dev-sample", type=int, metavar="N",
                            help="Use N previously unused messages from the 50-message DEVELOPMENT split")
        parser.add_argument("--max-requests", type=int, help="Hard request limit for this job (default: max messages)")
        parser.add_argument("--stop-on-error", action="store_true",
                            help="Stop the whole job at the first failed / privacy-blocked / provider-error message")
        parser.add_argument("--seed", type=int, default=20261011)

    def handle(self, user, selection, max_messages, eval_sample, seed, dev_sample=None, max_requests=None,
               stop_on_error=False, **opts):
        creator = get_user_model().objects.filter(username=user).first()
        if creator is None:
            raise CommandError(f"Unknown user '{user}'.")
        ids = None
        if eval_sample and dev_sample:
            raise CommandError("Choose either --eval-sample or --dev-sample.")
        if dev_sample:
            ids = jobs.unused_dev_message_ids(dev_sample, seed)
            if len(ids) < dev_sample:
                raise CommandError(f"Only {len(ids)} unused development-split messages are available.")
            selection, max_messages = "message_ids", len(ids)
        elif eval_sample:
            tweet_ids = eval_sample_ids()
            if not tweet_ids:
                raise CommandError("data/eval/eval_sample.csv not found.")
            chosen = random.Random(seed).sample(tweet_ids, min(eval_sample, len(tweet_ids)))
            ids = list(CustomerMessage.objects.filter(source_message_id__in=chosen).values_list("pk", flat=True))
            selection, max_messages = "message_ids", len(ids)
        try:
            job = jobs.create_job(user=creator, selection=selection, max_messages=max_messages, message_ids=ids,
                                  max_requests=max_requests, stop_on_error=stop_on_error)
        except (LLMConfigError, ActionError) as exc:
            raise CommandError(str(exc))
        est = jobs.estimate(job.provider, job.max_requests)
        est["cost_usd"] = None if est["cost_usd"] is None else str(est["cost_usd"])
        self.stdout.write(f"Draft job #{job.pk} created ({job.provider}/{job.model_name}, {job.selection}, "
                          f"max {job.max_messages} messages, max {job.max_requests} requests, stop_on_error={job.stop_on_error}). "
                          "Nothing was sent.")
        if ids:
            self.stdout.write(f"Message ids: {ids}")
        self.stdout.write("Admission estimate (conservative): " + json.dumps(est))
        self.stdout.write(f"Next: a manager authorises it at /jobs/{job.pk}/ ; then run `manage.py process_jobs`.")
