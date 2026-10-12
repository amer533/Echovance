import json

from django.core.management.base import BaseCommand, CommandError

from feedback.models import LLMRequestLog
from feedback.services import llm_budget


class Command(BaseCommand):
    help = "Show or change the persisted real-LLM request budget (hard ceiling 20)."

    def add_arguments(self, parser):
        parser.add_argument("--authorize", type=int, metavar="TOTAL",
                            help="Set the cumulative authorised total of real requests (<= 20)")
        parser.add_argument("--confirm", action="store_true", help="Required with --authorize")
        parser.add_argument("--clear-halt", action="store_true", help="Re-enable live calls after a halt")
        parser.add_argument("--log", action="store_true", help="List every real request")

    def handle(self, authorize, confirm, clear_halt, log, **opts):
        if authorize is not None:
            if not confirm:
                raise CommandError("Changing the budget requires --confirm (explicit human authorisation).")
            try:
                llm_budget.authorize(authorize)
            except ValueError as exc:
                raise CommandError(str(exc))
        if clear_halt:
            llm_budget.clear_halt()
        self.stdout.write(json.dumps(llm_budget.status(), indent=2))
        if log:
            for r in LLMRequestLog.objects.order_by("created_at"):
                self.stdout.write(f"{r.created_at:%Y-%m-%d %H:%M:%S} {r.provider}/{r.model_name} {r.purpose:20} "
                                  f"{r.status:9} {r.error_code or '-':18} {r.latency_ms or '-'} ms "
                                  f"tokens in/out {r.input_tokens}/{r.output_tokens}")
