from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from feedback.models import ImportBatch
from feedback.services.imports import import_csv


class Command(BaseCommand):
    help = "Import a prepared CSV (twcs schema) through the same validation as web uploads."

    def add_arguments(self, parser):
        parser.add_argument("path")
        parser.add_argument("--brand", default="SpotifyCares")
        parser.add_argument("--max-mb", type=int, default=50, help="Upper bound for command imports")
        parser.add_argument("--max-rows", type=int, default=200_000)
        parser.add_argument("--data-classification", choices=["public", "internal", "confidential"],
                            default="internal", help="Privacy gate input (use 'public' only for published data)")

    def handle(self, path, brand, max_mb, max_rows, data_classification="internal", **opts):
        p = Path(path)
        if not p.is_file():
            raise CommandError(f"File not found: {p}")
        with p.open("rb") as f:
            batch = import_csv(f, p.name, brand, source=ImportBatch.Source.COMMAND,
                               max_bytes=max_mb * 1024 * 1024, max_rows=max_rows,
                               data_classification=data_classification)
        self.stdout.write(f"Import #{batch.pk}: status={batch.status} total={batch.total_rows} "
                          f"accepted={batch.accepted_count} duplicates={batch.duplicate_count} "
                          f"rejected={batch.rejected_count} customer_roots={batch.customer_root_count} "
                          f"agent_replies={batch.agent_count}")
        if batch.status == ImportBatch.Status.FAILED:
            raise CommandError(batch.failure_reason)
