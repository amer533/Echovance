import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from feedback.services.snapshots import SnapshotError, import_llm_snapshot


class Command(BaseCommand):
    help = ("Load a snapshot of genuine real-LLM analyses (no API request, no budget use). "
            "Import the matching messages first.")

    def add_arguments(self, parser):
        parser.add_argument("path", nargs="?", default="data/spotifycares_public/gemini_snapshot.json")

    def handle(self, path, **opts):
        p = Path(path)
        if not p.exists():
            raise CommandError(f"File not found: {p}")
        snapshot = json.loads(p.read_text(encoding="utf-8"))
        try:
            r = import_llm_snapshot(snapshot)
        except SnapshotError as exc:
            raise CommandError(str(exc))
        self.stdout.write(f"Snapshot: {snapshot['provider']}/{snapshot['model_name']} {snapshot['prompt_version']} "
                          f"({len(snapshot['analyses'])} genuine analyses; no API request made)")
        self.stdout.write(f"imported={r.imported} already_present={r.already_present} "
                          f"missing_messages={len(r.missing_messages)} rejected={len(r.rejected)}"
                          + (f" run=#{r.run.pk}" if r.run else ""))
        for tweet_id, reason in r.rejected:
            self.stderr.write(f"  rejected tweet {tweet_id}: {reason}")
        if r.missing_messages:
            self.stderr.write(f"  messages not in this database (import the dataset first): {r.missing_messages}")
        if r.rejected or r.missing_messages:
            raise CommandError("Snapshot only partially imported.")
