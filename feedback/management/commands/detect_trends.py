from datetime import date

from django.core.management.base import BaseCommand

from feedback.services.trends import TrendService


class Command(BaseCommand):
    help = "Run deterministic trend detection for one window or backfill all historical windows."

    def add_arguments(self, parser):
        parser.add_argument("--window-end", type=date.fromisoformat, help="Exclusive end date YYYY-MM-DD (UTC)")
        parser.add_argument("--all", action="store_true", help="Evaluate every complete historical window")
        parser.add_argument("--dry-run", action="store_true", help="Evaluate and print without saving alerts")

    def handle(self, window_end, all, dry_run, **opts):
        svc = TrendService()
        if all:
            results = [svc.evaluate(e) for e in svc.window_ends()] if dry_run else svc.detect_all()
        else:
            end = window_end or svc.latest_window_end()
            if end is None:
                self.stderr.write("No analysed mentions yet.")
                return
            results = [svc.evaluate(end) if dry_run else svc.detect(end)]
        for r in results:
            self.stdout.write(f"\nWindow {r.current.label()} volume={r.current_volume} baselines={r.baseline_volumes}")
            if not r.sufficient_data:
                self.stdout.write(f"  insufficient data: {r.insufficient_reason}")
                continue
            for ev in r.evaluations:
                if ev.rule or ev.current >= 5:
                    growth = f"x{ev.growth_ratio:.2f}" if ev.growth_ratio is not None else "new"
                    flag = f"ALERT {ev.rule}" if ev.rule else "-"
                    self.stdout.write(f"  {flag:14} {ev.issue.key:48} cur={ev.current:3} base={ev.baseline_counts} {growth}"
                                      + (f"  ({'; '.join(ev.reasons)})" if ev.reasons else ""))
            if not dry_run:
                self.stdout.write(f"  created={len(r.created)} existing={len(r.existing)}")
