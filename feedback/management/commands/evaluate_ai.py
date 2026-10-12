import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from django.core.management.base import BaseCommand

from feedback.services.evaluation import GoldFileError, evaluate, gold_from_db, load_gold, usage_summary


class Command(BaseCommand):
    help = "Compare current analyses with independent human labels; writes results JSON + error CSV."

    def add_arguments(self, parser):
        parser.add_argument("--gold-source", choices=["db", "csv"], default="db",
                            help="db = blind-labelling UI labels (default); csv = data/eval/eval_labels.csv")
        parser.add_argument("--min-annotators", type=int, default=1,
                            help="DB gold: annotators that must agree when no adjudication exists")
        parser.add_argument("--gold", default="data/eval/eval_labels.csv")
        parser.add_argument("--out", default="docs/results/evaluation_results.json")
        parser.add_argument("--errors", default="data/eval/evaluation_errors.csv",
                            help="Disagreements incl. message text (kept out of git)")
        parser.add_argument("--allow-mock", action="store_true",
                            help="Pipeline smoke test only; output is labelled NOT AI EVALUATION")

    def handle(self, gold, out, errors, allow_mock, gold_source="db", min_annotators=1, **opts):
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        agreement = None
        try:
            if gold_source == "db":
                items, agreement = gold_from_db(min_annotators)
            else:
                from feedback.services.evaluation import _drop_dev_split
                items, dropped = _drop_dev_split(load_gold(Path(gold)))
                agreement = {"dropped_dev_split_items": dropped}
        except GoldFileError as exc:
            result = {"status": "pending", "reason": f"No usable human labels: {exc}", "generated_at": stamp}
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            self.stdout.write(self.style.WARNING(result["reason"]))
            return
        if not items:
            from feedback.services import split_guard
            result = {"status": "pending", "reason": f"No human gold labels yet (source: {gold_source}).",
                      "agreement": agreement, "real_usage": usage_summary(),
                      "split_integrity": split_guard.integrity_report(split_guard.current_prompts()),
                      "generated_at": stamp}
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            self.stdout.write(self.style.WARNING(result["reason"]))
            return
        result = evaluate(items, allow_mock=allow_mock)
        result["generated_at"] = stamp
        result["gold_source"] = gold_source if gold_source == "db" else gold
        result["agreement"] = agreement
        from feedback.services import split_guard
        result["split_integrity"] = split_guard.integrity_report(split_guard.current_prompts())
        err_rows = result.pop("errors", [])
        if err_rows:
            Path(errors).parent.mkdir(parents=True, exist_ok=True)
            with open(errors, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(err_rows[0]))
                w.writeheader()
                w.writerows(err_rows)
        result["disagreements"] = len(err_rows)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.stdout.write(json.dumps({k: result.get(k) for k in (
            "status", "evaluated", "missing_analysis", "complaint_detection")}, indent=2))
        self.stdout.write(f"Full results: {out_path}; disagreements: {errors if err_rows else 'none'}")
