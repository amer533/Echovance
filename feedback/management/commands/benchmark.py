"""Reproducible performance measurements -> docs/results/performance_results.json.

Everything that writes data runs inside a transaction that is rolled back at the end, using tweet
ids shifted by a large offset, so the benchmark never changes the application database.

Measured:
  * environment (OS, CPU, RAM, Python/Django/PostgreSQL versions)
  * CSV import: wall time, rows/s, Python peak allocation (tracemalloc), RSS delta; re-import (duplicates)
  * local analysis overhead per message with the MOCK provider (prompt build, JSON parse, validation,
    persistence) - i.e. everything except external network/model time
  * trend evaluation time per window
  * in-process HTTP latency of main pages/API endpoints (Django test client; excludes network)
  * external LLM latency/tokens ONLY from stored real (non-mock) analyses; otherwise "not measured"
"""
from __future__ import annotations

import csv
import io
import json
import os
import platform
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import django
import psutil
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction
from django.test import Client

from feedback.ai.client import LLMClient, MockProvider
from feedback.models import AnalysisRun, CustomerMessage, Issue, MessageAnalysis
from feedback.services.analysis import AnalysisService
from feedback.services.imports import import_csv
from feedback.services.trends import TrendService

OFFSET = 9 * 10**17


class Rollback(Exception):
    pass


def summarise(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0}
    out = {"n": len(ms), "mean_ms": round(statistics.fmean(ms), 2), "min_ms": round(min(ms), 2),
           "max_ms": round(max(ms), 2)}
    if len(ms) >= 20:
        q = statistics.quantiles(ms, n=100, method="inclusive")
        out.update(p50_ms=round(q[49], 2), p95_ms=round(q[94], 2))
    else:
        out["note"] = "fewer than 20 samples: percentiles not reported"
    return out


def shifted_csv(path: Path) -> bytes:
    """Copy of the subset with ids shifted so the rows are new to the database."""
    src = io.StringIO(path.read_text(encoding="utf-8"), newline="")
    reader = csv.DictReader(src)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=reader.fieldnames)
    w.writeheader()
    shift = lambda v: str(int(v) + OFFSET) if v.strip().isdigit() else v  # noqa: E731
    for r in reader:
        r["tweet_id"] = shift(r["tweet_id"])
        r["in_response_to_tweet_id"] = shift(r["in_response_to_tweet_id"])
        # response_tweet_id is left as-is: shifted 18-digit lists can exceed the 255-char field limit,
        # and the pipeline does not use it.
        w.writerow(r)
    return out.getvalue().encode("utf-8")


class Command(BaseCommand):
    help = "Measure import, local analysis, trend and HTTP performance (rolled back), plus stored real-LLM latency."

    def add_arguments(self, parser):
        parser.add_argument("--subset", default="data/processed/spotifycares_subset.csv")
        parser.add_argument("--analyse", type=int, default=500, help="Messages for local (mock) analysis timing")
        parser.add_argument("--http-repeats", type=int, default=30)
        parser.add_argument("--out", default="docs/results/performance_results.json")

    def handle(self, subset, analyse, http_repeats, out, **opts):
        path = Path(subset)
        if not path.exists():
            raise CommandError(f"{path} not found; run scripts/prepare_dataset.py first")
        proc = psutil.Process(os.getpid())
        result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "environment": self.environment(), "notes": []}
        result["external_llm"] = self.stored_llm_latency()
        data = shifted_csv(path)
        try:
            with transaction.atomic():
                result["import"] = self.bench_import(data, path.name, proc)
                result["local_analysis"] = self.bench_analysis(analyse)
                result["trend_detection"] = self.bench_trends()
                result["http"] = self.bench_http(http_repeats)
                raise Rollback
        except Rollback:
            pass
        result["notes"].append("All benchmark writes were rolled back; the database is unchanged.")
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        self.stdout.write(json.dumps(result, indent=2, default=str))

    # ---------------------------------------------------------------------------------------------
    def environment(self) -> dict:
        with connection.cursor() as c:
            c.execute("SELECT version()" if connection.vendor == "postgresql" else "SELECT sqlite_version()")
            db_version = c.fetchone()[0]
        vm = psutil.virtual_memory()
        return {"os": platform.platform(), "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
                "ram_gb": round(vm.total / 2**30, 1), "python": platform.python_version(),
                "django": django.get_version(), "database_vendor": connection.vendor, "database_version": db_version,
                "existing_rows": {"messages": CustomerMessage.objects.count(),
                                  "analyses": MessageAnalysis.objects.count()}}

    def stored_llm_latency(self) -> dict:
        # Provider-only time comes from the request log (MessageAnalysis.latency_ms includes rate-limit waits).
        from feedback.models import LLMRequestLog
        logs = LLMRequestLog.objects.exclude(latency_ms=None)
        if not logs.exists():
            return {"status": "not measured", "reason": "No requests to a real LLM provider are logged."}
        rows = [(lg.latency_ms, 1, lg.input_tokens, lg.output_tokens, lg.status, lg.provider, lg.model_name)
                for lg in logs]
        waits = [lg.queue_wait_ms for lg in logs if lg.queue_wait_ms is not None]
        ok = [r for r in rows if r[4] != "failed"]
        runs = AnalysisRun.objects.filter(is_mock=False, finished_at__isnull=False, requested__gt=0)
        throughput = [r.requested / max((r.finished_at - r.started_at).total_seconds(), 1e-3) for r in runs]
        return {
            "status": "measured", "models": sorted({f"{r[5]}/{r[6]}" for r in rows}),
            "provider_call_latency": summarise([float(r[0]) for r in ok]),
            "single_attempt_latency": summarise([float(r[0]) for r in ok if r[1] == 1]),
            "retried_calls": sum(1 for r in rows if r[1] and r[1] > 1),
            "failed_analyses": MessageAnalysis.objects.filter(is_mock=False, status="failed").count(),
            "input_tokens_mean": round(statistics.fmean([r[2] for r in ok if r[2]]), 1) if any(r[2] for r in ok) else None,
            "output_tokens_mean": round(statistics.fmean([r[3] for r in ok if r[3]]), 1) if any(r[3] for r in ok) else None,
            "run_throughput_messages_per_s": ({"n": len(throughput), "mean": round(statistics.fmean(throughput), 3),
                                               "min": round(min(throughput), 3), "max": round(max(throughput), 3),
                                               "note": "per AnalysisRun; includes DB persistence"}
                                              if throughput else {"n": 0}),
            "queue_wait": summarise([float(w) for w in waits]) if waits else {"n": 0},
            "note": "Provider call time (network + model) from LLMRequestLog; queue_wait is rate-limiter/admission "
                    "waiting before sending (recorded since the latency fix).",
        }

    def bench_import(self, data: bytes, name: str, proc) -> dict:
        rss0 = proc.memory_info().rss
        tracemalloc.start()
        t0 = time.perf_counter()
        batch = import_csv(io.BytesIO(data), name, "SpotifyCares", source="command", max_bytes=len(data) + 1,
                           max_rows=10**6)
        secs = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss1 = proc.memory_info().rss
        t1 = time.perf_counter()
        again = import_csv(io.BytesIO(data), name, "SpotifyCares", source="command", max_bytes=len(data) + 1,
                           max_rows=10**6)
        secs2 = time.perf_counter() - t1
        return {"rows": batch.total_rows, "file_bytes": len(data), "accepted": batch.accepted_count,
                "seconds": round(secs, 3), "rows_per_second": round(batch.total_rows / secs, 1),
                "python_peak_alloc_mb": round(peak / 2**20, 2), "rss_delta_mb": round((rss1 - rss0) / 2**20, 2),
                "rss_after_mb": round(rss1 / 2**20, 1),
                "reimport": {"duplicates": again.duplicate_count, "accepted": again.accepted_count,
                             "seconds": round(secs2, 3)}}

    def bench_analysis(self, n: int) -> dict:
        svc = AnalysisService(LLMClient(MockProvider(model="benchmark-mock"), max_retries=0))
        msgs = list(CustomerMessage.objects.filter(source_message_id__gte=OFFSET, author_role="customer",
                                                   is_thread_root=True).order_by("created_at")[:n])
        run = AnalysisRun.objects.create(provider="mock", model_name="benchmark-mock", prompt_version="bench",
                                         is_mock=True)
        timings = []
        for m in msgs:
            t0 = time.perf_counter()
            svc._persist(run, svc._call(m))
            timings.append((time.perf_counter() - t0) * 1000)
        return {"provider": "mock (no network)", "per_message": summarise(timings),
                "meaning": "Local processing per message excluding external LLM time: prompt construction, "
                           "JSON parsing, schema/evidence validation, grouping and DB persistence."}

    def bench_trends(self) -> dict:
        svc = TrendService()
        ends = svc.window_ends()
        timings = []
        for e in ends:
            t0 = time.perf_counter()
            svc.evaluate(e)
            timings.append((time.perf_counter() - t0) * 1000)
        return {"windows": len(ends), "per_window": summarise(timings)}

    def bench_http(self, repeats: int) -> dict:
        User = get_user_model()
        user = User.objects.create_user("benchmark-user", password=None)
        user.groups.add(Group.objects.get(name="Managers"))
        client = Client(HTTP_HOST="localhost")
        client.force_login(user)
        issue = Issue.objects.order_by("pk").first()
        msg = CustomerMessage.objects.filter(analyses__is_current=True).first()
        targets = {"GET / (dashboard page)": "/", "GET /messages/ (page)": "/messages/",
                   "GET /issues/ (page)": "/issues/", "GET /api/dashboard/summary/": "/api/dashboard/summary/",
                   "GET /api/messages/": "/api/messages/", "GET /api/issues/": "/api/issues/",
                   "GET /api/alerts/": "/api/alerts/", "GET /api/actions/": "/api/actions/"}
        if issue:
            targets["GET /api/issues/{id}/"] = f"/api/issues/{issue.pk}/"
            targets["GET /issues/{id}/ (page)"] = f"/issues/{issue.pk}/"
        if msg:
            targets["GET /api/messages/{id}/"] = f"/api/messages/{msg.pk}/"
        out = {}
        for label, url in targets.items():
            client.get(url)  # warm-up
            ms = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                r = client.get(url)
                ms.append((time.perf_counter() - t0) * 1000)
                if r.status_code != 200:
                    raise CommandError(f"{url} returned {r.status_code}")
            out[label] = summarise(ms)
        return {"method": "Django test client, in-process (excludes network and web server)", "endpoints": out}
