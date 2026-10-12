"""Read-side queries shared by the dashboard and the API. Everything is computed from stored rows.

Every metric is computed in ONE data scope ("real" = llm + human analyses, default; "demo" = mock).
Mock and real results are never aggregated into a single number.
"""
from __future__ import annotations

from datetime import timedelta

from django.db.models import Count, Exists, Max, Min, OuterRef, Q

from feedback import taxonomy
from feedback.models import (ActionItem, AnalysisJob, AnalysisRun, CustomerMessage, GroupingSuggestion, ImportBatch,
                             Issue, IssueAlert, MessageAnalysis)

from .trends import SCOPE_SOURCES, DETECTION_SCOPES, TrendService

SCOPES = DETECTION_SCOPES


def clean_scope(value: str | None) -> str:
    return value if value in SCOPES else "real"


def _active(scope: str) -> Q:
    return Q(mentions__is_active=True, mentions__source__in=SCOPE_SOURCES[scope])


def issues_with_counts(scope: str = "real", include_empty: bool = False):
    act = _active(scope)
    qs = (Issue.objects.annotate(
        mention_count=Count("mentions__message", filter=act, distinct=True),
        negative_count=Count("mentions__message", filter=act & Q(mentions__sentiment="negative"), distinct=True),
        first_seen=Min("mentions__message_created_at", filter=act),
        last_seen=Max("mentions__message_created_at", filter=act),
        open_alerts=Count("alerts", filter=Q(alerts__status=IssueAlert.Status.OPEN, alerts__is_stale=False,
                                             alerts__data_scope=scope), distinct=True),
    ))
    return qs if include_empty else qs.filter(mention_count__gt=0)


def provenance() -> dict:
    """Each customer conversation in exactly one bucket, so mock data can never pass as real."""
    roots = CustomerMessage.objects.filter(author_role="customer", is_thread_root=True)
    cur = MessageAnalysis.objects.filter(message=OuterRef("pk"), is_current=True)
    real_failed = MessageAnalysis.objects.filter(message=OuterRef("pk"), source="llm", status="failed").exclude(
        error_code="privacy_blocked")
    blocked = MessageAnalysis.objects.filter(message=OuterRef("pk"), status="failed", error_code="privacy_blocked")
    has_real_current = Exists(cur.filter(source="llm"))
    has_human_current = Exists(cur.filter(source="human"))
    has_mock_current = Exists(cur.filter(source="mock"))
    real_analysed = roots.filter(has_real_current).count()
    human = roots.filter(has_human_current).count()
    no_real = roots.exclude(has_real_current).exclude(has_human_current)
    mock_only = no_real.filter(has_mock_current).count()
    others = no_real.exclude(has_mock_current)
    privacy_blocked = others.filter(Exists(blocked)).count()
    failed = others.exclude(Exists(blocked)).filter(Exists(real_failed)).count()
    total = roots.count()
    return {"total": total, "real_analysed": real_analysed, "human_corrected": human, "mock_only": mock_only,
            "real_failed": failed, "privacy_blocked": privacy_blocked,
            "unprocessed": total - real_analysed - human - mock_only - failed - privacy_blocked}


def overview(scope: str = "real") -> dict:
    scope = clean_scope(scope)
    sources = SCOPE_SOURCES[scope]
    roots = CustomerMessage.objects.filter(author_role="customer", is_thread_root=True)
    current = MessageAnalysis.objects.filter(is_current=True, source__in=sources)
    type_counts = dict(current.filter(status=MessageAnalysis.Status.SUCCEEDED)
                       .values_list("message_type").annotate(n=Count("id")))
    status_counts = dict(current.values_list("status").annotate(n=Count("id")))
    action_counts = dict(ActionItem.objects.values_list("status").annotate(n=Count("id")))
    root_status_counts = dict(roots.values_list("analysis_status").annotate(n=Count("id")))
    totals = CustomerMessage.objects.aggregate(
        messages=Count("id"), agent=Count("id", filter=Q(author_role="agent")),
        roots=Count("id", filter=Q(author_role="customer", is_thread_root=True)),
        followups=Count("id", filter=Q(author_role="customer", is_thread_root=False)))
    issues = issues_with_counts(scope)
    categories = {}
    for row in issues.values("category", "mention_count"):
        categories[row["category"]] = categories.get(row["category"], 0) + row["mention_count"]
    category_rows = sorted(
        ({"code": c, "label": taxonomy.CATEGORIES.get(c, {}).get("label", c), "count": n}
         for c, n in categories.items()), key=lambda r: -r["count"])
    rng = roots.aggregate(lo=Min("created_at"), hi=Max("created_at"))
    fresh_alerts = IssueAlert.objects.filter(data_scope=scope, is_stale=False)
    return {
        "scope": scope,
        "provenance": provenance(),
        "messages_total": totals["messages"],
        "customer_roots": totals["roots"],
        "agent_replies": totals["agent"],
        "customer_followups": totals["followups"],
        "analysed_in_scope": status_counts.get("succeeded", 0) + status_counts.get("needs_review", 0),
        "needs_review_in_scope": status_counts.get("needs_review", 0),
        "message_types": type_counts,
        "complaints": type_counts.get("complaint", 0),
        "issues_with_mentions": issues.count(),
        "categories": category_rows,
        "top_issues": list(issues.order_by("-mention_count")[:8]),
        "alerts_open": fresh_alerts.filter(status=IssueAlert.Status.OPEN).count(),
        "alerts_total": fresh_alerts.count(),
        "alerts_stale": IssueAlert.objects.filter(is_stale=True).count(),
        "alerts_legacy": IssueAlert.objects.filter(data_scope=IssueAlert.Scope.LEGACY_MIXED).count(),
        "actions": {s: action_counts.get(s, 0) for s in ActionItem.Status.values},
        "imports": ImportBatch.objects.count(),
        "last_run": AnalysisRun.objects.first(),
        "jobs_open": AnalysisJob.objects.filter(status__in=["draft", "authorized", "running", "paused"]).count(),
        "grouping_suggestions_pending": GroupingSuggestion.objects.filter(status="pending").count(),
        "date_range": (rng["lo"], rng["hi"]),
        # kept for backwards compatibility with the original dashboard/API fields
        "current_analyses_mock": MessageAnalysis.objects.filter(is_current=True, source="mock").count(),
        "current_analyses_real": MessageAnalysis.objects.filter(is_current=True, source__in=("llm", "human")).count(),
        "analysis_status": {s: root_status_counts.get(s, 0)
                            for s in CustomerMessage.AnalysisStatus.values if s != "not_applicable"},
    }


def overall_weekly_trend(weeks: int = 8, scope: str = "real") -> list[dict]:
    svc = TrendService(scope=clean_scope(scope))
    end = svc.latest_window_end()
    if end is None:
        return []
    start = end - timedelta(days=svc.rules["WINDOW_DAYS"] * weeks)
    return svc.weekly_series(None, start, end)
