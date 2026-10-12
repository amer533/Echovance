"""Deterministic trend detection (rule version in settings.TREND_RULES["RULE_VERSION"]).

Data scope: every query runs in ONE explicit scope — "real" (sources llm + human, the default) or
"demo" (mock only). There is deliberately no scope that mixes mock and real data.

Counting unit: distinct customer thread-root messages with an active IssueMention for the issue,
bucketed by the ORIGINAL message timestamp (UTC), in half-open windows [start, end).
Denominator ("volume"): customer thread roots whose CURRENT analysis succeeded, in scope.

Rules (all thresholds in settings and stored on each alert):
  insufficient data  current or any baseline window volume < MIN_WINDOW_ANALYZED_MESSAGES -> no alerts
  SURGE     baseline_mean > 0 and current >= MIN_CURRENT_MENTIONS
            and current - baseline_mean >= MIN_ABSOLUTE_INCREASE
            and current / baseline_mean >= MIN_GROWTH_RATIO
            and (baseline_share == 0 or current_share / baseline_share >= MIN_SHARE_RATIO)
  EMERGING  baseline_total == 0 and current >= MIN_EMERGING_MENTIONS (no growth % is reported)

Alerts are immutable snapshots. Each stores an input fingerprint (hash of contributing mention ids).
When re-analysis changes those inputs the alert is marked STALE (review decision, note and linked
actions are kept) and re-detection creates a new revision linked via superseded_by.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from datetime import timezone as dt_timezone

from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import Count, Max, Min
from django.db.models.functions import TruncDate
from django.utils import timezone

from feedback.models import CustomerMessage, Issue, IssueAlert, IssueMention, MessageAnalysis

SCOPE_SOURCES = {
    "real": ("llm", "human"),
    "demo": ("mock",),
    "legacy_mixed": ("llm", "human", "mock"),  # only for re-checking pre-upgrade alerts
}
DETECTION_SCOPES = ("real", "demo")


def to_utc_midnight(d: date | datetime) -> datetime:
    if isinstance(d, datetime):
        d = d.astimezone(dt_timezone.utc).date()
    return datetime.combine(d, time.min, tzinfo=dt_timezone.utc)


def fingerprint(ids) -> str:
    return hashlib.sha256(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()


def mentions_in_scope(scope: str):
    return IssueMention.objects.filter(is_active=True, source__in=SCOPE_SOURCES[scope])


@dataclass
class Window:
    start: datetime
    end: datetime

    def label(self) -> str:
        return f"{self.start:%Y-%m-%d} – {(self.end - timedelta(days=1)):%Y-%m-%d}"


@dataclass
class IssueEvaluation:
    issue: Issue
    current: int
    baseline_counts: list[int]
    baseline_mean: float
    growth_ratio: float | None
    current_share: float
    baseline_share: float
    rule: str | None
    reasons: list[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    current: Window
    baselines: list[Window]
    current_volume: int
    baseline_volumes: list[int]
    sufficient_data: bool
    insufficient_reason: str = ""
    scope: str = "real"
    evaluations: list[IssueEvaluation] = field(default_factory=list)
    created: list[IssueAlert] = field(default_factory=list)
    existing: list[IssueAlert] = field(default_factory=list)
    staled: list[IssueAlert] = field(default_factory=list)


class TrendService:
    def __init__(self, rules: dict | None = None, scope: str = "real"):
        if scope not in DETECTION_SCOPES:
            raise ValueError("scope must be 'real' or 'demo'")
        self.rules = rules or settings.TREND_RULES
        self.scope = scope

    # ---------- windows ----------
    def windows(self, window_end: date | datetime) -> tuple[Window, list[Window]]:
        days = timedelta(days=self.rules["WINDOW_DAYS"])
        end = to_utc_midnight(window_end)
        current = Window(end - days, end)
        baselines = []
        cursor = current.start
        for _ in range(self.rules["BASELINE_WINDOWS"]):
            baselines.append(Window(cursor - days, cursor))
            cursor -= days
        baselines.reverse()  # chronological
        return current, baselines

    def data_range(self) -> tuple[datetime | None, datetime | None]:
        # Anchored on analysed messages (the denominator), not on mentions: a quiet last day still counts.
        agg = self._analysed_roots().aggregate(lo=Min("created_at"), hi=Max("created_at"))
        return agg["lo"], agg["hi"]

    def latest_window_end(self) -> datetime | None:
        _, hi = self.data_range()
        return to_utc_midnight(hi) + timedelta(days=1) if hi else None

    # ---------- counting ----------
    def _mentions(self):
        return mentions_in_scope(self.scope)

    def _analysed_roots(self):
        return CustomerMessage.objects.filter(
            author_role=CustomerMessage.AuthorRole.CUSTOMER, is_thread_root=True,
            analyses__is_current=True, analyses__status=MessageAnalysis.Status.SUCCEEDED,
            analyses__source__in=SCOPE_SOURCES[self.scope])

    def volume(self, w: Window) -> int:
        return self._analysed_roots().filter(created_at__gte=w.start, created_at__lt=w.end).count()

    def issue_counts(self, w: Window) -> dict[int, int]:
        rows = (self._mentions().filter(message_created_at__gte=w.start, message_created_at__lt=w.end)
                .values("issue_id").annotate(n=Count("message_id", distinct=True)))
        return {r["issue_id"]: r["n"] for r in rows}

    def daily_series(self, start: datetime, end: datetime, issue: Issue | None = None) -> list[dict]:
        qs = self._mentions().filter(message_created_at__gte=start, message_created_at__lt=end)
        if issue is not None:
            qs = qs.filter(issue=issue)
        rows = dict(qs.annotate(day=TruncDate("message_created_at", tzinfo=dt_timezone.utc))
                    .values("day").annotate(n=Count("message_id", distinct=True)).values_list("day", "n"))
        out, d = [], start.date()
        while d < end.date():
            out.append({"day": d, "count": rows.get(d, 0)})
            d += timedelta(days=1)
        return out

    def weekly_series(self, issue: Issue | None, start: datetime, end: datetime) -> list[dict]:
        """Consecutive WINDOW_DAYS buckets ending at `end`, with counts and analysed volume."""
        days = timedelta(days=self.rules["WINDOW_DAYS"])
        buckets, cursor = [], to_utc_midnight(end)
        while cursor - days >= to_utc_midnight(start):
            buckets.append(Window(cursor - days, cursor))
            cursor -= days
        buckets.reverse()
        out = []
        for w in buckets:
            qs = self._mentions().filter(message_created_at__gte=w.start, message_created_at__lt=w.end)
            if issue is not None:
                qs = qs.filter(issue=issue)
            out.append({"start": w.start, "end": w.end, "label": w.label(),
                        "count": qs.values("message_id").distinct().count(), "volume": self.volume(w)})
        return out

    def contributing_ids(self, issue_id: int, w: Window) -> list[int]:
        return list(self._mentions().filter(issue_id=issue_id, message_created_at__gte=w.start,
                                            message_created_at__lt=w.end).values_list("id", flat=True))

    # ---------- rules ----------
    def evaluate(self, window_end: date | datetime) -> DetectionResult:
        r = self.rules
        current, baselines = self.windows(window_end)
        cur_vol = self.volume(current)
        base_vols = [self.volume(w) for w in baselines]
        result = DetectionResult(current, baselines, cur_vol, base_vols, sufficient_data=True, scope=self.scope)
        min_vol = r["MIN_WINDOW_ANALYZED_MESSAGES"]
        if cur_vol < min_vol or any(v < min_vol for v in base_vols):
            result.sufficient_data = False
            result.insufficient_reason = (
                f"Each window needs at least {min_vol} analysed customer messages in the '{self.scope}' scope "
                f"(current {cur_vol}, baselines {base_vols}).")
            return result
        cur_counts = self.issue_counts(current)
        base_counts = [self.issue_counts(w) for w in baselines]
        issue_ids = set(cur_counts)
        for bc in base_counts:
            issue_ids |= set(bc)
        issues = Issue.objects.in_bulk(issue_ids)
        base_vol_total = sum(base_vols)
        for iid in sorted(issue_ids):
            c = cur_counts.get(iid, 0)
            bl = [bc.get(iid, 0) for bc in base_counts]
            mean = sum(bl) / len(bl)
            growth = (c / mean) if mean > 0 else None
            cur_share = c / cur_vol
            base_share = sum(bl) / base_vol_total if base_vol_total else 0.0
            ev = IssueEvaluation(issues[iid], c, bl, mean, growth, cur_share, base_share, None)
            if mean == 0:
                if c >= r["MIN_EMERGING_MENTIONS"]:
                    ev.rule = IssueAlert.Rule.EMERGING
                else:
                    ev.reasons.append(f"no baseline mentions and current {c} < {r['MIN_EMERGING_MENTIONS']}")
            else:
                checks = [
                    (c >= r["MIN_CURRENT_MENTIONS"], f"current {c} < {r['MIN_CURRENT_MENTIONS']}"),
                    (c - mean >= r["MIN_ABSOLUTE_INCREASE"], f"increase {c - mean:.1f} < {r['MIN_ABSOLUTE_INCREASE']}"),
                    (growth >= r["MIN_GROWTH_RATIO"], f"growth x{growth:.2f} < x{r['MIN_GROWTH_RATIO']}"),
                    (base_share == 0 or cur_share / base_share >= r["MIN_SHARE_RATIO"],
                     f"share ratio x{(cur_share / base_share) if base_share else 0:.2f} < x{r['MIN_SHARE_RATIO']}"),
                ]
                failed = [msg for ok, msg in checks if not ok]
                if failed:
                    ev.reasons.extend(failed)
                else:
                    ev.rule = IssueAlert.Rule.SURGE
            result.evaluations.append(ev)
        result.evaluations.sort(key=lambda e: (e.rule is None, -e.current))
        return result

    def explain(self, ev: IssueEvaluation, res: DetectionResult) -> str:
        base = ", ".join(str(b) for b in ev.baseline_counts)
        source = "real AI and human-reviewed analyses" if self.scope == "real" else "MOCK demo analyses (not AI)"
        text = (f"{ev.current} customer messages mentioned this issue in {res.current.label()} "
                f"({ev.current_share:.1%} of {res.current_volume} analysed messages). "
                f"The {len(res.baselines)} preceding {self.rules['WINDOW_DAYS']}-day windows had {base} "
                f"(mean {ev.baseline_mean:.1f}, share {ev.baseline_share:.1%}). ")
        if ev.rule == IssueAlert.Rule.SURGE:
            text += (f"Mentions are x{ev.growth_ratio:.2f} the baseline mean, an increase of "
                     f"{ev.current - ev.baseline_mean:.1f}. ")
        else:
            text += "The issue had no mentions in the baseline windows, so no growth percentage is reported. "
        text += (f"Data: {source}. Counts cover only analysed public support tweets in the dataset sample, "
                 "not all customers, and do not establish a cause.")
        return text

    def detect(self, window_end: date | datetime) -> DetectionResult:
        res = self.evaluate(window_end)
        if not res.sufficient_data:
            return res
        thresholds = {k: v for k, v in self.rules.items() if k != "RULE_VERSION"}
        by_issue = {ev.issue.pk: ev for ev in res.evaluations}
        rv = self.rules["RULE_VERSION"]
        base_lookup = dict(window_start=res.current.start, window_end=res.current.end, rule_version=rv,
                           data_scope=self.scope)
        # Existing fresh alerts in this window whose inputs changed (or that no longer fire) become stale.
        for alert in IssueAlert.objects.filter(is_stale=False, **base_lookup):
            fp = fingerprint(self.contributing_ids(alert.issue_id, res.current))
            if fp != alert.input_fingerprint:
                ev = by_issue.get(alert.issue_id)
                reason = ("inputs changed after re-analysis; superseded by a new revision" if ev and ev.rule
                          else "inputs changed after re-analysis; the rule no longer fires")
                mark_stale(alert, reason)
                res.staled.append(alert)
        for ev in res.evaluations:
            if ev.rule is None:
                continue
            fresh = IssueAlert.objects.filter(issue=ev.issue, is_stale=False, **base_lookup).first()
            if fresh:
                res.existing.append(fresh)
                continue
            previous = IssueAlert.objects.filter(issue=ev.issue, **base_lookup).order_by("-revision").first()
            ids = self.contributing_ids(ev.issue.pk, res.current)
            try:
                with transaction.atomic():
                    alert = IssueAlert.objects.create(
                        issue=ev.issue, **base_lookup, revision=(previous.revision + 1) if previous else 1,
                        input_fingerprint=fingerprint(ids), rule=ev.rule, baseline_start=res.baselines[0].start,
                        baseline_end=res.baselines[-1].end, current_count=ev.current,
                        baseline_total=sum(ev.baseline_counts), baseline_mean=ev.baseline_mean,
                        growth_ratio=ev.growth_ratio, current_share=ev.current_share,
                        baseline_share=ev.baseline_share, current_window_volume=res.current_volume,
                        baseline_window_volumes=res.baseline_volumes, thresholds=thresholds,
                        explanation=self.explain(ev, res))
                    evidence = (self._mentions().filter(issue=ev.issue, message_created_at__gte=res.current.start,
                                                        message_created_at__lt=res.current.end)
                                .order_by("-message_created_at")[: self.rules["EVIDENCE_LIMIT"]])
                    alert.evidence.set(list(evidence))
                    if previous is not None and previous.is_stale and previous.superseded_by_id is None:
                        IssueAlert.objects.filter(pk=previous.pk).update(superseded_by=alert)
                res.created.append(alert)
            except IntegrityError:  # concurrent run created it first
                res.existing.append(IssueAlert.objects.get(issue=ev.issue, is_stale=False, **base_lookup))
        return res

    def window_ends(self) -> list[datetime]:
        """Every complete window end (newest first) whose full baseline lies within the data range."""
        lo, _ = self.data_range()
        end = self.latest_window_end()
        if lo is None:
            return []
        step = timedelta(days=self.rules["WINDOW_DAYS"])
        span = step * (self.rules["BASELINE_WINDOWS"] + 1)
        ends = []
        while end - span >= to_utc_midnight(lo):
            ends.append(end)
            end -= step
        return ends

    def detect_all(self) -> list[DetectionResult]:
        """Evaluate every complete window from the latest data backwards (historical backfill)."""
        return [self.detect(end) for end in self.window_ends()]


def mark_stale(alert: IssueAlert, reason: str) -> None:
    """Flag only. Counts, evidence, review decision, note and linked actions are preserved."""
    IssueAlert.objects.filter(pk=alert.pk, is_stale=False).update(
        is_stale=True, stale_reason=reason[:300], stale_at=timezone.now())
    alert.refresh_from_db()


def refresh_alert_staleness() -> int:
    """Re-check every fresh alert's inputs against current active mentions; mark changed ones stale."""
    n = 0
    for alert in IssueAlert.objects.filter(is_stale=False):
        if not alert.input_fingerprint:
            continue
        ids = mentions_in_scope(alert.data_scope).filter(
            issue_id=alert.issue_id, message_created_at__gte=alert.window_start,
            message_created_at__lt=alert.window_end).values_list("id", flat=True)
        if fingerprint(ids) != alert.input_fingerprint:
            mark_stale(alert, "contributing analyses changed after this alert was created (re-analysis)")
            n += 1
    return n
