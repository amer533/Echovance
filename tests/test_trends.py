from datetime import date, datetime, timedelta, timezone

import pytest

from feedback.models import IssueAlert
from feedback.services.trends import TrendService, to_utc_midnight
from tests import factories

pytestmark = pytest.mark.django_db

RULES = {"RULE_VERSION": "test-v1", "WINDOW_DAYS": 7, "BASELINE_WINDOWS": 3, "MIN_WINDOW_ANALYZED_MESSAGES": 20,
         "MIN_CURRENT_MENTIONS": 8, "MIN_ABSOLUTE_INCREASE": 5, "MIN_GROWTH_RATIO": 1.5, "MIN_SHARE_RATIO": 1.3,
         "MIN_EMERGING_MENTIONS": 6, "EVIDENCE_LIMIT": 5}
END = date(2017, 11, 1)  # current window = Oct 25 .. Oct 31 inclusive
KEY = "billing_subscription.unexpected_charge"


def utc(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


def build(current: dict, baselines: list[dict], volume=30, current_volume=None):
    """baselines: oldest first. Each dict: issue_key -> mention count."""
    b = factories.batch()
    start = utc(2017, 10, 4)
    for counts in baselines:
        factories.fill_window(b, start, 7, volume, counts)
        start += timedelta(days=7)
    factories.fill_window(b, start, 7, current_volume or volume, current)


def test_window_arithmetic_is_half_open_and_contiguous():
    svc = TrendService(RULES)
    cur, base = svc.windows(END)
    assert cur.start == utc(2017, 10, 25) and cur.end == utc(2017, 11, 1)
    assert [w.start for w in base] == [utc(2017, 10, 4), utc(2017, 10, 11), utc(2017, 10, 18)]
    assert base[-1].end == cur.start
    assert to_utc_midnight(datetime(2017, 11, 1, 23, 59, tzinfo=timezone.utc)) == utc(2017, 11, 1)
    assert cur.label() == "2017-10-25 – 2017-10-31"


def test_boundary_message_belongs_to_next_window():
    b = factories.batch()
    factories.analysed_message(b, utc(2017, 11, 1), (KEY,))           # exactly at END -> excluded
    factories.analysed_message(b, utc(2017, 10, 31) + timedelta(hours=23, minutes=59), (KEY,))
    cur, _ = TrendService(RULES).windows(END)
    assert sum(TrendService(RULES).issue_counts(cur).values()) == 1


def test_surge_alert_with_explanation_and_evidence():
    build({KEY: 14}, [{KEY: 4}, {KEY: 5}, {KEY: 3}])
    res = TrendService(RULES).detect(END)
    assert res.sufficient_data and len(res.created) == 1
    a = res.created[0]
    assert a.rule == "surge" and a.current_count == 14 and a.baseline_mean == pytest.approx(4.0)
    assert a.growth_ratio == pytest.approx(3.5) and a.baseline_total == 12
    assert a.evidence.count() == 5 and all(m.issue_id == a.issue_id for m in a.evidence.all())
    assert "not all customers" in a.explanation and "x3.50" in a.explanation
    assert a.thresholds["MIN_GROWTH_RATIO"] == 1.5 and a.status == "open"


def test_detection_is_deduplicated_on_rerun():
    build({KEY: 14}, [{KEY: 4}, {KEY: 5}, {KEY: 3}])
    svc = TrendService(RULES)
    svc.detect(END)
    again = svc.detect(END)
    assert again.created == [] and len(again.existing) == 1
    assert IssueAlert.objects.count() == 1


def test_emerging_issue_without_baseline_has_no_growth_percentage():
    build({KEY: 7}, [{}, {}, {}])
    a = TrendService(RULES).detect(END).created[0]
    assert a.rule == "emerging" and a.growth_ratio is None and a.baseline_mean == 0
    assert "no growth percentage" in a.explanation


def test_small_emerging_count_does_not_alert():
    build({KEY: 5}, [{}, {}, {}])
    res = TrendService(RULES).detect(END)
    assert res.created == [] and "no baseline mentions" in res.evaluations[0].reasons[0]


def test_below_minimum_volume_never_alerts():
    build({KEY: 7}, [{KEY: 1}, {KEY: 1}, {KEY: 1}])  # x7 growth but current < 8
    ev = TrendService(RULES).detect(END).evaluations[0]
    assert ev.rule is None and any("current 7 < 8" in r for r in ev.reasons)


def test_small_absolute_increase_does_not_alert():
    build({KEY: 13}, [{KEY: 9}, {KEY: 9}, {KEY: 9}])  # +4 only, x1.44
    ev = TrendService(RULES).detect(END).evaluations[0]
    assert ev.rule is None and any(r.startswith("increase") for r in ev.reasons)


def test_volume_growth_alone_is_not_a_surge():
    # Mentions double but so does total analysed volume: share is flat -> no alert.
    build({KEY: 16}, [{KEY: 8}, {KEY: 8}, {KEY: 8}], volume=30, current_volume=60)
    ev = TrendService(RULES).detect(END).evaluations[0]
    assert ev.rule is None and any(r.startswith("share ratio") for r in ev.reasons)


def test_insufficient_history_produces_no_alerts():
    build({KEY: 14}, [{KEY: 4}, {KEY: 5}], volume=30)  # only 2 baseline windows of data
    res = TrendService(RULES).detect(END)
    assert not res.sufficient_data and res.created == [] and "at least 20" in res.insufficient_reason


def test_detect_all_backfills_each_window_once():
    b = factories.batch()
    start = utc(2017, 10, 4)
    for counts in [{KEY: 3}, {KEY: 3}, {KEY: 3}, {KEY: 14}, {KEY: 4}]:
        factories.fill_window(b, start, 7, 30, counts)
        start += timedelta(days=7)
    svc = TrendService(RULES)
    ends = svc.window_ends()
    assert ends[0] == utc(2017, 11, 8) and ends[-1] == utc(2017, 11, 1)
    svc.detect_all()
    svc.detect_all()
    assert IssueAlert.objects.count() == 1


def test_superseded_mentions_are_not_counted():
    b = factories.batch()
    m = factories.analysed_message(b, utc(2017, 10, 26), (KEY,))
    m.mentions.update(is_active=False)
    cur, _ = TrendService(RULES).windows(END)
    assert TrendService(RULES).issue_counts(cur) == {}


def test_weekly_series_reports_counts_and_volume():
    build({KEY: 14}, [{KEY: 4}, {KEY: 5}, {KEY: 3}])
    series = TrendService(RULES).weekly_series(None, utc(2017, 10, 4), utc(2017, 11, 1))
    assert [p["count"] for p in series] == [4, 5, 3, 14] and all(p["volume"] == 30 for p in series)
