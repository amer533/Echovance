"""Test data builders. All data here is SYNTHETIC — no rows from the real dataset are used in tests."""
from __future__ import annotations

import csv
import io
import itertools
from datetime import datetime, timedelta, timezone

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group

from feedback.models import CustomerMessage, ImportBatch, IssueMention, MessageAnalysis
from feedback.services.grouping import get_or_create_issue

HEADER = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
_ids = itertools.count(10_000_000)


def tw_date(dt: datetime) -> str:
    return dt.strftime("%a %b %d %H:%M:%S +0000 %Y")


def csv_bytes(rows: list[dict | list], header: list[str] = HEADER) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow([r.get(h, "") for h in header] if isinstance(r, dict) else r)
    return buf.getvalue().encode("utf-8")


def row(tweet_id, author="1001", inbound="True", created=None, text="I was charged twice for premium",
        response="", in_reply=""):
    created = created or datetime(2017, 10, 10, 12, 0, tzinfo=timezone.utc)
    return {"tweet_id": str(tweet_id), "author_id": author, "inbound": inbound, "created_at": tw_date(created),
            "text": text, "response_tweet_id": response, "in_response_to_tweet_id": in_reply}


def user(username, group=None, password="pw-Test-12345"):
    User = get_user_model()
    u = User.objects.create_user(username=username, password=password)
    if group:
        u.groups.add(Group.objects.get(name=group))
    return u


def batch(data_classification="public"):
    return ImportBatch.objects.create(source="command", original_filename="synthetic.csv", file_sha256="0" * 64,
                                      file_size_bytes=1, brand_account="SpotifyCares", status="completed",
                                      data_classification=data_classification)


def message(b=None, text="My premium stopped working", created=None, role="customer", root=True):
    b = b or batch()
    return CustomerMessage.objects.create(
        source_message_id=next(_ids), import_batch=b, author_role=role, author_hash="abc",
        brand_account="SpotifyCares", created_at=created or datetime(2017, 10, 10, tzinfo=timezone.utc),
        text=text, in_response_to_id=None if root else 1, is_thread_root=root,
        analysis_status="pending" if (role == "customer" and root) else "not_applicable")


def analysed_message(b, created, issue_keys=(), text="synthetic complaint text"):
    """A root message with a current SUCCEEDED analysis and active mentions for the given issue keys."""
    m = message(b, text=text, created=created)
    a = MessageAnalysis.objects.create(message=m, provider="test", model_name="t", prompt_version="v",
                                       is_mock=False, source="llm", status="succeeded", is_current=True,
                                       message_type="complaint" if issue_keys else "question",
                                       is_complaint=bool(issue_keys))
    for key in issue_keys:
        cat, sub = key.split(".")
        IssueMention.objects.create(issue=get_or_create_issue(cat, sub), analysis=a, message=m,
                                    message_created_at=created, sentiment="negative", summary="s",
                                    evidence_text=text[:10], evidence_start=0, evidence_end=min(10, len(text)))
    m.analysis_status = "completed"
    m.save(update_fields=["analysis_status"])
    return m


def fill_window(b, start: datetime, days: int, total: int, issue_counts: dict[str, int]):
    """Create `total` analysed messages spread over [start, start+days), with given issue mention counts."""
    created = 0
    slot = timedelta(days=days) / max(total, 1)
    for key, n in issue_counts.items():
        for _ in range(n):
            analysed_message(b, start + slot * created, (key,))
            created += 1
    while created < total:
        analysed_message(b, start + slot * created, ())
        created += 1
