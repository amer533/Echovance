import io
from datetime import datetime, timedelta, timezone

import pytest

from feedback.models import CustomerMessage, ImportBatch
from feedback.services.imports import import_csv, redact_pii
from tests.factories import HEADER, csv_bytes, row

pytestmark = pytest.mark.django_db
BRAND = "SpotifyCares"


def do_import(data: bytes, name="data.csv", **kw):
    return import_csv(io.BytesIO(data), name, BRAND, **kw)


def thread_rows():
    """Customer root, brand reply, customer follow-up."""
    return [
        row(1, text="@SpotifyCares charged twice this month", response="2"),
        row(2, author=BRAND, inbound="False", text="@1001 Sorry! Please DM us.", in_reply="1", response="3"),
        row(3, text="@SpotifyCares done, sent DM", in_reply="2"),
    ]


def test_valid_import_separates_customers_agents_and_roots():
    b = do_import(csv_bytes(thread_rows()))
    assert b.status == ImportBatch.Status.COMPLETED
    assert (b.total_rows, b.accepted_count, b.duplicate_count, b.rejected_count) == (3, 3, 0, 0)
    assert b.customer_root_count == 1 and b.agent_count == 1
    root = CustomerMessage.objects.get(source_message_id=1)
    agent = CustomerMessage.objects.get(source_message_id=2)
    follow = CustomerMessage.objects.get(source_message_id=3)
    assert root.is_complaint_candidate and root.analysis_status == "pending"
    assert agent.author_role == "agent" and agent.analysis_status == "not_applicable"
    assert follow.author_role == "customer" and not follow.is_thread_root
    assert follow.analysis_status == "not_applicable"
    assert root.created_at == datetime(2017, 10, 10, 12, 0, tzinfo=timezone.utc)
    assert root.author_hash != "1001" and len(root.author_hash) == 16


def test_reimport_is_idempotent():
    data = csv_bytes(thread_rows())
    do_import(data)
    second = do_import(data)
    assert second.status == ImportBatch.Status.COMPLETED
    assert second.accepted_count == 0 and second.duplicate_count == 3
    assert CustomerMessage.objects.count() == 3


def test_duplicate_rows_inside_one_file_counted_once():
    b = do_import(csv_bytes([row(1), row(1), row(2)]))
    assert b.accepted_count == 2 and b.duplicate_count == 1
    assert CustomerMessage.objects.count() == 2


def test_partial_failure_keeps_valid_rows_and_reports_errors_without_text():
    rows = [row(1), row(2, created=None) | {"created_at": "not a date"}, row(3) | {"tweet_id": "abc"},
            row(4) | {"text": ""}, row(5) | {"inbound": "maybe"}]
    b = do_import(csv_bytes(rows))
    assert b.status == ImportBatch.Status.COMPLETED_WITH_ERRORS
    assert b.accepted_count == 1 and b.rejected_count == 4
    codes = {(e["field"], e["code"]) for e in b.errors}
    assert ("created_at", "invalid_date") in codes
    assert ("tweet_id", "invalid_integer") in codes
    assert ("text", "required") in codes
    assert ("inbound", "invalid_boolean") in codes
    assert all(set(e) == {"row", "field", "code"} for e in b.errors)  # no message content leaked


def test_malformed_and_timezone_less_dates_rejected():
    rows = [row(1) | {"created_at": "2017-10-10 12:00:00"}, row(2) | {"created_at": "2017-10-10T12:00:00Z"},
            row(3) | {"created_at": "Mon Feb 30 10:00:00 +0000 2017"}]
    b = do_import(csv_bytes(rows))
    codes = {e["row"]: e["code"] for e in b.errors}
    assert codes == {2: "missing_timezone", 4: "invalid_date"}
    assert b.accepted_count == 1  # ISO 8601 with Z is valid


def test_future_dates_rejected():
    future = datetime.now(timezone.utc) + timedelta(days=10)
    b = do_import(csv_bytes([row(1, created=future)]))
    assert b.errors[0]["code"] == "future_date"


def test_incorrect_columns_fail_whole_file():
    b = do_import(csv_bytes([["1", "x"]], header=["id", "body"]))
    assert b.status == ImportBatch.Status.FAILED
    assert "missing_columns" in b.failure_reason
    assert CustomerMessage.objects.count() == 0


@pytest.mark.parametrize("data,code", [
    (b"", "empty_file"),
    (b"   \n", "empty_file"),
    (",".join(HEADER).encode() + b"\n", "no_data_rows"),
    (b"\xff\xfe\x00bad", "invalid_encoding"),
])
def test_file_level_failures(data, code):
    b = do_import(data)
    assert b.status == ImportBatch.Status.FAILED and b.failure_reason.startswith(code)


def test_wrong_extension_rejected():
    assert do_import(csv_bytes([row(1)]), name="data.xlsx").failure_reason.startswith("invalid_extension")


def test_size_and_row_limits():
    data = csv_bytes([row(i) for i in range(1, 6)])
    assert do_import(data, max_bytes=50).failure_reason.startswith("file_too_large")
    b = do_import(data, max_rows=3)
    assert b.failure_reason.startswith("too_many_rows")
    assert CustomerMessage.objects.count() == 0  # validation happens before any insert


def test_inconsistent_brand_flags_rejected():
    rows = [row(1, author="OtherBrand", inbound="False"), row(2, author=BRAND, inbound="True")]
    b = do_import(csv_bytes(rows))
    assert {e["code"] for e in b.errors} == {"outbound_from_other_account", "brand_account_marked_inbound"}


def test_extra_cells_rejected():
    data = csv_bytes([row(1)]) + b"2,1001,True,Tue Oct 10 12:00:00 +0000 2017,hi,,,EXTRA\n"
    b = do_import(data)
    assert b.accepted_count == 1 and b.errors[0]["code"] == "too_many_fields"


def test_pii_redaction():
    text = "mail me at jane.doe@example.com or +1 (555) 123-4567, app v8.4.25.906 costs 9.99"
    out = redact_pii(text)
    assert "[email]" in out and "[phone]" in out
    assert "8.4.25.906" in out and "9.99" in out
    b = do_import(csv_bytes([row(1, text=text)]))
    assert "example.com" not in CustomerMessage.objects.get(import_batch=b).text
