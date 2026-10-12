"""CSV ingestion: validation, PII redaction, duplicate detection and batch tracking.

Two-phase design: every row is validated first (bounded by CSV_MAX_ROWS), then valid rows are
inserted in chunks. Re-importing the same file is safe: rows whose tweet_id already exists are
counted as duplicates and never inserted twice (DB unique constraint is the final guard).
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from feedback.models import CustomerMessage, ImportBatch

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("tweet_id", "author_id", "inbound", "created_at", "text")
OPTIONAL_COLUMNS = ("response_tweet_id", "in_response_to_tweet_id")
TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
MAX_TEXT_CHARS = 2000
MAX_STORED_ERRORS = 200
INSERT_CHUNK = 1000
MAX_ID = 2**63 - 1

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ImportFileError(Exception):
    """File-level problem: the whole import is rejected."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ParsedRow:
    source_message_id: int
    author_role: str
    author_hash: str
    created_at: datetime
    text: str
    in_response_to_id: int | None
    response_ids: str


def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    # Only sequences with >= 9 digits are treated as phone/card numbers (keeps app versions, prices).
    return _PHONE_RE.sub(lambda m: "[phone]" if sum(c.isdigit() for c in m.group()) >= 9 else m.group(), text)


def clean_text(text: str) -> str:
    return redact_pii(_CONTROL_RE.sub("", text)).strip()


def hash_author(author_id: str) -> str:
    return hashlib.sha256(author_id.encode("utf-8")).hexdigest()[:16]


def parse_datetime(value: str) -> datetime:
    value = value.strip()
    try:
        dt = datetime.strptime(value, TWITTER_DATE_FORMAT)
    except ValueError:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))  # may raise ValueError
    if dt.tzinfo is None:
        raise ValueError("missing_timezone")
    return dt.astimezone(dt_timezone.utc)


def _parse_id(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if not value.isdigit() or int(value) > MAX_ID:
        raise ValueError
    return int(value)


def validate_row(raw: dict, brand_account: str, now: datetime) -> tuple[ParsedRow | None, list[tuple[str, str]]]:
    """Return (parsed_row, errors). Errors are (field, code) pairs and never include row content."""
    errors: list[tuple[str, str]] = []
    get = lambda k: (raw.get(k) or "").strip()  # noqa: E731

    try:
        tweet_id = _parse_id(get("tweet_id"))
        if tweet_id is None:
            errors.append(("tweet_id", "required"))
    except ValueError:
        tweet_id = None
        errors.append(("tweet_id", "invalid_integer"))

    author = get("author_id")
    if not author:
        errors.append(("author_id", "required"))
    elif len(author) > 64:
        errors.append(("author_id", "too_long"))

    inbound_raw = get("inbound").lower()
    if inbound_raw not in {"true", "false"}:
        errors.append(("inbound", "invalid_boolean"))
    inbound = inbound_raw == "true"

    created_at = None
    if not get("created_at"):
        errors.append(("created_at", "required"))
    else:
        try:
            created_at = parse_datetime(get("created_at"))
            if created_at > now + timedelta(days=1):
                errors.append(("created_at", "future_date"))
        except ValueError as exc:
            errors.append(("created_at", "missing_timezone" if str(exc) == "missing_timezone" else "invalid_date"))

    text = clean_text(raw.get("text") or "")
    if not text:
        errors.append(("text", "required"))
    elif len(text) > MAX_TEXT_CHARS:
        errors.append(("text", "too_long"))

    try:
        in_response_to = _parse_id(get("in_response_to_tweet_id"))
    except ValueError:
        in_response_to = None
        errors.append(("in_response_to_tweet_id", "invalid_integer"))

    response_ids = get("response_tweet_id")
    if response_ids and (len(response_ids) > 255 or not all(p.strip().isdigit() for p in response_ids.split(","))):
        errors.append(("response_tweet_id", "invalid_id_list"))

    # Customer vs agent separation uses the inbound flag AND the brand account, never text matching.
    role = None
    if author and inbound_raw in {"true", "false"}:
        if not inbound and author == brand_account:
            role = CustomerMessage.AuthorRole.AGENT
        elif not inbound:
            errors.append(("author_id", "outbound_from_other_account"))
        elif author == brand_account:
            errors.append(("inbound", "brand_account_marked_inbound"))
        else:
            role = CustomerMessage.AuthorRole.CUSTOMER

    if errors:
        return None, errors
    return ParsedRow(
        source_message_id=tweet_id, author_role=role, author_hash=hash_author(author),
        created_at=created_at, text=text, in_response_to_id=in_response_to, response_ids=response_ids,
    ), []


def _read_limited(fileobj, max_bytes: int) -> bytes:
    data = fileobj.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ImportFileError("file_too_large", f"File exceeds the {max_bytes} byte limit.")
    if not data.strip():
        raise ImportFileError("empty_file", "The file is empty.")
    return data


def import_csv(fileobj, filename: str, brand_account: str, *, user=None,
               source: str = ImportBatch.Source.UPLOAD, max_bytes: int | None = None,
               max_rows: int | None = None,
               data_classification: str = ImportBatch.DataClass.INTERNAL) -> ImportBatch:
    """Import a CSV file. Always returns the ImportBatch; status FAILED means nothing was imported."""
    max_bytes = max_bytes or settings.CSV_MAX_UPLOAD_BYTES
    max_rows = max_rows or settings.CSV_MAX_ROWS
    brand_account = brand_account.strip()
    filename = (filename or "upload.csv")[:255]
    batch = ImportBatch(source=source, original_filename=filename, file_sha256="", file_size_bytes=0,
                        brand_account=brand_account, created_by=user,
                        data_classification=data_classification)
    try:
        if data_classification not in ImportBatch.DataClass.values:
            raise ImportFileError("invalid_data_classification", "Choose public, internal or confidential.")
        if not brand_account or len(brand_account) > 64:
            raise ImportFileError("invalid_brand_account", "A brand support account handle is required.")
        if not filename.lower().endswith(".csv"):
            raise ImportFileError("invalid_extension", "Only .csv files are accepted.")
        data = _read_limited(fileobj, max_bytes)
        batch.file_sha256 = hashlib.sha256(data).hexdigest()
        batch.file_size_bytes = len(data)
        try:
            content = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ImportFileError("invalid_encoding", "File must be UTF-8 encoded.") from None
        batch.save()
        _import_rows(batch, content, max_rows)
    except ImportFileError as exc:
        batch.status = ImportBatch.Status.FAILED
        batch.failure_reason = f"{exc.code}: {exc.message}"
        batch.finished_at = timezone.now()
        batch.save()
    except Exception:
        logger.exception("Unexpected import failure for batch %s", batch.pk)
        batch.status = ImportBatch.Status.FAILED
        batch.failure_reason = "internal_error: the import stopped unexpectedly; rows already committed are kept and a re-import is safe."
        batch.finished_at = timezone.now()
        batch.save()
    return batch


def _import_rows(batch: ImportBatch, content: str, max_rows: int) -> None:
    try:
        reader = csv.DictReader(io.StringIO(content, newline=""))
        header = [h.strip() for h in (reader.fieldnames or [])]
    except csv.Error:
        raise ImportFileError("malformed_csv", "The CSV header could not be parsed.") from None
    reader.fieldnames = header
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ImportFileError("missing_columns", f"Missing required columns: {', '.join(missing)}.")

    now = timezone.now()
    valid: list[ParsedRow] = []
    errors: list[dict] = []
    rejected = 0
    total = 0
    seen: set[int] = set()
    in_file_duplicates = 0
    try:
        for line_no, raw in enumerate(reader, start=2):
            total += 1
            if total > max_rows:
                raise ImportFileError("too_many_rows", f"File has more than {max_rows} data rows.")
            if None in raw:  # more cells than header columns
                parsed, row_errors = None, [("row", "too_many_fields")]
            else:
                parsed, row_errors = validate_row(raw, batch.brand_account, now)
            if row_errors:
                rejected += 1
                for field, code in row_errors:
                    if len(errors) < MAX_STORED_ERRORS:
                        errors.append({"row": line_no, "field": field, "code": code})
                continue
            if parsed.source_message_id in seen:
                in_file_duplicates += 1
                continue
            seen.add(parsed.source_message_id)
            valid.append(parsed)
    except csv.Error:
        raise ImportFileError("malformed_csv", f"CSV parsing failed near data row {total + 1}.") from None
    if total == 0:
        raise ImportFileError("no_data_rows", "The file has a header but no data rows.")

    batch.total_rows = total
    batch.rejected_count = rejected
    batch.errors = errors
    batch.save(update_fields=["total_rows", "rejected_count", "errors"])

    for start in range(0, len(valid), INSERT_CHUNK):
        chunk = valid[start:start + INSERT_CHUNK]
        ids = [r.source_message_id for r in chunk]
        existing = set(CustomerMessage.objects.filter(source_message_id__in=ids)
                       .values_list("source_message_id", flat=True))
        objs = []
        for r in chunk:
            if r.source_message_id in existing:
                continue
            is_root = r.in_response_to_id is None
            is_candidate = r.author_role == CustomerMessage.AuthorRole.CUSTOMER and is_root
            objs.append(CustomerMessage(
                source_message_id=r.source_message_id, import_batch=batch, author_role=r.author_role,
                author_hash=r.author_hash, brand_account=batch.brand_account, created_at=r.created_at,
                text=r.text, in_response_to_id=r.in_response_to_id, response_ids=r.response_ids,
                is_thread_root=is_root,
                analysis_status=(CustomerMessage.AnalysisStatus.PENDING if is_candidate
                                 else CustomerMessage.AnalysisStatus.NOT_APPLICABLE),
            ))
        with transaction.atomic():
            # ignore_conflicts guards against a concurrent import inserting the same tweet_id.
            CustomerMessage.objects.bulk_create(objs, ignore_conflicts=True)

    mine = CustomerMessage.objects.filter(import_batch=batch)
    batch.accepted_count = mine.count()
    batch.duplicate_count = len(valid) - batch.accepted_count + in_file_duplicates
    batch.customer_root_count = mine.filter(author_role="customer", is_thread_root=True).count()
    batch.agent_count = mine.filter(author_role="agent").count()
    batch.status = (ImportBatch.Status.COMPLETED_WITH_ERRORS if rejected else ImportBatch.Status.COMPLETED)
    batch.finished_at = timezone.now()
    batch.save()
    logger.info("Import %s: total=%s accepted=%s duplicates=%s rejected=%s", batch.pk, total,
                batch.accepted_count, batch.duplicate_count, rejected)
