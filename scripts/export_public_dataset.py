"""Build the redistributable, privacy-reviewed copy of the development data (author-side tool).

    python scripts/export_public_dataset.py

Reads (local, git-ignored): data/processed/spotifycares_subset.csv, data/eval/{dev_examples,eval_sample}.csv
and, read-only, the development database (for the genuine Gemini analyses).
Writes: data/spotifycares_public/ (subset, evaluation splits, Gemini snapshot, manifest).

Sanitisation (every text, customer and brand):
  * outbound redaction v2 (feedback/ai/privacy.py) - the same transformation applied before any text
    was sent to Gemini: e-mails, URLs, @handles, card/phone/ip/id-like numbers, addresses, secrets and
    names in explicit contexts become placeholders such as [email], [link], @user, [name];
  * Spotify usernames / account names in free text ("username: m.x", "my account (abc)") become [username];
  * brand replies additionally: the customer's first name in the greeting ("Hey Duncan!") and the
    agent's initials signature ("/NJ") become [name].
tweet_id/author_id are kept: they are already anonymised ids in the source dataset, which keeps the
subset traceable to it. Gemini evidence quotes are passed through the same redaction and must still be
found verbatim in the published text; otherwise the export stops (nothing is invented or edited by hand).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

from feedback.ai import extraction  # noqa: E402
from feedback.ai.privacy import redact  # noqa: E402
from feedback.models import MessageAnalysis  # noqa: E402

OUT = ROOT / "data" / "spotifycares_public"
SANITISATION_VERSION = "public-v1 (outbound redaction v2 + usernames + brand-reply greeting/signature names)"
# Name = capitalised word in any script (Zoë, Iñigo, Björn); only the greeting word is case-insensitive.
_GREETING_RE = re.compile(r"^((?:@user\s+)*(?i:hey|hi|hello|hiya|heya|howdy)\s+)(?!(?i:there|again|all|everyone)\b)"
                          r"((?=[^\W\d_])(?=\w)[^\Wa-z\d_][\w'’-]*)(?=[\s!,.:?]|$)")
_SIGNATURE_RE = re.compile(r"(\s)/\s?\^?[A-Z]{1,3}(\s*(?:\[link\])?\s*)$")
# Spotify usernames / account names written in free text ("username: jane.doe7", "account (jdoe)") - invented examples.
_LABEL = r"(?:user ?name|account name|display name|profile name|login (?:name|id)|spotify (?:user ?name|name|id))"
_TOKEN = r"([^\W_](?:[\w.\-]*[^\W_])?)"  # a word or identifier, without trailing punctuation
_OPEN = r"(?:\s*[(\"“‘]|\s+')"  # opening bracket/quote; a straight ' only after a space (not "account's")
_USERNAME_EXPLICIT = re.compile(rf"(?i)(\b{_LABEL}(?:\s*[:=]\s*|{_OPEN}|\s+(?:is|was)\s+[(\"'“‘]?)){_TOKEN}")
_USERNAME_BARE = re.compile(rf"(?i)(\b{_LABEL}\s+){_TOKEN}")
_ACCOUNT_QUOTED = re.compile(rf"(?i)(\baccount(?:\s+(?:is|was))?{_OPEN}){_TOKEN}")
_ACCOUNT_BARE = re.compile(rf"(?i)(\baccount(?:\s+(?:is|was))?\s+){_TOKEN}")
_FUNCTION_WORDS = set("""a an the and or of is was not no still just now here there via to in for on at as if it its
that this these those my your you we they he she his her their our me us them by from with when what which be been
being has have had do does did can could would should will may might must so but also only even too very""".split())


def _identifier_like(token: str) -> bool:
    return bool(re.search(r"\d|\w[._\-]\w", token))


def redact_usernames(s: str, vocabulary: Counter, counts: Counter) -> str:
    def replace(match, keep):
        token = match.group(2)
        if token.lower() in _FUNCTION_WORDS or token.startswith(("__", "[")) or not keep(token):
            return match.group(0)
        counts["username"] += 1
        return match.group(1) + "[username]"

    rare = lambda t: _identifier_like(t) or (len(t) >= 4 and vocabulary[t.lower()] <= 2)  # noqa: E731
    s = _USERNAME_EXPLICIT.sub(lambda m: replace(m, lambda t: True), s)
    s = _USERNAME_BARE.sub(lambda m: replace(m, rare), s)
    s = _ACCOUNT_QUOTED.sub(lambda m: replace(m, lambda t: True), s)
    s = _ACCOUNT_BARE.sub(lambda m: replace(m, _identifier_like), s)
    # a rare capitalised word right after a redacted name is its second part (e.g. an invented "Ana Kowalczyk")
    return re.sub(r"(\[username\])(\s+)((?=[^\W\d_])[^\Wa-z\d_][\w'’-]*)",
                  lambda m: m.group(1) if rare(m.group(3)) else m.group(0), s)


def sanitise(text: str, brand_reply: bool, counts: Counter, vocabulary: Counter) -> str:
    result = redact(text)
    counts.update(result.counts)
    s = redact_usernames(result.text, vocabulary, counts)
    if brand_reply:
        s, n = _GREETING_RE.subn(r"\1[name]", s)
        counts["greeting_name"] += n
        s, n = _SIGNATURE_RE.subn(r"\1/[name]\2", s)
        counts["agent_signature"] += n
    return s


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    counts: Counter = Counter()
    src = ROOT / "data" / "processed" / "spotifycares_subset.csv"
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        source_rows = list(reader)
    # word frequencies over the whole (redacted) corpus: a username is a rare token, "via" is not
    vocabulary = Counter(w.lower() for r in source_rows for w in re.findall(r"[^\W_][\w.\-]*", redact(r["text"]).text))
    rows = [{**r, "text": sanitise(r["text"], r["inbound"] == "False", counts, vocabulary)} for r in source_rows]
    published = {r["tweet_id"]: r["text"] for r in rows}
    write_csv(OUT / "spotifycares_subset.csv", header, rows)

    splits = {}
    for name in ("dev_examples.csv", "eval_sample.csv"):
        with (ROOT / "data" / "eval" / name).open(newline="", encoding="utf-8") as f:
            split = list(csv.DictReader(f))
        missing = [r["tweet_id"] for r in split if r["tweet_id"] not in published]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} ids are not in the subset")
        write_csv(OUT / "eval" / name, ["tweet_id", "created_at", "text"],
                  [{"tweet_id": r["tweet_id"], "created_at": r["created_at"], "text": published[r["tweet_id"]]}
                   for r in split])
        splits[name] = len(split)

    analyses = []
    qs = (MessageAnalysis.objects.filter(source="llm", is_mock=False).select_related("message")
          .order_by("created_at", "pk"))
    for a in qs:
        msg = a.message
        text = published[str(msg.source_message_id)]
        out = a.validated_output or {}
        aspects = [{"category": x["category"], "subtype": x["subtype"], "sentiment": x["sentiment"],
                    "summary": redact(x["summary"]).text, "evidence": redact(x["evidence"]).text}
                   for x in out.get("aspects", [])]
        item = {"tweet_id": int(msg.source_message_id), "message_type": out["message_type"],
                "is_complaint": out["is_complaint"], "overall_sentiment": out.get("overall_sentiment", "neutral"),
                "aspects": aspects, "review_reasons": list(a.review_reasons), "status": a.status,
                "analysed_at": a.created_at.isoformat(timespec="seconds"), "attempts": a.attempts,
                "latency_ms": a.latency_ms, "input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
                "origin": f"analysis job #{a.job_id}" if a.job_id else "single verification request"}
        # Same validator the application uses: every quote must be found verbatim in the published text.
        extraction.validate_extraction({**item, "needs_review": False, "review_reason": ""}, text,
                                       prompt_version=a.prompt_version)
        for asp in aspects:
            if extraction.locate_evidence(asp["evidence"], text) is None:
                raise SystemExit(f"tweet {msg.source_message_id}: evidence not found after redaction")
        analyses.append((a, item))
    if analyses:
        first = analyses[0][0]
        snapshot = {
            "kind": "echovance.llm_analysis_snapshot",
            "description": "Genuine Google Gemini analyses produced during development of Echovance. "
                           "Captured from the development database; NOT re-generated, NOT mock output. "
                           "Evidence and summaries are expressed in the redacted published text.",
            "provider": first.provider, "model_name": first.model_name, "prompt_version": first.prompt_version,
            "requests_made": len(analyses), "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": "data/spotifycares_public/spotifycares_subset.csv",
            "analyses": [item for _, item in analyses],
        }
        assert all(a.provider == first.provider and a.model_name == first.model_name and
                   a.prompt_version == first.prompt_version for a, _ in analyses)
        (OUT / "gemini_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in ("manifest.json", "README.md"))
    manifest = {
        "sanitisation": SANITISATION_VERSION,
        "source": {"title": "Customer Support on Twitter", "creator": "Stuart Axelbrooke (Thought Vector)",
                   "url": "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter",
                   "license": "CC BY-NC-SA 4.0", "file": "twcs.csv",
                   "sha256": json.loads((ROOT / "data" / "processed" / "spotifycares_subset_manifest.json")
                                        .read_text(encoding="utf-8"))["source_sha256"]},
        "rows": len(rows), "customer_thread_roots": sum(r["inbound"] == "True" for r in rows),
        "brand_replies": sum(r["inbound"] == "False" for r in rows), "splits": splits,
        "gemini_analyses": len(analyses), "redactions": dict(sorted(counts.items())),
        "files": {p.relative_to(OUT).as_posix(): sha256(p) for p in files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "customer_thread_roots", "brand_replies", "splits",
                                                "gemini_analyses", "redactions")}, indent=2))


if __name__ == "__main__":
    main()
