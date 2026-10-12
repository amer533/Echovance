"""Build the reproducible Echovance MVP subset from the raw twcs.csv dump.

Usage:
    python scripts/prepare_dataset.py --input data/raw/twcs.csv --brand SpotifyCares \
        --start 2017-10-01 --end 2017-12-01 --sample-size 4000 --seed 20171001

Selection criteria:
  1. Brand relationship is reconstructed from reply links, NOT from text keyword matching:
     a customer tweet belongs to the brand when an outbound tweet authored by the brand
     account has in_response_to_tweet_id == that customer tweet id.
  2. Only customer-initiated thread roots are complaint candidates
     (inbound == True, in_response_to_tweet_id empty, numeric author id).
     Customer follow-ups and brand replies are context, never independent complaints.
  3. created_at within [start, end) UTC (the period with dense, continuous coverage).
  4. English text (langdetect on the text with @mentions and URLs removed).
  5. At least MIN_CONTENT_CHARS of content after removing @mentions and URLs.
  6. Exact duplicate texts from the same author are kept once (earliest tweet).
  7. Simple random sample of `sample-size` roots with a fixed seed.
The output keeps the original 7-column schema so it can be imported through the web UI or
`manage.py import_messages`. For each sampled root, the brand's first reply is included so the
import can demonstrate customer/agent separation and thread traceability.

Outputs:
  data/processed/<brand>_subset.csv          sampled roots + first brand reply
  data/processed/<brand>_subset_manifest.json selection counts, criteria, seed and sha256
  data/eval/dev_examples.csv                  50 roots reserved for prompt development
  data/eval/eval_sample.csv                   200 roots held out for human annotation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]
MIN_CONTENT_CHARS = 15
CHUNK = 250_000


def content_only(text: str) -> str:
    return " ".join(w for w in text.split() if not w.startswith(("@", "http")))


def chunks(path: Path):
    return pd.read_csv(
        path, dtype=str, keep_default_na=False, chunksize=CHUNK, encoding="utf-8"
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/twcs.csv")
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--start", default="2017-10-01")
    ap.add_argument("--end", default="2017-12-01")
    ap.add_argument("--sample-size", type=int, default=4000)
    ap.add_argument("--dev-size", type=int, default=50)
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20171001)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    t0 = time.time()
    src = Path(args.input)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    DetectorFactory.seed = 0

    # pass 1: which customer tweets did the brand reply to, and the brand's earliest reply to each
    first_reply: dict[int, tuple[datetime, dict]] = {}
    for ch in chunks(src):
        out = ch[(ch["inbound"] == "False") & (ch["author_id"] == args.brand)]
        for row in out.to_dict("records"):
            irt = row["in_response_to_tweet_id"]
            if not irt.isdigit():
                continue
            ts = datetime.strptime(row["created_at"], DATE_FORMAT)
            prev = first_reply.get(int(irt))
            if prev is None or ts < prev[0]:
                first_reply[int(irt)] = (ts, row)
    print(f"pass 1: brand replied to {len(first_reply)} tweets ({time.time()-t0:.0f}s)")

    # pass 2: candidate customer thread roots
    funnel = Counter()
    candidates: list[dict] = []
    seen_author_text: set[tuple[str, str]] = set()
    for ch in chunks(src):
        inb = ch[ch["inbound"] == "True"]
        for row in inb.to_dict("records"):
            if not row["tweet_id"].isdigit() or int(row["tweet_id"]) not in first_reply:
                continue
            funnel["customer_tweets_replied_by_brand"] += 1
            if row["in_response_to_tweet_id"] != "":
                funnel["excluded_followup_not_thread_root"] += 1
                continue
            if not row["author_id"].isdigit():
                funnel["excluded_non_customer_author"] += 1
                continue
            ts = datetime.strptime(row["created_at"], DATE_FORMAT)
            if not (start <= ts < end):
                funnel["excluded_outside_date_range"] += 1
                continue
            content = content_only(row["text"])
            if len(content) < MIN_CONTENT_CHARS:
                funnel["excluded_too_short_after_removing_mentions_urls"] += 1
                continue
            try:
                lang = detect(content)
            except LangDetectException:
                lang = "unknown"
            if lang != "en":
                funnel["excluded_non_english"] += 1
                continue
            key = (row["author_id"], " ".join(row["text"].lower().split()))
            if key in seen_author_text:
                funnel["excluded_duplicate_text_same_author"] += 1
                continue
            seen_author_text.add(key)
            row["_ts"] = ts
            candidates.append(row)
    funnel["eligible_thread_roots"] = len(candidates)
    print(f"pass 2: {len(candidates)} eligible roots ({time.time()-t0:.0f}s)")

    # deterministic ordering before sampling so the result does not depend on chunk order
    candidates.sort(key=lambda r: int(r["tweet_id"]))
    rnd = random.Random(args.seed)
    n = min(args.sample_size, len(candidates))
    sample = rnd.sample(candidates, n)
    sample.sort(key=lambda r: r["_ts"])
    sample_ids = [int(r["tweet_id"]) for r in sample]

    # dev / eval split: disjoint random subsets of the sample, chosen with a separate RNG stream
    split_rnd = random.Random(args.seed + 1)
    shuffled = sample_ids[:]
    split_rnd.shuffle(shuffled)
    dev_ids = set(shuffled[: args.dev_size])
    eval_ids = set(shuffled[args.dev_size : args.dev_size + args.eval_size])

    out_dir = Path(args.out_dir)
    proc = out_dir / "processed"
    ev = out_dir / "eval"
    proc.mkdir(parents=True, exist_ok=True)
    ev.mkdir(parents=True, exist_ok=True)
    subset_path = proc / f"{args.brand.lower()}_subset.csv"
    replies = 0
    with subset_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in sample:
            w.writerow({c: r[c] for c in COLUMNS})
            reply = first_reply[int(r["tweet_id"])][1]
            w.writerow({c: reply[c] for c in COLUMNS})
            replies += 1

    by_id = {int(r["tweet_id"]): r for r in sample}
    for name, ids in (("dev_examples.csv", dev_ids), ("eval_sample.csv", eval_ids)):
        with (ev / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tweet_id", "created_at", "text"])
            for i in sorted(ids):
                w.writerow([i, by_id[i]["created_at"], by_id[i]["text"]])

    daily = Counter(r["_ts"].strftime("%Y-%m-%d") for r in sample)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(src).replace("\\", "/"),
        "source_sha256": sha256(src),
        "brand_account": args.brand,
        "window_utc": [args.start, args.end],
        "seed": args.seed,
        "criteria": {
            "relationship": "brand outbound tweet in_response_to_tweet_id == customer tweet id",
            "unit": "customer-initiated thread root (inbound, no in_response_to, numeric author)",
            "language": "langdetect == 'en' on text without @mentions/URLs",
            "min_content_chars": MIN_CONTENT_CHARS,
            "dedupe": "same author + identical normalised text kept once",
            "sampling": "simple random sample without replacement (Python random.Random(seed))",
        },
        "funnel": dict(funnel),
        "sampled_thread_roots": n,
        "brand_replies_included": replies,
        "subset_rows": n + replies,
        "dev_examples": len(dev_ids),
        "eval_sample": len(eval_ids),
        "daily_sampled_roots": dict(sorted(daily.items())),
        "subset_file": str(subset_path).replace("\\", "/"),
        "subset_sha256": sha256(subset_path),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (proc / f"{args.brand.lower()}_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("funnel", "sampled_thread_roots", "subset_rows")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
