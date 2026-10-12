"""Keeps the development split and the held-out evaluation split apart.

* The 50-message DEVELOPMENT split may be read and reviewed by developers (prompt work, pilots).
* The 200-message HELD-OUT EVALUATION split is only for independent human labels and final metrics.

Guards:
  - the two splits must be disjoint (checked by id);
  - development-split messages can never become evaluation gold labels;
  - no prompt may contain dataset text from either split (8-word overlap check), so reviewed examples
    cannot leak into extraction/evaluation prompts as few-shot examples.
Reading the evaluation file here only serves these integrity checks; its texts are never used as input.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from django.conf import settings

NGRAM = 8


def _split_path(name: str) -> Path:
    return Path(settings.BASE_DIR) / "data" / "eval" / name


def _rows(name: str) -> list[dict]:
    path = _split_path(name)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dev_ids() -> set[int]:
    return {int(r["tweet_id"]) for r in _rows("dev_examples.csv")}


def eval_ids() -> set[int]:
    return {int(r["tweet_id"]) for r in _rows("eval_sample.csv")}


def _ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def leaked_texts(prompt: str, texts: list[str], n: int = NGRAM) -> int:
    """Number of dataset texts sharing at least one n-word sequence with the prompt."""
    grams = _ngrams(prompt, n)
    return sum(1 for t in texts if _ngrams(t, n) & grams)


def integrity_report(prompts: dict[str, str]) -> dict:
    dev_rows, eval_rows = _rows("dev_examples.csv"), _rows("eval_sample.csv")
    d = {int(r["tweet_id"]) for r in dev_rows}
    e = {int(r["tweet_id"]) for r in eval_rows}
    leaks = {name: {"dev_split_texts": leaked_texts(p, [r["text"] for r in dev_rows]),
                    "eval_split_texts": leaked_texts(p, [r["text"] for r in eval_rows])}
             for name, p in prompts.items()}
    return {"dev_size": len(d), "eval_size": len(e), "overlap_ids": len(d & e), "prompt_leakage": leaks,
            "ok": not (d & e) and all(v["dev_split_texts"] == 0 and v["eval_split_texts"] == 0 for v in leaks.values())}


def current_prompts() -> dict[str, str]:
    from feedback.ai import actions as action_ai
    from feedback.ai import extraction
    return {f"extraction:{extraction.PROMPT_VERSION}": extraction.SYSTEM_PROMPT,
            f"action:{action_ai.PROMPT_VERSION}": action_ai.SYSTEM_PROMPT}
