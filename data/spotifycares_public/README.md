# SpotifyCares subset (privacy-reviewed) — data notice

This folder contains the real customer-support data used to develop Echovance, prepared for
redistribution, and a snapshot of the genuine Gemini analyses made during development.

## Source and licence

* **Source:** "Customer Support on Twitter" by Stuart Axelbrooke (Thought Vector), Kaggle —
  <https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter> (file `twcs.csv`,
  sha256 in `manifest.json`).
* **Licence:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This folder is an
  adaptation of that dataset and is shared under the **same licence**: attribution required,
  **non-commercial use only**, adaptations must be shared alike. The licence of this folder does not depend on
  the licence of the application code.
* The source dataset already uses anonymised `tweet_id` and `author_id` values (not Twitter ids), replaces
  @mentions with those anonymised ids and masks e-mail addresses and phone numbers.

## What was changed (adaptation)

1. **Selection** (`scripts/prepare_dataset.py`, seed 20171001): 4,000 English, customer-initiated
   conversations addressed to SpotifyCares between 1 Oct and 1 Dec 2017 that received a brand reply,
   sampled at random from 23,226 eligible, plus the first brand reply to each (8,000 rows). The 50-message
   development and 200-message evaluation splits are subsets of these conversations.
2. **Additional redaction** (`scripts/export_public_dataset.py`, `public-v1`), applied to every text:
   outbound redaction v2 (`feedback/ai/privacy.py`, the same transformation used before any text is sent to
   an external LLM): URLs → `[link]`, @handles → `@user`, card/phone/IP/id-like numbers, street addresses,
   secrets and names in explicit contexts → placeholders. Spotify usernames/account names written in the
   text ("username: …", "my account (…)") → `[username]`. In brand replies, the customer's first name in the
   greeting ("Hey Duncan!") and the agent's initials signature ("/NJ") → `[name]`. Counts per rule are in
   `manifest.json` (2,770 links, 8,589 handles, 1,517 greeting names, 3,904 agent signatures, 20 usernames, …).
   The redacted subset was additionally reviewed by pattern searches (URLs, e-mails, handles, long digit runs,
   usernames, greeting names) and random samples.

**Residual risk:** name detection is heuristic. Names written freely inside a message (other than in the
patterns above) can remain, as can public figures' names (artists, albums). Verbatim text may still be found
by web search if the original tweet is public. Use the data for non-commercial education and research only,
do not attempt to re-identify anyone, and open an issue on this repository to request removal of a message.

## Files

| File | Content |
|---|---|
| `spotifycares_subset.csv` | 8,000 rows in the original `twcs.csv` columns: 4,000 customer thread roots + 4,000 brand replies |
| `eval/dev_examples.csv`, `eval/eval_sample.csv` | development (50) and held-out evaluation (200) splits: `tweet_id, created_at, text` |
| `gemini_snapshot.json` | the **13 genuine Gemini analyses** (see below) |
| `manifest.json` | source, counts, redaction counts and sha256 of every file |

## Gemini snapshot (genuine AI output, not mock)

During development, exactly 13 real requests were made to Google Gemini (`gemini-3.5-flash-lite`, prompt
`extract-v1`) on 11 Oct 2026: 3 single verification requests and a 10-message pilot job. The snapshot
contains each validated result (message type, issues, sentiment, Gemini's summary and quoted evidence) and
the recorded usage (tokens, latency, attempts). It was captured from the development database, not
re-generated. Gemini only ever received redacted text, so its quotes are in the redacted form; quotes and
summaries are given in the published redacted text and every quote is found verbatim in it (checked by the
export and again by the importer). No raw API responses or prompts are included.

`manage.py import_llm_snapshot` loads it into a database that contains the subset. The importer accepts only
real-provider results, re-validates each one and never sends a request or uses the database's AI budget.
These 13 results are a qualitative sample, **not** an evaluation: they are too few to measure accuracy.
