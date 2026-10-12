# Echovance — AI-Powered Customer Feedback Intelligence & Action Management

Echovance turns historical customer-support conversations into operational decisions:

**import → validate → detect complaints → extract issues with an LLM (verified evidence) → group →
weekly trends → deterministic alerts → AI-proposed actions → manager approval → task tracking**

It is a decision-support tool, not a chatbot. It was built on the public *Customer Support on Twitter*
dataset (SpotifyCares, Oct–Nov 2017); the privacy-reviewed development subset and the 13 genuine Gemini
results are included (`data/spotifycares_public/`, CC BY-NC-SA 4.0). Final project for the Kernel CAIE
Level 2 AI Engineering program.

| | |
|---|---|
| Stack | Python 3.12, Django 5.1, Django REST Framework, PostgreSQL 16, server-rendered HTML/CSS + minimal JS |
| AI | Google Gemini (`gemini-3.5-flash-lite`) via REST with structured output, behind a persisted request/cost budget and a privacy gate. A clearly labelled keyword **mock** runs the whole workflow offline for demos and tests. OpenAI integration is present but **disabled**. |
| Tests | 264 pytest tests (PostgreSQL, no network), incl. 5 real-browser tests and 43 real-AI-path tests with mocked HTTP |

**Two AI modes — never mixed.** *Demo mode* (the `.env.example` default) uses the deterministic keyword
mock: no API key, no network, no cost. Its results are labelled `MOCK` everywhere and live in a separate
**demo data scope**; they are not AI analysis. *Real mode* uses Gemini and only works with a valid API key
**and** a request budget explicitly authorised by the owner (see [Real Gemini integration](#real-gemini-integration)).

## Quick start (clean machine)

Requirements: Python 3.12, PostgreSQL 16 (14+ should work), Git. Commands are for Windows PowerShell;
on macOS/Linux use `.venv/bin/python` and `cp`.

```powershell
git clone https://github.com/amer533/Echovance.git
cd Echovance
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(50))"   # paste into DJANGO_SECRET_KEY
```

Edit `.env`: set `DJANGO_SECRET_KEY` and `DB_PASSWORD`. Keep `ECHOVANCE_AI_MODE=demo` and `LLM_PROVIDER=mock`
to work offline (no API key).

### Database (fresh PostgreSQL database)

Create a role and an empty database once, as a PostgreSQL superuser (psql or pgAdmin):

```sql
CREATE ROLE echovance WITH LOGIN CREATEDB PASSWORD '<DB_PASSWORD from .env>';
CREATE DATABASE echovance OWNER echovance ENCODING 'UTF8';
```

`CREATEDB` lets pytest create its own temporary `test_<DB_NAME>` database. Alternatively, set
`POSTGRES_ADMIN_PASSWORD` in `.env` and run `.venv\Scripts\python scripts\setup_db.py` (creates the role and
database if missing; it never drops anything).

```powershell
.venv\Scripts\python manage.py migrate                    # creates all tables and the role groups
.venv\Scripts\python manage.py seed_demo_users            # manager, analyst, viewer, agent1
.venv\Scripts\python manage.py createsuperuser            # optional, for /admin
.venv\Scripts\python manage.py runserver                  # http://127.0.0.1:8000
```

`seed_demo_users` generates one random password and prints it **once** (pass `--password` to choose
your own; existing users are never changed). Roles: **Managers** approve/assign actions, review alerts,
authorise jobs (+ everything analysts can do) · **Analysts** import, analyse, detect, correct, label ·
**Viewers** read only (an assignee may still start/complete their own actions).

## Real-data walkthrough (SpotifyCares, no API key needed)

`data/spotifycares_public/` holds the real data used during development, prepared for redistribution:
4,000 SpotifyCares customer conversations (Oct–Nov 2017) + 4,000 brand replies, with URLs, handles, names
and number-like identifiers redacted, plus the **13 genuine Gemini analyses** made during development.
Source, licence (CC BY-NC-SA 4.0, non-commercial), changes and residual privacy risk:
[`data/spotifycares_public/README.md`](data/spotifycares_public/README.md).

**1. Import the data and the genuine Gemini results** (into the fresh database from the quick start):

```powershell
.venv\Scripts\python manage.py import_messages data\spotifycares_public\spotifycares_subset.csv --data-classification public
#   -> total=8000 accepted=8000 customer_roots=4000 agent_replies=4000   (re-running it only finds duplicates)
.venv\Scripts\python manage.py import_llm_snapshot
#   -> 13 genuine analyses imported; no API request made, the AI budget of this database is not used
```

**2. Inspect the real AI results offline** (log in as `manager`; the dashboard defaults to *Real AI data*):

* **Dashboard**: provenance shows 13 conversations analysed by Gemini and 3,987 unprocessed; 9 issues.
* **Messages** → filter *Status: completed* → open one: Gemini's analysis (provider `gemini`,
  model `gemini-3.5-flash-lite`, prompt `extract-v1`, tokens, latency) with the quoted evidence highlighted in
  the message. The originating run says "Imported snapshot of genuine gemini results" (`GET /api/analyses/`).
* **Issues** (real scope): the issues found by Gemini, each linked to its evidence.
* **Alerts** (real scope): *Detect trends* reports **insufficient data** — 13 analyses are far below the
  100 per week the rules require. This is the honest result, not a failure.

**3. Optional — run the full pipeline on the real texts offline with the mock** (results labelled `MOCK`,
kept in the separate demo scope; the 13 Gemini results are not touched):

```powershell
.venv\Scripts\python manage.py run_analysis --limit 5000   # Provider: MOCK ... 3,987 conversations
```

Then **Dashboard → Mock demo data**, **Alerts → Detect trends → scope Mock demo analyses** (5 weekly
windows, 20 `surge` alerts on the real conversations, e.g. `account_access.login_failure`: 35 vs. a
baseline mean of 19 in the week of 24 Nov), open an alert, *Acknowledge*, *Suggest action with AI*
(a `[MOCK]` template proposal, status **Proposed**), *Approve* → *Assign* to `agent1` → log in as `agent1`
→ *Start* → *Complete*. The audit trail records every step; completing an action does not resolve the issue.

**4. Optional — blind labelling of the held-out evaluation sample:**

```powershell
New-Item -ItemType Directory -Force data\eval | Out-Null
Copy-Item data\spotifycares_public\eval\*.csv data\eval\
```

**Labelling** (`/evaluation/label/`) then serves the 200 held-out messages without model output, and
`manage.py evaluate_ai` reports metrics once enough independent labels and real predictions exist.

### What works offline vs. what needs an API key

| Offline (no key, no cost) | Needs `GEMINI_API_KEY` + an owner-authorised budget |
|---|---|
| import the real subset; inspect the 13 genuine Gemini results, their evidence, issues and usage | analysing any further conversation with Gemini (UI *Analyse*, jobs, `promote_mock`) |
| full pipeline with the labelled mock: grouping, trends, alerts, `[MOCK]` action proposals, approval workflow | real (non-mock) AI action proposals |
| human corrections, blind labelling, REST API, tests, benchmark | real-scope trend alerts (need ≥ 100 real analyses per week) and AI quality metrics |

The REST API offers the same steps, e.g. `POST /api/alerts/detect/ {"scope": "demo"}`;
`manage.py detect_trends` evaluates the **real** scope only.

## Optional: synthetic offline fixture

`data/sample/synthetic_demo_messages.csv` is a small **invented** dataset (not real customers) in the same
format, for quick offline tests or demos: 440 conversations over 4 weeks (2–29 Oct 2017) plus 220 brand
replies, designed so that one `surge` alert (`billing_subscription.unexpected_charge`, 15 vs. baseline 3)
and one `emerging` alert (`playback_performance.service_outage`) appear. Use it in a separate, empty
database (import it with `import_messages`, then `run_analysis --limit 500` and detect in the demo scope).

A live HTTP smoke test exercises login, permissions, pages and the action workflow against a running
server. It writes records (an action, an alert review) and needs at least one open alert, so run it on a
review database after step 3 of the real-data walkthrough (or with the fixture):
`.venv\Scripts\python scripts\smoke_test.py --base http://127.0.0.1:8000 --password <demo password>`.

## Real Gemini integration

Real analysis is **off by default** and does not work without all of the following:

1. A valid Google AI Studio key in `GEMINI_API_KEY`, `ECHOVANCE_AI_MODE=real`, `LLM_PROVIDER=gemini`.
2. An owner-authorised request budget. A new database starts with `LLM_INITIAL_AUTHORIZED_REQUESTS`
   (default 3); raising it is explicit and capped by `LLM_LIFETIME_REQUEST_CEILING`:
   ```powershell
   .venv\Scripts\python manage.py llm_budget                           # used / authorised / remaining / halted
   .venv\Scripts\python manage.py llm_budget --authorize 10 --confirm  # cumulative total
   ```
3. Data the privacy gate may send: the Gemini **free tier only ever receives imports classified `public`**
   (`GEMINI_TIER=paid` requires both token prices and allows `LLM_PAID_ALLOWED_DATA_CLASSES`).

Bulk work runs as **jobs**: *AI jobs → New job* (new conversations | re-analyse mock-analysed | retry
failed | specific ids) → a manager clicks *Authorise* (refused if the global budget cannot cover it) →
worker: `.venv\Scripts\python manage.py process_jobs --loop` (safe to stop and restart). The UI/API
*Analyse* button sends at most `LLM_REAL_MAX_PER_REQUEST` (default 1) message per click;
`run_analysis` refuses real providers without `--allow-real`. Afterwards run
`manage.py detect_trends --all` (real scope) and `manage.py llm_budget --log` (every real request with
status, tokens, latency, cost). On the real subset, a *new conversations* job analyses still-unprocessed
messages and a *re-analyse mock-analysed* job replaces mock results; new analyses use the current prompt
(`extract-v2`), while the 13 imported `extract-v1` results stay as they are.

Guarantees: every request is logged *before* it is sent and admitted only within the global budget, the
job's limits, token/cost/rate limits and the privacy gate; quota/rate-limit/auth errors halt live calls
with no retry (`llm_budget --clear-halt` after resolving); there is no fallback to another provider or
to the mock; a failed real analysis is stored as failed and never shown as a result; identical redacted
inputs reuse an earlier validated result. Before sending, names in explicit contexts, e-mails, phone-like
and card-like numbers, account/order codes, IPs, street addresses, secrets, URLs and @handles are redacted.

**Project status (honest):** during development 13 real Gemini requests were made (3 verification + a
10-message pilot); their results are the snapshot in `data/spotifycares_public/gemini_snapshot.json`. AI
quality metrics (precision/recall/F1) are **not yet measured**: they require
independent human labels (see [Evaluation](#evaluation)) and an authorised real batch.

## Environment variables (`.env`)

| Variable | `.env.example` | Meaning |
|---|---|---|
| `DJANGO_SECRET_KEY` | placeholder | required; the app refuses to start without it when `DJANGO_DEBUG=False` |
| `DJANGO_DEBUG` | `True` | development mode (code default `False`; secure cookies when `False`) |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | comma-separated |
| `DB_ENGINE` | `postgresql` | `sqlite` only for quick experiments (tests target PostgreSQL) |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | `echovance`, `echovance`, —, `localhost`, `5432` | application DB |
| `POSTGRES_ADMIN_USER`, `POSTGRES_ADMIN_PASSWORD` | `postgres`, empty | only for `scripts/setup_db.py` |
| `ECHOVANCE_AI_MODE` | `demo` | `demo`: mock allowed, every result labelled MOCK · `real`: only real providers analyse, mock refused (code default `real`) |
| `LLM_PROVIDER` | `mock` | `mock` \| `gemini` \| `openai` (refused unless enabled) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | empty / `gemini-3.5-flash-lite` | real mode only |
| `LLM_OPENAI_ENABLED` | `false` | must be `true` (with authorisation) before OpenAI can be used |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `30` / `0` | quota/rate-limit errors are never retried |
| `LLM_MAX_INPUT_CHARS` / `LLM_MAX_OUTPUT_TOKENS` | `1000` / `1024` | input/output bounds |
| `LLM_INITIAL_AUTHORIZED_REQUESTS` | `3` | initial persisted authorised total (then only `llm_budget --authorize`) |
| `LLM_REAL_MAX_PER_REQUEST` | `1` | real-provider messages per UI/API *Analyse* click |
| `LLM_LIFETIME_REQUEST_CEILING` | `20` | maximum the authorised total may be raised to |
| `GEMINI_TIER` | `free` | `free` ($0; public data only) \| `paid` (requires both prices) |
| `GEMINI_PRICE_INPUT_PER_MTOK` / `..._OUTPUT_...` | empty | USD per 1M tokens; unknown prices on a paid tier ⇒ refused |
| `LLM_MONTHLY_COST_CEILING_USD` | `0` | month-to-date spend ceiling |
| `LLM_MAX_INPUT_TOKENS_PER_REQUEST` / `LLM_RATE_LIMIT_RPM` | `4000` / `10` | per-request bound / request spacing |
| `LLM_PAID_ALLOWED_DATA_CLASSES` | `public,internal` | privacy gate for paid tiers (`confidential` is never sent) |
| `GROUPING_SIMILARITY` | `tfidf` | `tfidf` (local, free) \| `gemini_embedding` (budgeted API) |
| `CSV_MAX_UPLOAD_BYTES` / `CSV_MAX_ROWS` | `5242880` / `20000` | web upload limits |

## Tests

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt   # Playwright for the browser tests (drives installed Microsoft Edge)
.venv\Scripts\python -m pytest              # all tests; PostgreSQL from .env (temporary test_<DB_NAME> database)
.venv\Scripts\python -m pytest -m e2e       # end-to-end workflow only
.venv\Scripts\python -m pytest -m browser   # real-browser tests only (skipped if Playwright/Edge is missing)
```

Tests never call a real provider, whatever `.env` says: they force the mock or mock the HTTP transport,
and all fixtures are synthetic (one test additionally checks that the shipped public data and Gemini snapshot import consistently).

| Test module | Covers |
|---|---|
| `test_imports.py` | valid/idempotent import, duplicates, partial failure, bad dates/columns/encoding, size/row limits, customer vs agent separation, PII redaction |
| `test_llm_client.py` / `test_llm_budget.py` | budget guard, bounded retries, halt on quota/rate-limit/auth, malformed JSON, OpenAI disabled, Gemini request shape, persisted budget, CLI confirmation, 1-message cap, outbound redaction |
| `test_real_ai.py` | real Gemini code path with mocked HTTP: mock refused in real mode, no fallback, mock→real promotion, jobs and limits, pricing, rate limiting, privacy gate, stale alerts, grouping suggestions, corrections, blind labelling |
| `test_extraction.py` / `test_prompt_quality.py` | multi-aspect extraction, hallucinated evidence, taxonomy pairs, prompt-injection delimiting, strict schema, split/leakage guards |
| `test_analysis.py` / `test_latency.py` | idempotent runs, failure persistence & retry, supersession, latency breakdown |
| `test_trends.py` | half-open windows, surge/emerging thresholds, insufficient history, de-duplication |
| `test_actions.py` | state machine, permissions, audit trail, AI proposals stay proposed |
| `test_api.py` / `test_web.py` | auth, permissions, status codes, pagination, filters, contracts, forms, escaping |
| `test_browser_actions.py` | real Edge clicks: approve, reject (reason required), edit, assignee start/complete, viewer has no buttons |
| `test_e2e.py` | import ×2 → analyse → evidence traceability → detect → review → AI proposal → approve → assign → complete |
| `test_evaluation.py` | metric arithmetic, confusion matrix, pairwise grouping, mock refusal |
| `test_snapshots.py` | genuine-LLM snapshot import: validated, attributed, idempotent, supersedes mock, rejects mock/human or non-matching evidence, never uses the AI budget; shipped data consistency |

Performance (all writes are rolled back; output JSON is written to `docs/results/`, which is git-ignored):
`.venv\Scripts\python manage.py benchmark --subset data\spotifycares_public\spotifycares_subset.csv`

## Evaluation

`manage.py evaluate_ai` compares **real** LLM predictions (never the mock) with independent human labels
and reports precision/recall/F1, confusion matrices and Cohen's κ; without labels it reports `pending`.
Labels are entered on **Labelling** (`/evaluation/label/`, analysts and managers), which shows one
held-out message at a time **without any model output**; managers may add an adjudication label. It reads
the held-out sample from `data/eval/eval_sample.csv`; copy the published splits there first (step 4 of the
real-data walkthrough), otherwise the page is empty. No human labels exist yet.

Labelling rules: `message_type` is `complaint` (something does not work, an unwanted behaviour or charge,
dissatisfaction), `feature_request` (asks for something new without a malfunction), `question` (neutral
how-to), `praise` or `other`. A question that reports a failure, a mixed praise+complaint and sarcasm
describing a failure are all **complaints**. Issue keys (`category.subtype` from `feedback/taxonomy.py`)
apply only to complaints and feature requests: one key per distinct problem, never one problem split into
several keys. Do not change labels after seeing model output.

## Trend rules

Deterministic, versioned (`TREND_RULES` in `config/settings.py`, logic in `feedback/services/trends.py`).
Windows are 7-day, half-open `[start, end)` in UTC, compared with the 3 preceding windows; counts are
distinct customer conversations, in ONE data scope (real = LLM + human, demo = mock).

* **insufficient data** — current or any baseline window has < 100 successfully analysed conversations → no alerts
* **surge** — baseline mean > 0, current ≥ 8, increase ≥ 5, growth ≥ ×1.5 and share growth ≥ ×1.3
* **emerging** — no baseline mentions and current ≥ 6

Alerts are immutable snapshots with an input fingerprint; if re-analysis changes their inputs they are
marked stale (reviews and actions are kept) and re-detection creates a new revision.

## REST API

All endpoints require authentication (session or HTTP Basic — use HTTPS outside local development).
Lists are paginated (`?page=`, 25 per page) and support `?ordering=`. Errors return `{"detail": ...}`;
validation errors return field-keyed messages. Browsable at `/api/` after logging in.

| Method & path | Permission | Purpose |
|---|---|---|
| `POST /api/imports/` (multipart `file`, `brand_account`, `data_classification`) | run_pipeline | 201 batch / 400 failed batch (body includes reason) |
| `GET /api/imports/`, `/api/imports/{id}/` | any user | batch counts, row errors (row/field/code) |
| `GET /api/messages/` | any user | filters: `author_role`, `analysis_status`, `is_thread_root`, `import_batch`, `issue`, `q`, `created_after`, `created_before` |
| `GET /api/messages/{id}/` | any user | analyses (provider, model, prompt, is_mock), mentions with evidence offsets, replies |
| `POST /api/messages/{id}/correct/` `{message_type, aspects[{category, subtype, sentiment, evidence}], note}` | run_pipeline | human correction (evidence must quote the message) |
| `POST /api/analyses/run/` `{limit, message_ids?, retry_failed?}` | run_pipeline | 201 run; limit ≤ 25 (mock) / 1 (real); 503 if the provider is not configured |
| `GET /api/analyses/` | any user | analysis runs incl. `stopped_reason` |
| `GET /api/issues/?scope=real\|demo` | any user | counts, first/last seen, open alerts in ONE scope; filters `category`, `status` |
| `GET /api/issues/{id}/` · `/api/issues/{id}/mentions/` | any user | + 8-week `weekly_trend` + recent mentions · paginated evidence |
| `PATCH /api/issues/{id}/` `{status}` | manage_actions | open / monitoring / resolved |
| `GET /api/alerts/`, `/api/alerts/{id}/` | any user | filters `status`, `rule`, `issue`, `data_scope`, `is_stale`; detail includes evidence |
| `POST /api/alerts/detect/` `{window_end?, all_windows?, scope?}` | run_pipeline | per-window result: created / existing / stale / insufficient-data reason |
| `POST /api/alerts/{id}/review/` `{status: acknowledged\|dismissed, note}` | manage_actions | note required to dismiss |
| `GET /api/actions/`, `/api/actions/{id}/` | any user | detail includes the audit `events` |
| `POST /api/actions/` `{issue, alert?, title, description, rationale?}` | manage_actions | creates a **proposed** action |
| `POST /api/actions/suggest/` `{issue, alert?}` | manage_actions | AI proposal (201), 502 if the AI failed (nothing created), 503 provider unavailable |
| `PATCH /api/actions/{id}/` | see rules | exactly one of `{status, note}`, `{assignee, due_date}`, `{title, description}`; 409 invalid transition, 403 not allowed |
| `GET /api/dashboard/summary/?scope=real\|demo` | any user | dashboard statistics for ONE scope + provenance breakdown |
| `GET/POST /api/jobs/` `{selection, max_messages, message_ids?, max_requests?, max_cost_usd?}` | run_pipeline (create) | analysis jobs (created as draft) |
| `GET /api/jobs/{id}/estimate/` | any user | estimated tokens / cost / minimum duration |
| `POST /api/jobs/{id}/authorize/` · `resume/` | manage_actions | 409 if not allowed (e.g. global budget too small) |
| `POST /api/jobs/{id}/pause/` · `cancel/` | run_pipeline | |
| `GET /api/grouping-suggestions/` · `POST …/{id}/accept/` · `…/reject/` `{note}` | run_pipeline (decide) | hybrid grouping review |

Example: `curl -u manager:<password> "http://127.0.0.1:8000/api/dashboard/summary/?scope=demo"`

## Project structure

```
config/                 settings (environment-driven), urls, wsgi/asgi
feedback/
  models/               imports, messages, analysis, issues, alerts, actions, llm_usage, evaluation
  migrations/           schema + role groups + data backfills (0001–0009)
  services/             imports, analysis, jobs, grouping, trends, actions, review, corrections,
                        llm_budget, privacy_gate, split_guard, stats, evaluation, snapshots
  ai/                   client (Gemini / OpenAI / Mock), extraction, actions, privacy (redaction),
                        similarity (TF-IDF, Gemini embeddings), mock_rules
  api/                  DRF serializers, viewsets, routes
  web/                  views, forms, urls, context processor
  templates/feedback/   server-rendered pages + partials
  static/feedback/      app.css, app.js
  management/commands/  import_messages, import_llm_snapshot, run_analysis, create_job, process_jobs,
                        detect_trends, llm_budget, evaluate_ai, benchmark, seed_demo_users
  taxonomy.py           single source of truth for categories/subtypes
scripts/                setup_db (create role/DB), smoke_test (live HTTP checks), prepare_dataset and
                        export_public_dataset (author-side: how data/spotifycares_public was produced)
data/spotifycares_public/  real SpotifyCares subset (redacted), evaluation splits, Gemini snapshot, manifest,
                        data notice (CC BY-NC-SA 4.0)
data/sample/            synthetic_demo_messages.csv (invented offline fixture)
tests/                  pytest suite (synthetic fixtures; one test checks the shipped public data)
```

## How the published data was produced

1. `scripts/prepare_dataset.py` selects the subset from the Kaggle dump `twcs.csv` (~516 MB, not in the
   repository; seed 20171001) into the git-ignored `data/processed/` and `data/eval/`.
2. `scripts/export_public_dataset.py` redacts it and captures the genuine Gemini results from the development
   database (read-only) into `data/spotifycares_public/`, refusing to publish any quote that is not found
   verbatim in the redacted text.

Both need the private inputs, so reviewers do not need to run them; they document exactly what was done.
Only `data/spotifycares_public/` and `data/sample/` are committed; everything else under `data/` is git-ignored.

## Data licence & privacy

`data/spotifycares_public/` is an adaptation of "Customer Support on Twitter" (Stuart Axelbrooke, Kaggle)
under **CC BY-NC-SA 4.0** — non-commercial use, attribution, share-alike; see its
[data notice](data/spotifycares_public/README.md) for the changes and the residual privacy risk. In the
application, stored author ids are hashed, e-mails and phone-like numbers are redacted on import, and URLs,
handles and other identifiers are additionally masked before any external LLM call. Counts describe answered
public tweets only, never all customers, and trends never imply causality. Secrets live only in `.env`
(git-ignored).
