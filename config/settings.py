"""Django settings for Echovance.

All secrets and environment-specific values come from environment variables (see .env.example).
"""
import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ImproperlyConfigured(f"Missing required environment variable {name}")
    return value


def env_bool(name: str, default: bool = False) -> bool:
    return env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(env(name, str(default)))


DEBUG = env_bool("DJANGO_DEBUG", False)
SECRET_KEY = env("DJANGO_SECRET_KEY", "insecure-dev-key-only" if DEBUG else None)
ALLOWED_HOSTS = [h.strip() for h in env("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "rest_framework",
    "django_filters",
    "feedback",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "feedback.web.context.llm_mode",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DB_ENGINE = env("DB_ENGINE", "postgresql")
if DB_ENGINE == "postgresql":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env("DB_NAME", "echovance"),
            "USER": env("DB_USER", "echovance"),
            "PASSWORD": env("DB_PASSWORD", ""),
            "HOST": env("DB_HOST", "localhost"),
            "PORT": env("DB_PORT", "5432"),
            "CONN_MAX_AGE": 60,
        }
    }
elif DB_ENGINE == "sqlite":
    DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
else:
    raise ImproperlyConfigured("DB_ENGINE must be 'postgresql' or 'sqlite'")

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "login"

SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = False
X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

# Bounded web uploads; large raw dataset preparation happens offline in scripts/.
CSV_MAX_UPLOAD_BYTES = env_int("CSV_MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
CSV_MAX_ROWS = env_int("CSV_MAX_ROWS", 20_000)
DATA_UPLOAD_MAX_MEMORY_SIZE = 2_621_440
FILE_UPLOAD_MAX_MEMORY_SIZE = 2_621_440

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_THROTTLE_CLASSES": ["rest_framework.throttling.UserRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {"user": "600/minute"},
}

# ---- LLM integration (see feedback/ai) ----
# Providers: "mock" (deterministic test double), "gemini" (live, budget-guarded),
# "openai" (kept for future use; refused unless LLM_OPENAI_ENABLED=true is set with authorisation).
LLM = {
    "PROVIDER": env("LLM_PROVIDER", "mock").strip().lower(),
    "GEMINI_API_KEY": env("GEMINI_API_KEY", ""),
    "GEMINI_MODEL": env("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "GEMINI_BASE_URL": env("GEMINI_BASE_URL", "") or None,
    "OPENAI_ENABLED": env_bool("LLM_OPENAI_ENABLED", False),
    "OPENAI_API_KEY": env("OPENAI_API_KEY", ""),
    "OPENAI_MODEL": env("OPENAI_MODEL", "gpt-4o-mini"),
    "OPENAI_BASE_URL": env("OPENAI_BASE_URL", "") or None,
    "TIMEOUT_SECONDS": float(env("LLM_TIMEOUT_SECONDS", "30")),
    # 0 = never retry. Quota/rate-limit errors are never retried regardless of this value.
    "MAX_RETRIES": env_int("LLM_MAX_RETRIES", 0),
    "MAX_INPUT_CHARS": env_int("LLM_MAX_INPUT_CHARS", 1000),
    "MAX_OUTPUT_TOKENS": env_int("LLM_MAX_OUTPUT_TOKENS", 1024),
    # Empty string = omit the parameter (some reasoning models reject temperature).
    "TEMPERATURE": float(env("LLM_TEMPERATURE", "0")) if env("LLM_TEMPERATURE", "0") != "" else None,
    # Real-request budget: initial authorised total (persisted in DB). Changed only via `llm_budget --authorize`.
    "INITIAL_AUTHORIZED_REQUESTS": env_int("LLM_INITIAL_AUTHORIZED_REQUESTS", 3),
    # Real providers: messages per web/API "Analyse" click (bulk real work goes through authorised jobs).
    "REAL_MAX_PER_REQUEST": env_int("LLM_REAL_MAX_PER_REQUEST", 1),
    # ---- Budget v2 (configurable ceilings; see services/llm_budget.py) ----
    "LIFETIME_REQUEST_CEILING": env_int("LLM_LIFETIME_REQUEST_CEILING", 20),
    "MONTHLY_COST_CEILING_USD": env("LLM_MONTHLY_COST_CEILING_USD", "0"),
    "MAX_INPUT_TOKENS_PER_REQUEST": env_int("LLM_MAX_INPUT_TOKENS_PER_REQUEST", 4000),
    "RATE_LIMIT_RPM": env_int("LLM_RATE_LIMIT_RPM", 10),
    # Provider tier and prices. Free tier = $0. A paid tier REQUIRES explicit prices (USD per 1M tokens);
    # without them real requests are refused rather than guessed.
    "GEMINI_TIER": env("GEMINI_TIER", "free").strip().lower(),
    "GEMINI_PRICE_INPUT_PER_MTOK": env("GEMINI_PRICE_INPUT_PER_MTOK", ""),
    "GEMINI_PRICE_OUTPUT_PER_MTOK": env("GEMINI_PRICE_OUTPUT_PER_MTOK", ""),
    "OPENAI_PRICE_INPUT_PER_MTOK": env("OPENAI_PRICE_INPUT_PER_MTOK", ""),
    "OPENAI_PRICE_OUTPUT_PER_MTOK": env("OPENAI_PRICE_OUTPUT_PER_MTOK", ""),
    # ---- Privacy gate: which data classifications each provider tier may receive ----
    # Gemini free tier may use submitted content to improve Google products -> public data only.
    "ALLOWED_DATA_CLASSES": {
        "gemini:free": ["public"],
        "gemini:paid": [c.strip() for c in env("LLM_PAID_ALLOWED_DATA_CLASSES", "public,internal").split(",") if c.strip()],
        "openai:paid": [c.strip() for c in env("LLM_PAID_ALLOWED_DATA_CLASSES", "public,internal").split(",") if c.strip()],
    },
    # ---- Embeddings (optional, for grouping similarity) ----
    "GEMINI_EMBEDDING_MODEL": env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
}

# real = only real providers may analyse (mock refused); demo = mock allowed, everything labelled MOCK.
ECHOVANCE_AI_MODE = env("ECHOVANCE_AI_MODE", "real").strip().lower()
if ECHOVANCE_AI_MODE not in {"real", "demo"}:
    raise ImproperlyConfigured("ECHOVANCE_AI_MODE must be 'real' or 'demo'")

# Hybrid grouping: similarity method ("tfidf" = local, zero cost; "gemini_embedding" = budgeted API)
GROUPING = {
    "SIMILARITY": env("GROUPING_SIMILARITY", "tfidf").strip().lower(),
    "MIN_SIMILARITY": float(env("GROUPING_MIN_SIMILARITY", "0.30")),
    "MARGIN": float(env("GROUPING_MARGIN", "0.10")),
    "MIN_ISSUE_MENTIONS": env_int("GROUPING_MIN_ISSUE_MENTIONS", 3),
}

# Analysis jobs (bounded, resumable batches processed by `manage.py process_jobs`)
JOBS = {
    "DEFAULT_MAX_MESSAGES": env_int("JOB_DEFAULT_MAX_MESSAGES", 25),
    "STALE_HEARTBEAT_MINUTES": env_int("JOB_STALE_HEARTBEAT_MINUTES", 10),
}
# Maximum messages analysed by one synchronous web/API request; bulk runs use the
# `run_analysis` management command.
ANALYSIS_MAX_BATCH_PER_REQUEST = env_int("ANALYSIS_MAX_BATCH_PER_REQUEST", 25)

# ---- Trend rules (summarised in README, "Trend rules"; logic in feedback/services/trends.py) ----
TREND_RULES = {
    "RULE_VERSION": "trend-v1",
    "WINDOW_DAYS": 7,
    "BASELINE_WINDOWS": 3,
    "MIN_WINDOW_ANALYZED_MESSAGES": 100,
    "MIN_CURRENT_MENTIONS": 8,
    "MIN_ABSOLUTE_INCREASE": 5,
    "MIN_GROWTH_RATIO": 1.5,
    "MIN_SHARE_RATIO": 1.3,
    "MIN_EMERGING_MENTIONS": 6,
    "EVIDENCE_LIMIT": 10,
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "loggers": {
        "feedback": {"handlers": ["console"], "level": env("LOG_LEVEL", "INFO")},
        "httpx": {"handlers": ["console"], "level": "WARNING"},
        "openai": {"handlers": ["console"], "level": "WARNING"},
    },
}
