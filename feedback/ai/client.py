"""Isolated LLM client.

Responsibilities: provider selection, credentials from settings/env, timeouts, bounded retries,
JSON parsing, provider/model tracking and (for real providers) a request-budget guard. It knows
nothing about Django models or business rules; callers own prompts and validation, and the
guard (feedback.services.llm_budget) owns persistence of request counts.

Safety policy for real providers:
  * every attempt is admitted by the guard first (hard, persisted request budget);
  * quota / rate-limit / auth / unknown-model errors are never retried and halt live calls;
  * there is NO fallback between providers: a missing key or unknown provider raises
    LLMConfigError. The mock provider must be chosen explicitly and is always labelled is_mock.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from django.conf import settings

logger = logging.getLogger(__name__)

# Errors after which no further real requests may be sent until a human clears the halt.
HALT_CODES = {"quota_exhausted", "rate_limited", "auth_error", "model_not_found"}


class LLMError(Exception):
    def __init__(self, code: str, detail: str = "", retryable: bool = False, attempts: int = 0,
                 http_status: int | None = None):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail[:300]
        self.retryable = retryable
        self.attempts = attempts
        self.http_status = http_status
        self.timing: dict | None = None  # queue_wait_ms / provider_latency_ms / total_ms when known


class LLMConfigError(LLMError):
    def __init__(self, detail: str):
        super().__init__("config_error", detail, retryable=False)


@dataclass
class ProviderReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class LLMResponse:
    data: dict
    provider: str
    model: str
    is_mock: bool
    latency_ms: int  # = total_ms (kept for backwards compatibility)
    attempts: int
    input_tokens: int | None
    output_tokens: int | None
    provider_latency_ms: int = 0  # time inside provider calls only (all attempts)
    queue_wait_ms: int = 0  # budget admission incl. rate-limiter sleep (all attempts)
    total_ms: int = 0  # wall time of complete_json (wait + provider + retry backoff + parsing)


class RequestGuard(Protocol):
    """Implemented by feedback.services.llm_budget.RequestBudget."""

    def acquire(self, provider: str, model: str, purpose: str, *, estimated_input_tokens: int = 0,
                fingerprint: str = "") -> int: ...
    def finish(self, ticket: int, *, ok: bool, latency_ms: int, error: LLMError | None = None,
               reply: ProviderReply | None = None) -> None: ...
    def halt(self, reason: str) -> None: ...


# ------------------------------------------------------------------------------------------------
_SECRET_IN_TEXT = re.compile(r"sk-[A-Za-z0-9_\-*]{4,}")


def _safe_detail(exc: Exception) -> str:
    """Provider error text for logs/UI, with anything that looks like an API key masked."""
    return _SECRET_IN_TEXT.sub("sk-***", str(exc))[:300]


class OpenAIProvider:
    """OpenAI Chat Completions with strict JSON-schema structured output (default model gpt-4o-mini).

    The provider's JSON is never trusted: callers validate it with their own validators. SDK retries
    are disabled because LLMClient owns the retry policy and the request budget counts every attempt.
    """

    name = "openai"
    is_mock = False

    def __init__(self, api_key: str, model: str, timeout: float, max_output_tokens: int,
                 base_url: str | None = None, temperature: float | None = 0.0, transport=None):
        if not api_key:
            raise LLMConfigError("OPENAI_API_KEY is not set; set it in .env (or use ECHOVANCE_AI_MODE=demo with "
                                 "LLM_PROVIDER=mock for clearly labelled offline demo data).")
        import openai  # imported lazily so tests and mock mode don't need network config

        self._openai = openai
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        http_client = None
        if transport is not None:  # tests: mocked HTTP, the real SDK code path, no network
            import httpx
            http_client = httpx.Client(transport=transport)
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0,
                                     http_client=http_client)

    def _error(self, exc) -> LLMError:
        oa = self._openai
        status = getattr(exc, "status_code", None)
        if isinstance(exc, oa.APITimeoutError):
            return LLMError("timeout", "OpenAI request timed out.", retryable=True)
        if isinstance(exc, oa.APIConnectionError):
            return LLMError("connection_error", type(exc).__name__, retryable=True)
        if isinstance(exc, oa.RateLimitError):
            # 429 is both "too many requests" and "insufficient_quota" (no credit left): never retried.
            code = str(getattr(exc, "code", "") or "")
            kind = "quota_exhausted" if code == "insufficient_quota" or "quota" in str(exc).lower() else "rate_limited"
            return LLMError(kind, _safe_detail(exc), http_status=status)
        if isinstance(exc, (oa.AuthenticationError, oa.PermissionDeniedError)):
            return LLMError("auth_error", "OpenAI rejected the API key or its permissions.", http_status=status)
        if isinstance(exc, oa.NotFoundError):
            return LLMError("model_not_found", _safe_detail(exc), http_status=status)
        if isinstance(exc, (oa.BadRequestError, oa.UnprocessableEntityError)):
            return LLMError("bad_request", _safe_detail(exc), http_status=status)
        if isinstance(exc, oa.InternalServerError):
            return LLMError("provider_unavailable", _safe_detail(exc), retryable=True, http_status=status)
        return LLMError("provider_error", _safe_detail(exc), http_status=status)

    def call(self, system: str, user: str, schema_name: str, schema: dict) -> ProviderReply:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except self._openai.APIError as exc:
            raise self._error(exc) from None
        if not resp.choices:
            raise LLMError("empty_response", "OpenAI returned no choices.")
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError("refused", "Model refused to answer.")
        if choice.finish_reason == "length":
            raise LLMError("truncated_output", "Model output hit the token limit.")
        if choice.finish_reason == "content_filter":
            raise LLMError("refused", "Output stopped by the provider's content filter.")
        usage = resp.usage
        return ProviderReply(choice.message.content or "",
                             usage.prompt_tokens if usage else None,
                             usage.completion_tokens if usage else None,
                             usage.total_tokens if usage else None)


class MockProvider:
    """Deterministic, keyword-based TEST DOUBLE. Not AI, not evaluated as AI.

    `responder(schema_name, user_prompt) -> str | dict | Exception` can be injected by tests to
    simulate malformed output or provider failures; by default feedback.ai.mock_rules is used.
    """

    name = "mock"
    is_mock = True

    def __init__(self, model: str = "mock-keyword-v2", responder: Callable[[str, str], object] | None = None):
        self.model = model
        self.responder = responder

    def call(self, system: str, user: str, schema_name: str, schema: dict) -> ProviderReply:
        if self.responder is not None:
            out = self.responder(schema_name, user)
        else:
            from . import mock_rules
            out = mock_rules.respond(schema_name, user)
        if isinstance(out, Exception):
            raise out
        return ProviderReply(out if isinstance(out, str) else json.dumps(out))


# ------------------------------------------------------------------------------------------------
class LLMClient:
    def __init__(self, provider, max_retries: int = 0, backoff_seconds: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep, guard: RequestGuard | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.provider = provider
        self._clock = clock
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        if guard is None and not provider.is_mock:
            raise LLMConfigError("Real providers require a request budget guard.")
        self.guard = guard

    @property
    def provider_name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def is_mock(self) -> bool:
        return self.provider.is_mock

    @staticmethod
    def request_fingerprint(provider: str, model: str, schema_name: str, system: str, user: str) -> str:
        payload = "".join((provider, model, schema_name, system, user))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ms(self, seconds: float) -> int:
        return max(0, int(round(seconds * 1000)))

    def _attempt(self, system, user, schema_name, schema, purpose, timing: dict) -> ProviderReply:
        """One admitted provider call. Adds to timing['queue_wait'] and timing['provider'] (seconds)."""
        ticket = None
        if self.guard:
            w0 = self._clock()
            try:
                ticket = self.guard.acquire(
                    self.provider.name, self.provider.model, purpose,
                    estimated_input_tokens=(len(system) + len(user) + len(json.dumps(schema))) // 3 + 1,
                    fingerprint=self.request_fingerprint(self.provider.name, self.provider.model, schema_name,
                                                         system, user))
            finally:
                timing["queue_wait"] += self._clock() - w0
        t0 = self._clock()
        try:
            reply = self.provider.call(system, user, schema_name, schema)
        except LLMError as exc:
            elapsed = self._clock() - t0
            timing["provider"] += elapsed
            if self.guard:
                self.guard.finish(ticket, ok=False, latency_ms=self._ms(elapsed), error=exc)
                if exc.code in HALT_CODES:
                    self.guard.halt(f"{exc.code}: {exc.detail}"[:300])
            raise
        elapsed = self._clock() - t0
        timing["provider"] += elapsed
        if self.guard:
            self.guard.finish(ticket, ok=True, latency_ms=self._ms(elapsed), reply=reply)
        return reply

    def _timing_ms(self, timing: dict, started: float) -> dict:
        return {"queue_wait_ms": self._ms(timing["queue_wait"]), "provider_latency_ms": self._ms(timing["provider"]),
                "total_ms": self._ms(self._clock() - started)}

    def complete_json(self, system: str, user: str, schema_name: str, schema: dict,
                      purpose: str = "unspecified") -> LLMResponse:
        attempts = 0
        started = self._clock()
        timing = {"queue_wait": 0.0, "provider": 0.0}
        while True:
            attempts += 1
            try:
                reply = self._attempt(system, user, schema_name, schema, purpose, timing)
                break
            except LLMError as exc:
                exc.attempts = attempts
                if not exc.retryable or exc.code in HALT_CODES or attempts > self.max_retries:
                    exc.timing = self._timing_ms(timing, started)
                    logger.warning("LLM call failed: provider=%s code=%s attempts=%s",
                                   self.provider.name, exc.code, attempts)
                    raise
                self._sleep(self.backoff_seconds * (2 ** (attempts - 1)))
        t = self._timing_ms(timing, started)
        try:
            data = json.loads(reply.text)
        except (json.JSONDecodeError, TypeError):
            err = LLMError("invalid_json", "Provider output was not valid JSON.", attempts=attempts)
            err.timing = t
            raise err from None
        if not isinstance(data, dict):
            err = LLMError("invalid_json", "Provider output was not a JSON object.", attempts=attempts)
            err.timing = t
            raise err
        return LLMResponse(data=data, provider=self.provider.name, model=self.provider.model,
                           is_mock=self.provider.is_mock, latency_ms=t["total_ms"], attempts=attempts,
                           input_tokens=reply.input_tokens, output_tokens=reply.output_tokens, **t)


def get_llm_client() -> LLMClient:
    cfg = settings.LLM
    name = cfg["PROVIDER"]
    if name == "mock":
        if settings.ECHOVANCE_AI_MODE != "demo":
            raise LLMConfigError("The keyword mock provider is disabled in real AI mode "
                                 "(ECHOVANCE_AI_MODE=real). Configure a real provider, or set "
                                 "ECHOVANCE_AI_MODE=demo for clearly labelled demo data.")
        return LLMClient(MockProvider(), max_retries=0)
    from feedback.services.llm_budget import RequestBudget

    if name != "openai":
        raise LLMConfigError(f"Unsupported LLM_PROVIDER '{name}'. Use 'openai' (or 'mock' in demo mode).")
    provider = OpenAIProvider(api_key=cfg["OPENAI_API_KEY"], model=cfg["OPENAI_MODEL"],
                              timeout=cfg["TIMEOUT_SECONDS"], max_output_tokens=cfg["MAX_OUTPUT_TOKENS"],
                              base_url=cfg["OPENAI_BASE_URL"], temperature=cfg["TEMPERATURE"])
    return LLMClient(provider, max_retries=cfg["MAX_RETRIES"], guard=RequestBudget())


def describe_configuration() -> dict:
    """Safe, secret-free description of the configured provider for the UI."""
    cfg = settings.LLM
    name = cfg["PROVIDER"]
    model = cfg["OPENAI_MODEL"] if name == "openai" else "mock-keyword-v2"
    return {"provider": name, "model": model, "is_mock": name == "mock",
            "credentials_present": bool(cfg["OPENAI_API_KEY"]) if name == "openai" else None,
            "unsupported": name not in ("openai", "mock"),
            "ai_mode": settings.ECHOVANCE_AI_MODE,
            "mock_refused": name == "mock" and settings.ECHOVANCE_AI_MODE != "demo",
            "tier": "paid" if name == "openai" else None}
