"""Isolated LLM client.

Responsibilities: provider selection, credentials from settings/env, timeouts, bounded retries,
JSON parsing, provider/model tracking and (for real providers) a request-budget guard. It knows
nothing about Django models or business rules; callers own prompts and validation, and the
guard (feedback.services.llm_budget) owns persistence of request counts.

Safety policy for real providers:
  * every attempt is admitted by the guard first (hard, persisted request budget);
  * quota / rate-limit / auth / unknown-model errors are never retried and halt live calls;
  * there is NO fallback between providers: a missing key or disabled provider raises
    LLMConfigError. The mock provider must be chosen explicitly and is always labelled is_mock.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from django.conf import settings

logger = logging.getLogger(__name__)

# Errors after which no further real requests may be sent until a human clears the halt.
HALT_CODES = {"quota_exhausted", "rate_limited", "auth_error", "model_not_found", "free_tier_unavailable"}


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
class GeminiProvider:
    """Google Gemini via the Generative Language REST API (generateContent).

    Structured output differences vs OpenAI: Gemini takes `responseMimeType=application/json` plus a
    `responseSchema` in its OpenAPI-subset dialect, which does not accept `additionalProperties`;
    `to_gemini_schema` converts our JSON Schema and adds `propertyOrdering`. Validation of the
    returned JSON is still done by our own validators, never trusted from the provider.
    """

    name = "gemini"
    is_mock = False
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str, timeout: float, max_output_tokens: int,
                 temperature: float | None = 0.0, base_url: str | None = None, transport=None):
        if not api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set; set it or choose LLM_PROVIDER=mock explicitly.")
        import httpx

        self._httpx = httpx
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        # The key travels in a header (never in the URL) so it cannot leak into logs of URLs.
        self._http = httpx.Client(base_url=base_url or self.DEFAULT_BASE_URL, timeout=timeout,
                                  headers={"x-goog-api-key": api_key}, transport=transport)

    @staticmethod
    def to_gemini_schema(schema: dict) -> dict:
        def convert(node):
            if isinstance(node, dict):
                out = {k: convert(v) for k, v in node.items() if k != "additionalProperties"}
                if "properties" in out:
                    out["propertyOrdering"] = list(out["properties"])
                return out
            if isinstance(node, list):
                return [convert(x) for x in node]
            return node
        return convert(copy.deepcopy(schema))

    @staticmethod
    def _error_from_response(resp) -> LLMError:
        try:
            err = resp.json().get("error", {})
        except ValueError:
            err = {}
        status, message = str(err.get("status", "")), str(err.get("message", ""))[:200]
        code = resp.status_code
        detail = f"HTTP {code} {status}: {message}".strip()
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            lowered = message.lower()
            kind = "quota_exhausted" if ("quota" in lowered or "limit: 0" in lowered) else "rate_limited"
            return LLMError(kind, detail, retryable=False, http_status=code)
        if code in (401, 403) or "api key not valid" in message.lower():
            if "billing" in message.lower() or "free tier" in message.lower():
                return LLMError("free_tier_unavailable", detail, http_status=code)
            return LLMError("auth_error", "Provider rejected the API credentials or permissions. " + status,
                            http_status=code)
        if code == 404:
            return LLMError("model_not_found", detail, http_status=code)
        if code == 400 and "location is not supported" in message.lower():
            return LLMError("free_tier_unavailable", detail, http_status=code)
        if code >= 500:
            return LLMError("provider_unavailable", detail, retryable=True, http_status=code)
        return LLMError("bad_request", detail, http_status=code)

    def call(self, system: str, user: str, schema_name: str, schema: dict) -> ProviderReply:
        gen = {"responseMimeType": "application/json", "responseSchema": self.to_gemini_schema(schema),
               "maxOutputTokens": self.max_output_tokens, "candidateCount": 1}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen}
        try:
            resp = self._http.post(f"/models/{self.model}:generateContent", json=body)
        except self._httpx.TimeoutException:
            raise LLMError("timeout", "Gemini request timed out.", retryable=True) from None
        except self._httpx.TransportError as exc:
            raise LLMError("connection_error", type(exc).__name__, retryable=True) from None
        if resp.status_code != 200:
            raise self._error_from_response(resp)
        try:
            data = resp.json()
        except ValueError:
            raise LLMError("invalid_json", "Gemini returned a non-JSON HTTP body.") from None
        usage = data.get("usageMetadata") or {}
        tokens = dict(input_tokens=usage.get("promptTokenCount"), output_tokens=usage.get("candidatesTokenCount"),
                      total_tokens=usage.get("totalTokenCount"))
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise LLMError("refused", f"Prompt blocked: {block}")
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError("empty_response", "Gemini returned no candidates.")
        cand = candidates[0]
        finish = cand.get("finishReason", "")
        if finish == "MAX_TOKENS":
            raise LLMError("truncated_output", "Model output hit the token limit.")
        if finish and finish not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            raise LLMError("refused", f"Generation stopped: {finish}")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return ProviderReply(text, **tokens)


class OpenAIProvider:
    """Kept for future use; only constructed when LLM_OPENAI_ENABLED=true is set explicitly."""

    name = "openai"
    is_mock = False

    def __init__(self, api_key: str, model: str, timeout: float, max_output_tokens: int,
                 base_url: str | None = None, temperature: float | None = 0.0):
        if not api_key:
            raise LLMConfigError("OPENAI_API_KEY is not set; set it or choose LLM_PROVIDER=mock explicitly.")
        import openai  # imported lazily so tests and mock mode don't need network config

        self._openai = openai
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        # SDK retries disabled: LLMClient owns the retry policy.
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def call(self, system: str, user: str, schema_name: str, schema: dict) -> ProviderReply:
        oa = self._openai
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
        except oa.APITimeoutError as exc:
            raise LLMError("timeout", str(exc), retryable=True) from None
        except oa.RateLimitError as exc:
            raise LLMError("rate_limited", str(exc), retryable=False, http_status=429) from None
        except oa.APIConnectionError as exc:
            raise LLMError("connection_error", str(exc), retryable=True) from None
        except oa.InternalServerError as exc:
            raise LLMError("provider_unavailable", str(exc), retryable=True) from None
        except (oa.AuthenticationError, oa.PermissionDeniedError):
            raise LLMError("auth_error", "Provider rejected the API credentials.") from None
        except oa.BadRequestError as exc:
            raise LLMError("bad_request", str(exc)) from None
        except oa.APIError as exc:
            raise LLMError("provider_error", str(exc)) from None
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError("refused", "Model refused to answer.")
        if choice.finish_reason == "length":
            raise LLMError("truncated_output", "Model output hit the token limit.")
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

    if name == "gemini":
        provider = GeminiProvider(api_key=cfg["GEMINI_API_KEY"], model=cfg["GEMINI_MODEL"],
                                  timeout=cfg["TIMEOUT_SECONDS"], max_output_tokens=cfg["MAX_OUTPUT_TOKENS"],
                                  temperature=cfg["TEMPERATURE"], base_url=cfg["GEMINI_BASE_URL"])
    elif name == "openai":
        if not cfg["OPENAI_ENABLED"]:
            raise LLMConfigError("The OpenAI provider is disabled to protect the shared quota. "
                                 "It may only be enabled (LLM_OPENAI_ENABLED=true) with explicit authorisation.")
        provider = OpenAIProvider(api_key=cfg["OPENAI_API_KEY"], model=cfg["OPENAI_MODEL"],
                                  timeout=cfg["TIMEOUT_SECONDS"], max_output_tokens=cfg["MAX_OUTPUT_TOKENS"],
                                  base_url=cfg["OPENAI_BASE_URL"], temperature=cfg["TEMPERATURE"])
    else:
        raise LLMConfigError(f"Unsupported LLM_PROVIDER '{name}'. Use 'gemini', 'mock' or (when authorised) 'openai'.")
    return LLMClient(provider, max_retries=cfg["MAX_RETRIES"], guard=RequestBudget())


def describe_configuration() -> dict:
    """Safe, secret-free description of the configured provider for the UI."""
    cfg = settings.LLM
    name = cfg["PROVIDER"]
    model = {"gemini": cfg["GEMINI_MODEL"], "openai": cfg["OPENAI_MODEL"]}.get(name, "mock-keyword-v2")
    key = {"gemini": cfg["GEMINI_API_KEY"], "openai": cfg["OPENAI_API_KEY"]}.get(name)
    return {"provider": name, "model": model, "is_mock": name == "mock",
            "credentials_present": bool(key) if name != "mock" else None,
            "disabled": name == "openai" and not cfg["OPENAI_ENABLED"],
            "ai_mode": settings.ECHOVANCE_AI_MODE,
            "mock_refused": name == "mock" and settings.ECHOVANCE_AI_MODE != "demo",
            "tier": cfg["GEMINI_TIER"] if name == "gemini" else ("paid" if name == "openai" else None)}
