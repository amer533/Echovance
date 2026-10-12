"""Privacy gate: decides whether a message may be sent to a given external provider/tier.

Policy (settings.LLM["ALLOWED_DATA_CLASSES"]):
  * gemini free tier  -> "public" only (free-tier content may be used to improve Google products)
  * paid tiers        -> LLM_PAID_ALLOWED_DATA_CLASSES (default public,internal)
  * "confidential"    -> never sent to an external provider
  * the local mock provider sends nothing externally and is always allowed
After redaction, residual-risk signals (e.g. long digit sequences) also block sending.
"""
from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings

from feedback.ai.privacy import RedactionResult, redact, residual_risks
from feedback.models import CustomerMessage, ImportBatch
from feedback.services.llm_budget import provider_tier


@dataclass
class GateDecision:
    allowed: bool
    reason: str
    redaction: RedactionResult


def allowed_classes(provider: str) -> list[str]:
    return settings.LLM["ALLOWED_DATA_CLASSES"].get(f"{provider}:{provider_tier(provider)}", [])


def check(message: CustomerMessage, provider: str, is_mock: bool) -> GateDecision:
    red = redact(message.text)
    if is_mock:
        return GateDecision(True, "local mock provider (nothing leaves the machine)", red)
    data_class = message.import_batch.data_classification
    if data_class == ImportBatch.DataClass.CONFIDENTIAL:
        return GateDecision(False, "confidential data is never sent to an external provider", red)
    allowed = allowed_classes(provider)
    if data_class not in allowed:
        return GateDecision(False, f"'{data_class}' data may not be sent to {provider} "
                                   f"({provider_tier(provider)} tier allows: {', '.join(allowed) or 'nothing'})", red)
    risks = residual_risks(red.text)
    if risks:
        return GateDecision(False, "possible personal data remains after redaction: " + ", ".join(risks), red)
    return GateDecision(True, f"'{data_class}' data allowed for {provider} ({provider_tier(provider)} tier)", red)
