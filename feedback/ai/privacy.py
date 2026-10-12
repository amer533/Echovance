"""Outbound redaction v2: what leaves the machine for an external LLM.

Stored text is already redacted at import (e-mails -> [email], long digit runs -> [phone]).
Before any external call we redact, in order:

  [secret]  values after password/pin/passcode/cvv labels
  [email]   e-mail addresses
  [link]    URLs (t.co links can resolve to personal media)
  @user     @handles (customer handles are pseudonymous ids, but still identifiers)
  [card]    13-19 digit card-like numbers
  [ip]      IPv4 addresses
  [phone]   phone-like sequences with >= 9 digits
  [id]      account/order/reference codes (mixed letters+digits, or 7-8 digit numbers)
  [address] simple street addresses ("12 Baker Street")
  [name]    names in explicit contexts: "my name is X", sign-offs ("Thanks, Chad"), dash signatures

Name detection is heuristic and cannot find every name; that is why the privacy gate
(services/privacy_gate.py) also restricts which data classifications may reach which provider tier.
`evidence_regex` maps placeholders in model-quoted evidence back to the original stored text.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from feedback.services.imports import redact_pii

_SECRET_RE = re.compile(r"(?i)\b(password|passcode|pin|cvv|cvc)\b(\s*(?:is|=|:)\s*)(\S+)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+")
_HANDLE_RE = re.compile(r"(?<![\w@])@\w+")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IP_RE = re.compile(rf"(?<![\w.])(?:{_OCTET}\.){{3}}{_OCTET}(?![\w.])")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
_ID_RE = re.compile(r"\b(?=[A-Za-z0-9]{7,}\b)(?=(?:[A-Za-z]*\d){3,})(?=(?:\d*[A-Za-z]){2,})[A-Za-z0-9]+\b|(?<![\d.,$£€])\b\d{7,8}\b(?![.,]\d)")
_ADDRESS_RE = re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct)\b\.?")
_NAME_WORD = r"[A-Z][a-z'’-]+"
_NAME_RES = [
    re.compile(rf"(?i:\bmy name is\s+)({_NAME_WORD}(?:\s+{_NAME_WORD})?)"),
    re.compile(rf"(?i:\b(?:thanks|thank you|regards|cheers|sincerely|best),?\s+)({_NAME_WORD}(?:\s+{_NAME_WORD})?)\s*[.!]*\s*$"),
    re.compile(rf"(?<=\s)[-–—]\s?({_NAME_WORD}|[A-Z]{{2,3}})\s*$"),
]

PLACEHOLDER_PATTERNS = {
    "[link]": r"https?://\S+|www\.\S+",
    "@user": r"@\w+",
    "[name]": rf"{_NAME_WORD}(?:\s+{_NAME_WORD})?|[A-Z]{{2,3}}",
    "[id]": r"[A-Za-z0-9]{7,}",
    "[card]": r"(?:\d[ -]?){12,18}\d",
    "[ip]": r"(?:\d{1,3}\.){3}\d{1,3}",
    "[address]": r"\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\.?",
    "[secret]": r"\S+",
    "[email]": r"\[email\]|[\w.+-]+@[\w-]+\.[\w.-]+",
    "[phone]": r"\[phone\]|\+?\d[\d\s().-]{7,}\d",
}
_PLACEHOLDER_SPLIT = re.compile("(" + "|".join(re.escape(k) for k in PLACEHOLDER_PATTERNS) + ")")


@dataclass
class RedactionResult:
    text: str
    counts: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    counts: Counter = Counter()

    def sub(rx, repl, s, label, group=None):
        def f(m):
            counts[label] += 1
            if group is None:
                return repl
            return m.group(0).replace(m.group(group), repl)
        return rx.sub(f, s)

    s = _SECRET_RE.sub(lambda m: (counts.update(["secret"]), f"{m.group(1)}{m.group(2)}[secret]")[1], text)
    s = sub(_EMAIL_RE, "[email]", s, "email")
    s = sub(_URL_RE, "[link]", s, "link")
    s = sub(_HANDLE_RE, "@user", s, "handle")
    s = sub(_CARD_RE, "[card]", s, "card")
    s = sub(_IP_RE, "[ip]", s, "ip")
    s = _PHONE_RE.sub(lambda m: (counts.update(["phone"]), "[phone]")[1]
                      if sum(c.isdigit() for c in m.group()) >= 9 else m.group(), s)
    s = sub(_ID_RE, "[id]", s, "id")
    s = sub(_ADDRESS_RE, "[address]", s, "address")
    for rx in _NAME_RES:
        s = sub(rx, "[name]", s, "name", group=1)
    s = redact_pii(s)  # same rules as import-time redaction (defence in depth)
    return RedactionResult(s, dict(counts))


def sanitize_for_llm(text: str) -> str:
    return redact(text).text


def residual_risks(text: str) -> list[str]:
    """Signals that personal/confidential data survived redaction (used to block sending)."""
    risks = []
    if re.search(r"\d{7,}", text):
        risks.append("long_digit_sequence")
    if re.search(r"(?i)\b(ssn|social security|passport|iban|sort code|routing number)\b", text):
        risks.append("government_or_bank_identifier_keyword")
    if _EMAIL_RE.search(text.replace("[email]", "")):
        risks.append("email_address")
    return risks


def evidence_regex(evidence: str) -> re.Pattern | None:
    """Regex matching `evidence` in the original text: placeholders -> patterns, flexible whitespace."""
    parts = []
    for tok in _PLACEHOLDER_SPLIT.split(evidence.strip()):
        if tok in PLACEHOLDER_PATTERNS:
            parts.append("(?:" + PLACEHOLDER_PATTERNS[tok] + ")")
        elif tok:
            core = r"\s+".join(re.escape(w) for w in tok.split()) if tok.strip() else r"\s+"
            if tok.strip():
                if tok[:1].isspace():
                    core = r"\s*" + core
                if tok[-1:].isspace():
                    core = core + r"\s*"
            parts.append(core)
    pattern = "".join(parts)
    return re.compile(pattern, re.IGNORECASE) if pattern else None
