"""Deterministic keyword rules used ONLY by the mock provider.

Purpose: let the full pipeline (analysis -> grouping -> trends -> actions) run offline and in
automated tests. These rules are not an AI model and their output must never be reported as
AI evaluation results. Every analysis produced through them is stored with is_mock=True.
"""
from __future__ import annotations

import json
import re

# (category, subtype, regex) — first match per (category, subtype); order matters.
RULES: list[tuple[str, str, str]] = [
    ("billing_subscription", "unexpected_charge", r"charged twice|double charged|charged (me )?again|been charged|got charged|i was charged|wrong amount"),
    ("billing_subscription", "refund_request", r"refund|money back"),
    ("billing_subscription", "premium_not_activated", r"(paid|paying|bought|purchased).{0,60}(still|says).{0,20}free|premium (isn'?t|is not|not) working|premium.{0,30}stopped"),
    ("billing_subscription", "student_plan", r"\bstudent\b"),
    ("billing_subscription", "family_plan", r"\bfamily\b"),
    ("billing_subscription", "promo_offer", r"99p|99c|\$0?\.99|offer|trial|deal\b"),
    ("billing_subscription", "partner_bundle", r"\bhulu\b"),
    ("billing_subscription", "cancellation", r"cancel"),
    ("account_access", "password_reset", r"password|reset email"),
    ("account_access", "login_failure", r"log ?in|sign ?in|logged out"),
    ("account_access", "unauthorized_access", r"hack|not mine|someone else|unauthori[sz]ed"),
    ("account_access", "region_restriction", r"abroad|14 days|overseas|another country|moved to"),
    ("playback_performance", "offline_downloads", r"offline|download"),
    ("playback_performance", "device_connectivity", r"chromecast|sonos|alexa|echo\b|\bps4\b|xbox|smart ?tv|\bcar\b|bluetooth|\bconnect\b"),
    ("playback_performance", "playback_interruption", r"skip|cut(ting)? (out|off)|stops? playing|keeps? (stopping|pausing)|pauses"),
    ("playback_performance", "slow_or_unresponsive", r"\bslow\b|buffer|loading|freez|lag"),
    ("playback_performance", "service_outage", r"\bdown\b|outage|not working for anyone"),
    ("application_errors", "app_crash", r"crash"),
    ("application_errors", "error_message", r"error"),
    ("application_errors", "library_data_loss", r"(playlists?|songs|library|music).{0,30}(disappeared|gone|missing|deleted)"),
    ("application_errors", "ui_change", r"update|new (design|layout)|removed"),
    ("content_catalog", "missing_content", r"not on spotify|isn'?t on spotify|unavailable|greyed|grayed|where('?s| is) .{0,40}album"),
    ("content_catalog", "country_availability", r"available in|in (india|egypt|pakistan|nigeria)"),
    ("recommendations_ads", "ads_experience", r"\bads?\b|advert"),
    ("recommendations_ads", "autoplay_shuffle", r"autoplay|shuffle"),
    ("recommendations_ads", "unwanted_recommendations", r"daily mix|discover weekly|recommend"),
    ("customer_support", "no_response", r"no (one|response|reply)|nobody|not responding|no contact"),
]
FEATURE_RE = r"(please add|can you (add|make)|would be (cool|nice|great)|feature|option to|any plans|when will)"
COMPLAINT_RE = r"(not|n't|never|can'?t|won'?t|problem|issue|help|fix|wrong|broken|annoy|keeps|stopped|why)"
PRAISE_RE = r"(love|thank|great|awesome|best)"


def _evidence(text: str, match: re.Match) -> str:
    # ~30 characters of context on each side, widened to whole words so placeholders such as
    # "@user" / "[link]" are never cut in half.
    start = max(0, match.start() - 30)
    end = min(len(text), match.end() + 30)
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end].strip()


def extract(text: str) -> dict:
    aspects = []
    seen = set()
    for cat, sub, pattern in RULES:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m and (cat, sub) not in seen and len(aspects) < 3:
            seen.add((cat, sub))
            aspects.append({"category": cat, "subtype": sub, "sentiment": "negative",
                            "summary": f"[MOCK] keyword rule matched '{m.group(0)}'.",
                            "evidence": _evidence(text, m), "evidence_basis": "explicit", "related_symptoms": []})
    is_feature = re.search(FEATURE_RE, text, flags=re.IGNORECASE)
    is_complaint = re.search(COMPLAINT_RE, text, flags=re.IGNORECASE)
    if aspects and is_complaint:
        mtype = "complaint"
    elif is_feature:
        mtype = "feature_request"
        if not aspects:
            m = is_feature
            aspects = [{"category": "feature_request", "subtype": "new_feature", "sentiment": "neutral",
                        "summary": "[MOCK] feature request phrase matched.", "evidence": _evidence(text, m),
                        "evidence_basis": "explicit", "related_symptoms": []}]
        else:
            for a in aspects:
                a["sentiment"] = "neutral"
    elif aspects:
        mtype = "complaint"
    elif re.search(PRAISE_RE, text, flags=re.IGNORECASE):
        mtype = "praise"
    elif "?" in text:
        mtype = "question"
    else:
        mtype = "other"
    if mtype not in ("complaint", "feature_request"):
        aspects = []
    return {
        "message_type": mtype,
        "is_complaint": mtype == "complaint",
        "overall_sentiment": "negative" if mtype == "complaint" else "neutral",
        "aspects": aspects,
        "needs_review": False,
        "review_reason": "",
    }


def suggest_action(prompt: str) -> dict:
    m = re.search(r"Issue: (.+)", prompt)
    issue = m.group(1).strip() if m else "this issue"
    return {
        "title": f"[MOCK] Investigate: {issue}"[:150],
        "steps": ["[MOCK] Read the cited customer messages.",
                  "[MOCK] Confirm whether they describe one problem.", "[MOCK] Assign an owner."],
        "evidence_refs": [1],
        "rationale": "[MOCK] Generated from a fixed template, not by an AI model.",
        "limitations": "[MOCK] Template output; it does not analyse the evidence.",
        "uncertainty": "high",
        "needs_more_information": False,
    }


def respond(schema_name: str, user_prompt: str) -> str:
    if schema_name == "message_extraction":
        m = re.search(r"<customer_message>\n(.*)\n</customer_message>", user_prompt, flags=re.DOTALL)
        text = m.group(1) if m else user_prompt
        # The prompt neutralises angle brackets; restore so evidence matches the stored text.
        return json.dumps(extract(text.replace("‹", "<").replace("›", ">")))
    if schema_name == "action_suggestion":
        return json.dumps(suggest_action(user_prompt))
    raise ValueError(f"mock provider has no rule for schema {schema_name}")
