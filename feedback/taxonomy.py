"""Issue taxonomy (version taxonomy-v1).

Derived from manual reading of random SpotifyCares thread roots during data profiling.
The same structure drives the LLM output schema, issue grouping,
the mock provider and the annotation guide, so there is exactly one source of truth.
"""
from __future__ import annotations

TAXONOMY_VERSION = "taxonomy-v1"

MESSAGE_TYPES = {
    "complaint": "Customer reports a problem, failure or dissatisfaction",
    "feature_request": "Customer asks for a new feature, content or improvement",
    "question": "Neutral how-to / information request without a reported problem",
    "praise": "Positive feedback",
    "other": "Unrelated, unintelligible or not enough content",
}

SENTIMENTS = ("negative", "neutral", "positive", "mixed")

CATEGORIES: dict[str, dict] = {
    "account_access": {
        "label": "Account Access",
        "subtypes": {
            "login_failure": "Cannot log in or account not recognised",
            "password_reset": "Password reset or email/username change problems",
            "unauthorized_access": "Unrecognised activity, hacked or unknown users on account",
            "region_restriction": "Country change or travel/abroad restriction",
            "other": "Other account access problem",
        },
    },
    "billing_subscription": {
        "label": "Billing / Subscription",
        "subtypes": {
            "unexpected_charge": "Unexpected, duplicate or wrong-amount charge",
            "refund_request": "Customer asks for money back",
            "premium_not_activated": "Paid but Premium/plan not active",
            "student_plan": "Student discount eligibility or verification",
            "family_plan": "Family plan invites, membership or address checks",
            "promo_offer": "Promotional offer or trial not applied",
            "payment_failure": "Payment method rejected or checkout fails",
            "partner_bundle": "Partner bundle (e.g. Hulu, carrier billing) problem",
            "cancellation": "Cannot cancel or downgrade subscription",
            "other": "Other billing problem",
        },
    },
    "playback_performance": {
        "label": "Playback / Performance",
        "subtypes": {
            "playback_interruption": "Songs skip, stop, pause or cut out",
            "offline_downloads": "Downloaded/offline music missing or not playing",
            "slow_or_unresponsive": "Slow loading, buffering or freezing",
            "device_connectivity": "Connect, Chromecast, speakers, car, TV or console",
            "service_outage": "Service down or unavailable for many users",
            "other": "Other playback problem",
        },
    },
    "application_errors": {
        "label": "Application Errors",
        "subtypes": {
            "app_crash": "App crashes or closes",
            "error_message": "Specific error code or error message shown",
            "library_data_loss": "Playlists, saved songs or library missing",
            "ui_change": "Unwanted UI change or removed feature",
            "other": "Other application error",
        },
    },
    "content_catalog": {
        "label": "Content / Catalog",
        "subtypes": {
            "missing_content": "Song, album or artist unavailable or greyed out",
            "country_availability": "Service not available in the customer's country",
            "metadata_error": "Wrong artist page, track info or lyrics",
            "other": "Other catalog problem",
        },
    },
    "recommendations_ads": {
        "label": "Recommendations / Ads",
        "subtypes": {
            "unwanted_recommendations": "Irrelevant recommendations or generated playlists",
            "ads_experience": "Ad frequency, length or broken ads",
            "autoplay_shuffle": "Autoplay or shuffle behaves unexpectedly",
            "other": "Other recommendation/ad problem",
        },
    },
    "customer_support": {
        "label": "Customer Support",
        "subtypes": {
            "no_response": "No reply or no way to contact support",
            "support_quality": "Unhelpful or unsatisfactory support",
            "other": "Other support problem",
        },
    },
    "feature_request": {
        "label": "Feature Requests",
        "subtypes": {
            "new_feature": "New feature or improvement",
            "platform_support": "Support for a device, platform or payment method",
            "other": "Other request",
        },
    },
    "other_unclear": {
        "label": "Other / Unclear",
        "subtypes": {"other": "Cannot be categorised from the message"},
    },
}

FALLBACK_CATEGORY = "other_unclear"
FALLBACK_SUBTYPE = "other"


def category_codes() -> list[str]:
    return list(CATEGORIES)


def subtype_codes() -> list[str]:
    return sorted({s for c in CATEGORIES.values() for s in c["subtypes"]})


def is_valid(category: str, subtype: str) -> bool:
    return category in CATEGORIES and subtype in CATEGORIES[category]["subtypes"]


def issue_key(category: str, subtype: str) -> str:
    return f"{category}.{subtype}"


def issue_title(category: str, subtype: str) -> str:
    cat = CATEGORIES[category]
    if subtype == "other":
        return f"{cat['label']}: other"
    return f"{cat['label']}: {cat['subtypes'][subtype]}"


def describe_for_prompt() -> str:
    lines = []
    for code, cat in CATEGORIES.items():
        lines.append(f"- {code} ({cat['label']}):")
        for sub, desc in cat["subtypes"].items():
            lines.append(f"    - {sub}: {desc}")
    return "\n".join(lines)
