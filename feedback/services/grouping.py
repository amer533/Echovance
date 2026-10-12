"""Explainable baseline issue grouping.

An issue is identified by a stable key `category.subtype` from the controlled taxonomy.
Messages are grouped only when their validated aspects map to the same key, so two messages that
share words ("premium" charged twice vs. "premium" stopped playing) stay in different issues,
while different wordings of the same problem converge on one key. One message can support several
issues (one IssueMention per distinct key). Similarity-based regrouping is only ever *suggested*
(GroupingSuggestion) and applied after a person accepts it.
"""
from __future__ import annotations

import re

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.db import IntegrityError, transaction
from django.utils import timezone

from feedback import taxonomy
from feedback.ai.similarity import TfidfSimilarity, dense_cosine
from feedback.models import GroupingSuggestion, Issue, IssueMention

# Common spelling variants a provider might emit; mapped before validation.
SYNONYMS = {
    "billing": "billing_subscription",
    "subscription": "billing_subscription",
    "playback": "playback_performance",
    "performance": "playback_performance",
    "app_errors": "application_errors",
    "account": "account_access",
    "catalog": "content_catalog",
    "content": "content_catalog",
    "unclear": "other_unclear",
    "other": "other_unclear",
    "double_charge": "unexpected_charge",
    "duplicate_charge": "unexpected_charge",
    "crash": "app_crash",
    "login": "login_failure",
}


def normalise_code(value: str) -> str:
    code = re.sub(r"[\s/\-]+", "_", (value or "").strip().lower())
    code = re.sub(r"[^a-z0-9_]", "", code)
    return SYNONYMS.get(code, code)


def resolve_key(category: str, subtype: str) -> tuple[str, str]:
    """Normalise to a valid (category, subtype); unknown combinations fall back to other_unclear.other."""
    cat, sub = normalise_code(category), normalise_code(subtype)
    if taxonomy.is_valid(cat, sub):
        return cat, sub
    if cat in taxonomy.CATEGORIES:
        return cat, "other"
    return taxonomy.FALLBACK_CATEGORY, taxonomy.FALLBACK_SUBTYPE


def get_or_create_issue(category: str, subtype: str) -> Issue:
    cat, sub = resolve_key(category, subtype)
    issue, _ = Issue.objects.get_or_create(
        key=taxonomy.issue_key(cat, sub),
        defaults={"category": cat, "subtype": sub, "title": taxonomy.issue_title(cat, sub)},
    )
    return issue


# ---------------------------------------------------------------------------------------------
# Hybrid grouping: similarity-based REVIEW SUGGESTIONS on top of the taxonomy key.
#
# For a new mention, compare it with the mentions of OTHER issues in the SAME category (category
# constraint prevents cross-category false merges). If it is clearly closer to another issue than to
# its own (score >= MIN_SIMILARITY and margin >= MARGIN), a GroupingSuggestion is created for a
# human. Nothing is merged automatically; every decision keeps the original issue and reviewer.
# Score(issue) = mean of the top-3 similarities to that issue's mentions (robust to outliers).
# ---------------------------------------------------------------------------------------------
REAL_SOURCES = ("llm", "human")


def _doc(m: IssueMention) -> str:
    return f"{m.summary} {m.evidence_text}"


def _top3(sims: list[float]) -> float:
    top = sorted(sims, reverse=True)[:3]
    return sum(top) / len(top) if top else 0.0


def _embedding_vectors(mentions: list[IssueMention]) -> tuple[list, str]:
    """Gemini embeddings with a per-mention cache; every API call is budget-guarded."""
    from feedback.ai.similarity import GeminiEmbeddingProvider
    from feedback.services.llm_budget import RequestBudget
    cfg = settings.LLM
    model = cfg["GEMINI_EMBEDDING_MODEL"]
    missing = [m for m in mentions if not m.embedding or m.embedding_model != model]
    if missing:
        provider = GeminiEmbeddingProvider(cfg["GEMINI_API_KEY"], model, cfg["TIMEOUT_SECONDS"], RequestBudget())
        for i in range(0, len(missing), 100):
            chunk = missing[i:i + 100]
            for m, vec in zip(chunk, provider.embed([_doc(x) for x in chunk])):
                m.embedding, m.embedding_model = vec, model
                IssueMention.objects.filter(pk=m.pk).update(embedding=vec, embedding_model=model)
    return [m.embedding for m in mentions], f"gemini-embedding:{model}"


def similarity_scores(mention: IssueMention, candidates: list[IssueMention]) -> tuple[dict[int, list], str]:
    """Return {issue_id: [(similarity, mention_id), ...]} for candidates, and the method name."""
    method = settings.GROUPING["SIMILARITY"]
    docs = [mention] + candidates
    if method == "gemini_embedding":
        vecs, name = _embedding_vectors(docs)
        sim = dense_cosine
    else:
        tf = TfidfSimilarity()
        vecs, name, sim = tf.vectors([_doc(m) for m in docs]), tf.name, tf.cosine
    out: dict[int, list] = {}
    for m, v in zip(candidates, vecs[1:]):
        out.setdefault(m.issue_id, []).append((sim(vecs[0], v), m.pk))
    return out, name


def suggest_for_mentions(mention_ids: list[int]) -> list[GroupingSuggestion]:
    cfg = settings.GROUPING
    created = []
    mentions = IssueMention.objects.filter(pk__in=mention_ids, is_active=True, source__in=REAL_SOURCES) \
        .select_related("issue")
    for m in mentions:
        candidates = list(IssueMention.objects.filter(
            is_active=True, source__in=REAL_SOURCES, issue__category=m.issue.category).exclude(pk=m.pk)
            .exclude(message_id=m.message_id).select_related("issue"))
        if not candidates:
            continue
        scores, method = similarity_scores(m, candidates)
        own = _top3([s for s, _ in scores.get(m.issue_id, [])])
        best_issue, best = None, 0.0
        for issue_id, sims in scores.items():
            if issue_id == m.issue_id or len(sims) < cfg["MIN_ISSUE_MENTIONS"]:
                continue
            score = _top3([s for s, _ in sims])
            if score > best:
                best_issue, best = issue_id, score
        if best_issue and best >= cfg["MIN_SIMILARITY"] and best - own >= cfg["MARGIN"]:
            nearest = [mid for _, mid in sorted(scores[best_issue], reverse=True)[:3]]
            try:
                created.append(GroupingSuggestion.objects.create(
                    mention=m, current_issue_id=m.issue_id, suggested_issue_id=best_issue, method=method,
                    similarity_current=round(own, 4), similarity_suggested=round(best, 4),
                    nearest_mentions=nearest))
            except IntegrityError:
                pass  # already suggested
    expire_obsolete_suggestions()
    return created


def expire_obsolete_suggestions() -> int:
    return GroupingSuggestion.objects.filter(status="pending", mention__is_active=False).update(
        status="obsolete", reviewed_at=timezone.now())


def _can_review(user) -> bool:
    return bool(user and user.is_authenticated and user.has_perm("feedback.run_pipeline"))


@transaction.atomic
def accept_suggestion(suggestion: GroupingSuggestion, user, note: str = "") -> GroupingSuggestion:
    """Move the mention to the suggested issue (or merge into an existing mention there)."""
    if not _can_review(user):
        raise PermissionDenied("Reviewing grouping suggestions needs the run_pipeline permission.")
    s = GroupingSuggestion.objects.select_for_update().get(pk=suggestion.pk)
    if s.status != "pending":
        from .actions import TransitionError
        raise TransitionError("This suggestion was already reviewed.")
    m = IssueMention.objects.select_for_update().get(pk=s.mention_id)
    if not m.is_active or m.issue_id != s.current_issue_id:
        s.status = "obsolete"
        s.save(update_fields=["status"])
        return s
    now = timezone.now()
    duplicate = IssueMention.objects.filter(message_id=m.message_id, issue_id=s.suggested_issue_id,
                                            is_active=True).exists()
    if duplicate:  # the message already supports the target issue: deactivate instead of double counting
        m.is_active = False
    else:
        m.issue_id = s.suggested_issue_id
    m.original_issue_id = m.original_issue_id or s.current_issue_id
    m.regrouped_by, m.regrouped_at = user, now
    m.save()
    s.status, s.reviewed_by, s.reviewed_at, s.review_note = "accepted", user, now, note[:300]
    s.save()
    GroupingSuggestion.objects.filter(mention=m, status="pending").exclude(pk=s.pk).update(
        status="obsolete", reviewed_at=now)
    from .trends import refresh_alert_staleness
    transaction.on_commit(refresh_alert_staleness)
    return s


@transaction.atomic
def reject_suggestion(suggestion: GroupingSuggestion, user, note: str = "") -> GroupingSuggestion:
    if not _can_review(user):
        raise PermissionDenied("Reviewing grouping suggestions needs the run_pipeline permission.")
    s = GroupingSuggestion.objects.select_for_update().get(pk=suggestion.pk)
    if s.status != "pending":
        from .actions import TransitionError
        raise TransitionError("This suggestion was already reviewed.")
    s.status, s.reviewed_by, s.reviewed_at, s.review_note = "rejected", user, timezone.now(), note[:300]
    s.save()
    return s
