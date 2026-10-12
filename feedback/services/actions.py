"""Human-in-the-loop action management.

Single source of truth for the action state machine and its permission policy; the API, web views
and tests all go through this service.

    proposed --approve--> approved --start--> in_progress --complete--> completed
        \\--reject--> rejected   \\--reject--> rejected

* Every action (AI or manual) starts as `proposed`; AI can never create approved/completed actions.
* approve / reject / assign / edit: users with feedback.manage_actions ("managers").
* start / complete: managers or the assigned user. `start` requires an assignee.
* Completing an action does NOT change the issue's status; issue resolution is a separate decision.
"""
from __future__ import annotations

from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.utils import timezone

from feedback.ai import actions as action_ai
from feedback.ai.client import LLMClient, LLMError, get_llm_client
from feedback.ai.privacy import sanitize_for_llm
from feedback.services.trends import SCOPE_SOURCES
from feedback.models import ActionEvent, ActionItem, Issue, IssueAlert, IssueMention

S = ActionItem.Status
TRANSITIONS: dict[str, set[str]] = {
    S.PROPOSED: {S.APPROVED, S.REJECTED},
    S.APPROVED: {S.IN_PROGRESS, S.REJECTED},
    S.IN_PROGRESS: {S.COMPLETED},
    S.COMPLETED: set(),
    S.REJECTED: set(),
}
MANAGE_PERM = "feedback.manage_actions"


class ActionError(ValueError):
    """Invalid input for an action operation (HTTP 400)."""


class TransitionError(ActionError):
    """The requested change is not allowed from the current state (HTTP 409)."""


class SuggestionError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def is_manager(user) -> bool:
    return bool(user and user.is_authenticated and user.has_perm(MANAGE_PERM))


def allowed_transitions(action: ActionItem, user) -> list[str]:
    """Transitions this user may perform now — used by the UI to render only valid buttons."""
    out = []
    for target in TRANSITIONS[action.status]:
        try:
            _check_transition(action, target, user)
            out.append(target)
        except (PermissionDenied, ActionError):
            pass
    return sorted(out)


def _check_transition(action: ActionItem, target: str, user) -> None:
    if target not in TRANSITIONS[action.status]:
        raise TransitionError(f"Cannot move an action from '{action.status}' to '{target}'.")
    if target in (S.APPROVED, S.REJECTED):
        if not is_manager(user):
            raise PermissionDenied("Only managers can approve or reject actions.")
    else:  # in_progress, completed
        if not (is_manager(user) or (action.assignee_id and action.assignee_id == user.pk)):
            raise PermissionDenied("Only a manager or the assignee can update progress.")
        if target == S.IN_PROGRESS and not action.assignee_id:
            raise ActionError("Assign the action to someone before starting it.")


def _log(action: ActionItem, actor, event: str, from_status: str = "", to_status: str = "", **details):
    ActionEvent.objects.create(action=action, actor=actor, event=event, from_status=from_status,
                               to_status=to_status, details=details)


@transaction.atomic
def create_manual(*, user, issue: Issue, title: str, description: str, rationale: str = "",
                  alert: IssueAlert | None = None) -> ActionItem:
    if not is_manager(user):
        raise PermissionDenied("Only managers can create actions.")
    if not title.strip() or not description.strip():
        raise ActionError("Title and description are required.")
    if alert is not None and alert.issue_id != issue.pk:
        raise ActionError("The alert belongs to a different issue.")
    action = ActionItem.objects.create(issue=issue, alert=alert, title=title.strip()[:160],
                                       description=description.strip()[:2000], rationale=rationale.strip()[:1000],
                                       origin=ActionItem.Origin.MANUAL, created_by=user, status=S.PROPOSED)
    _log(action, user, "created", to_status=S.PROPOSED, origin="manual")
    return action


def suggest_with_ai(*, user, issue: Issue, alert: IssueAlert | None = None,
                    client: LLMClient | None = None) -> ActionItem:
    """Ask the LLM for one proposed action. Never auto-approves. Raises SuggestionError on failure."""
    if not is_manager(user):
        raise PermissionDenied("Only managers can request AI suggestions.")
    if alert is not None and alert.issue_id != issue.pk:
        raise ActionError("The alert belongs to a different issue.")
    client = client or get_llm_client()
    # A real provider only ever sees real-scope evidence (llm + human), so mock data can never shape a
    # real proposal. The demo mock (labelled MOCK template output) may read any active evidence.
    sources = SCOPE_SOURCES["legacy_mixed" if client.is_mock else "real"]
    mentions = IssueMention.objects.filter(issue=issue, is_active=True, source__in=sources)
    if alert is not None:
        evidence_qs = alert.evidence.filter(is_active=True, source__in=sources).order_by("-message_created_at")
        stats = [f"- Alert rule: {alert.get_rule_display()} ({alert.rule_version}, data scope {alert.data_scope})",
                 f"- Window {alert.window_start:%Y-%m-%d} to {alert.window_last_day:%Y-%m-%d}: "
                 f"{alert.current_count} messages; baseline mean {alert.baseline_mean:.1f} per window"]
        if alert.is_stale:
            stats.append("- Note: this alert is STALE (its inputs changed after re-analysis)")
    else:
        evidence_qs = mentions.order_by("-message_created_at")
        stats = [f"- Supporting customer messages: {mentions.values('message_id').distinct().count()}"]
    cited_pool = list(evidence_qs[: action_ai.MAX_EVIDENCE_ITEMS])
    if not cited_pool:
        raise SuggestionError("no_evidence", "This issue has no supporting messages "
                              f"{'' if client.is_mock else 'from real AI or human analyses '}to base a suggestion on.")
    evidence = [sanitize_for_llm(m.evidence_text) for m in cited_pool]
    prompt = action_ai.build_user_prompt(issue.title, stats, evidence)
    try:
        resp = client.complete_json(action_ai.SYSTEM_PROMPT, prompt, "action_suggestion", action_ai.OUTPUT_SCHEMA,
                                    purpose="action_suggestion")
        suggestion = action_ai.validate_suggestion(resp.data, len(cited_pool))
    except LLMError as exc:
        raise SuggestionError(exc.code, f"The AI provider did not return a suggestion ({exc.code}). "
                                        "Nothing was created.") from None
    except action_ai.SuggestionSchemaError as exc:
        raise SuggestionError("schema_validation_failed", f"AI output was invalid ({exc}). Nothing was created.") from None
    rationale = suggestion.rationale
    if suggestion.needs_more_information:
        rationale = ("[AI flagged: evidence may be insufficient] " + rationale)[:1000]
    with transaction.atomic():
        action = ActionItem.objects.create(
            issue=issue, alert=alert, title=suggestion.title, description=suggestion.description,
            steps=suggestion.steps, rationale=rationale, limitations=suggestion.limitations,
            uncertainty=suggestion.uncertainty, origin=ActionItem.Origin.AI, ai_provider=resp.provider,
            ai_model=resp.model, ai_prompt_version=action_ai.PROMPT_VERSION, ai_is_mock=resp.is_mock,
            created_by=user, status=S.PROPOSED)
        action.evidence.set([cited_pool[r - 1] for r in suggestion.evidence_refs])
        _log(action, user, "ai_suggested", to_status=S.PROPOSED, provider=resp.provider, model=resp.model,
             is_mock=resp.is_mock, provider_latency_ms=resp.provider_latency_ms, queue_wait_ms=resp.queue_wait_ms,
             total_ms=resp.total_ms, evidence_refs=suggestion.evidence_refs,
             validation_notes=suggestion.notes)
    return action


@transaction.atomic
def transition(action: ActionItem, target: str, user, note: str = "") -> ActionItem:
    action = ActionItem.objects.select_for_update().get(pk=action.pk)
    _check_transition(action, target, user)
    note = (note or "").strip()
    if target == S.REJECTED and not note:
        raise ActionError("A reason is required to reject an action.")
    previous = action.status
    now = timezone.now()
    if target in (S.APPROVED, S.REJECTED):
        action.decided_by = user
        action.decided_at = now
        action.decision_note = note[:500]
    if target == S.COMPLETED:
        action.completed_at = now
        action.completion_note = note[:1000]
    action.status = target
    action.save()
    _log(action, user, "status_changed", previous, target, note=note[:500])
    return action


@transaction.atomic
def assign(action: ActionItem, user, assignee, due_date=None) -> ActionItem:
    action = ActionItem.objects.select_for_update().get(pk=action.pk)
    if not is_manager(user):
        raise PermissionDenied("Only managers can assign actions.")
    if action.status not in (S.APPROVED, S.IN_PROGRESS):
        raise TransitionError("Only approved or in-progress actions can be assigned.")
    if assignee is None or not assignee.is_active:
        raise ActionError("Choose an active user to assign.")
    old = action.assignee.username if action.assignee else None
    action.assignee = assignee
    action.due_date = due_date
    action.save()
    _log(action, user, "assigned", action.status, action.status, previous_assignee=old,
         assignee=assignee.username, due_date=due_date.isoformat() if due_date else None)
    return action


@transaction.atomic
def edit_proposal(action: ActionItem, user, *, title: str | None = None, description: str | None = None) -> ActionItem:
    action = ActionItem.objects.select_for_update().get(pk=action.pk)
    if not is_manager(user):
        raise PermissionDenied("Only managers can edit proposals.")
    if action.status != S.PROPOSED:
        raise TransitionError("Only proposed actions can be edited.")
    changed = {}
    if title is not None and title.strip() and title.strip() != action.title:
        changed["title"] = [action.title, title.strip()[:160]]
        action.title = title.strip()[:160]
    if description is not None and description.strip() and description.strip() != action.description:
        changed["description"] = "edited"
        action.description = description.strip()[:2000]
    if changed:
        action.save()
        _log(action, user, "edited", action.status, action.status, **changed)
    return action
