"""Server-rendered management UI. Like the API, these views delegate all rules to feedback.services."""
from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.contrib import messages as flash
from django.contrib.auth.decorators import login_required, permission_required
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator
from django.db.models import Prefetch, Q
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from feedback import taxonomy
from feedback.ai.client import LLMConfigError
from feedback.models import (ActionItem, AnalysisJob, AnalysisRun, CustomerMessage, GroupingSuggestion, ImportBatch,
                             Issue, IssueAlert, IssueMention)
from feedback.services import actions as action_service
from feedback.services import corrections, evaluation, grouping, jobs, llm_budget, review, stats
from feedback.services.analysis import AnalysisService
from feedback.services.imports import import_csv
from feedback.services.trends import SCOPE_SOURCES, TrendService

from . import forms


def _page(request, qs, per_page=25):
    return Paginator(qs, per_page).get_page(request.GET.get("page"))


def _query_without_page(request) -> str:
    q = request.GET.copy()
    q.pop("page", None)
    return q.urlencode()


def chart_bars(series: list[dict], highlight: tuple | None = None) -> dict:
    """Pre-compute SVG bar geometry for a weekly series (keeps templates logic-free)."""
    if not series:
        return {"bars": [], "max": 0}
    peak = max(max(p["count"] for p in series), 1)
    width, gap = 100 / len(series), 0.6
    bars = []
    for i, p in enumerate(series):
        h = p["count"] / peak * 100
        is_hl = bool(highlight and p["start"] == highlight[0] and p["end"] == highlight[1])
        bars.append({**p, "x": i * width + gap / 2, "w": width - gap, "h": h, "y": 100 - h, "highlight": is_hl,
                     "share_pct": round(p["count"] / p["volume"] * 100, 1) if p.get("volume") else None})
    return {"bars": bars, "max": peak}


def highlight_segments(text: str, mentions) -> list[tuple[str, bool]]:
    """Split text into (segment, is_evidence) using verified evidence offsets; overlaps are merged."""
    spans = sorted((m.evidence_start, m.evidence_end) for m in mentions if m.evidence_end <= len(text))
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, cursor = [], 0
    for a, b in merged:
        if a > cursor:
            out.append((text[cursor:a], False))
        out.append((text[a:b], True))
        cursor = b
    if cursor < len(text):
        out.append((text[cursor:], False))
    return out


# ---------------- dashboard ----------------
@login_required
def dashboard(request):
    scope = stats.clean_scope(request.GET.get("scope"))
    o = stats.overview(scope)
    trend = stats.overall_weekly_trend(scope=scope)
    peak_cat = max((c["count"] for c in o["categories"]), default=0)
    for c in o["categories"]:
        c["pct"] = c["count"] / peak_cat * 100 if peak_cat else 0
    return render(request, "feedback/dashboard.html", {
        "o": o, "trend": chart_bars(trend), "trend_rows": trend,
        "scope": scope,
        "open_alerts": IssueAlert.objects.filter(status=IssueAlert.Status.OPEN, data_scope=scope, is_stale=False)
                                         .select_related("issue")[:6],
        "proposed_actions": ActionItem.objects.filter(status=ActionItem.Status.PROPOSED).select_related("issue")[:6],
        "my_actions": (ActionItem.objects.filter(assignee=request.user, status__in=["approved", "in_progress"])
                       .select_related("issue")[:6]),
        "action_rows": [(v, label, o["actions"][v]) for v, label in ActionItem.Status.choices],
    })


# ---------------- imports ----------------
@login_required
def import_list(request):
    form = forms.ImportForm()
    if request.method == "POST":
        if not request.user.has_perm("feedback.run_pipeline"):
            raise PermissionDenied
        form = forms.ImportForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data["file"]
            batch = import_csv(f, f.name, form.cleaned_data["brand_account"], user=request.user,
                               data_classification=form.cleaned_data["data_classification"])
            if batch.status == ImportBatch.Status.FAILED:
                flash.error(request, f"Import failed: {batch.failure_reason}")
            else:
                flash.success(request, f"Imported {batch.accepted_count} new rows "
                                       f"({batch.duplicate_count} duplicates, {batch.rejected_count} rejected).")
            return redirect("import-detail", pk=batch.pk)
    return render(request, "feedback/imports.html", {
        "form": form, "max_upload": settings.CSV_MAX_UPLOAD_BYTES, "max_rows": settings.CSV_MAX_ROWS, "page": _page(request, ImportBatch.objects.select_related("created_by"))})


@login_required
def import_detail(request, pk):
    batch = get_object_or_404(ImportBatch, pk=pk)
    return render(request, "feedback/import_detail.html", {"batch": batch})


# ---------------- messages & analysis ----------------
@login_required
def message_list(request):
    qs = CustomerMessage.objects.all()
    f = request.GET
    role = f.get("role", "root")
    if role == "root":
        qs = qs.filter(author_role="customer", is_thread_root=True)
    elif role in ("customer", "agent"):
        qs = qs.filter(author_role=role)
    if f.get("status") in CustomerMessage.AnalysisStatus.values:
        qs = qs.filter(analysis_status=f["status"])
    if f.get("issue", "").isdigit():
        qs = qs.filter(mentions__issue_id=int(f["issue"]), mentions__is_active=True).distinct()
    if f.get("q"):
        qs = qs.filter(text__icontains=f["q"][:100])
    if f.get("batch", "").isdigit():
        qs = qs.filter(import_batch_id=int(f["batch"]))
    qs = qs.prefetch_related(Prefetch("mentions", IssueMention.objects.filter(is_active=True).select_related("issue"),
                                      to_attr="active_mentions"))
    return render(request, "feedback/messages.html", {
        "page": _page(request, qs), "filters": f, "role": role, "query": _query_without_page(request),
        "statuses": CustomerMessage.AnalysisStatus.choices,
        "issues": Issue.objects.order_by("key"), "run_form": forms.RunAnalysisForm(),
        "runs": AnalysisRun.objects.all()[:5],
    })


@login_required
def message_detail(request, pk):
    msg = get_object_or_404(CustomerMessage, pk=pk)
    analyses = msg.analyses.prefetch_related("mentions__issue").order_by("-is_current", "-created_at")
    current = next((a for a in analyses if a.is_current), None)
    mentions = list(current.mentions.all()) if current else []
    thread = []
    if msg.in_response_to_id:
        parent = CustomerMessage.objects.filter(source_message_id=msg.in_response_to_id).first()
        if parent:
            thread.append(("In response to", parent))
    for reply in CustomerMessage.objects.filter(in_response_to_id=msg.source_message_id).order_by("created_at")[:10]:
        thread.append(("Reply", reply))
    return render(request, "feedback/message_detail.html", {
        "msg": msg, "analyses": analyses, "current": current, "mentions": mentions, "thread": thread,
        "segments": highlight_segments(msg.text, mentions),
        "correction_form": forms.CorrectionForm(initial={"message_type": current.message_type if current else ""})})


@login_required
@require_POST
def message_correct(request, pk):
    msg = get_object_or_404(CustomerMessage, pk=pk)
    form = forms.CorrectionForm(request.POST)
    if not form.is_valid():
        flash.error(request, "Please correct the form: " + "; ".join(
            f"{k}: {' '.join(v)}" for k, v in form.errors.items()))
        return redirect("message-detail", pk=pk)
    try:
        corrections.correct_analysis(msg, request.user, message_type=form.cleaned_data["message_type"],
                                     aspects=form.aspects(), note=form.cleaned_data["note"])
        flash.success(request, "Correction saved as a human analysis; the previous analysis is kept in history.")
    except action_service.ActionError as exc:
        flash.error(request, str(exc))
    return redirect("message-detail", pk=pk)


@login_required
@permission_required("feedback.run_pipeline", raise_exception=True)
@require_POST
def run_analysis(request):
    form = forms.RunAnalysisForm(request.POST)
    if not form.is_valid():
        flash.error(request, "Choose between 1 and the per-request limit of messages.")
        return redirect("messages")
    try:
        service = AnalysisService()
        limit = form.cleaned_data["limit"]
        if not service.client.is_mock:  # real provider: no bulk analysis from the UI
            limit = min(limit, settings.LLM["REAL_MAX_PER_REQUEST"])
        run = service.run(limit=limit, user=request.user)
    except LLMConfigError as exc:
        flash.error(request, f"LLM is not configured: {exc.detail}")
        return redirect("messages")
    label = "MOCK provider (not real AI)" if run.is_mock else f"{run.provider}/{run.model_name}"
    level = flash.warning if (run.failed or run.stopped_reason) else flash.success
    level(request, f"Analysis run #{run.pk} with {label}: {run.succeeded} succeeded, "
                   f"{run.needs_review} need review, {run.failed} failed of {run.requested}."
                   + (f" {run.stopped_reason}" if run.stopped_reason else ""))
    return redirect("messages")


# ---------------- issues ----------------
@login_required
def issue_list(request):
    scope = stats.clean_scope(request.GET.get("scope"))
    qs = stats.issues_with_counts(scope)
    cat = request.GET.get("category")
    if cat in taxonomy.CATEGORIES:
        qs = qs.filter(category=cat)
    sort = request.GET.get("sort", "-mention_count")
    if sort not in ("-mention_count", "-last_seen", "key", "-open_alerts"):
        sort = "-mention_count"
    return render(request, "feedback/issues.html", {
        "issues": qs.order_by(sort, "key"), "categories": taxonomy.CATEGORIES, "category": cat, "sort": sort,
        "scope": scope})


@login_required
def issue_detail(request, pk):
    scope = stats.clean_scope(request.GET.get("scope"))
    issue = get_object_or_404(stats.issues_with_counts(scope, include_empty=True), pk=pk)
    svc = TrendService(scope=scope)
    end = svc.latest_window_end()
    series = []
    if end:
        series = svc.weekly_series(issue, end - timedelta(days=svc.rules["WINDOW_DAYS"] * 8), end)
    mentions = (IssueMention.objects.filter(issue=issue, is_active=True, source__in=SCOPE_SOURCES[scope])
                .select_related("message", "analysis").order_by("-message_created_at"))
    return render(request, "feedback/issue_detail.html", {
        "issue": issue, "trend": chart_bars(series), "trend_rows": series,
        "page": _page(request, mentions, 15),
        "alerts": issue.alerts.order_by("-window_end", "-revision")[:10], "scope": scope,
        "suggestions": GroupingSuggestion.objects.filter(status="pending").filter(
            models_q_current_or_suggested(issue)).select_related("current_issue", "suggested_issue", "mention")[:10],
        "actions": issue.actions.select_related("assignee")[:10],
        "status_form": forms.IssueStatusForm(initial={"status": issue.status}),
        "action_form": forms.ActionCreateForm(),
    })


@login_required
@require_POST
def issue_status(request, pk):
    issue = get_object_or_404(Issue, pk=pk)
    form = forms.IssueStatusForm(request.POST)
    if form.is_valid():
        review.set_issue_status(issue, request.user, form.cleaned_data["status"])
        flash.success(request, "Issue status updated. (This is a manager decision; it is not inferred from actions.)")
    return redirect("issue-detail", pk=pk)


# ---------------- alerts ----------------
@login_required
def alert_list(request):
    scope = request.GET.get("scope", "real")
    if scope not in IssueAlert.Scope.values:
        scope = "real"
    qs = IssueAlert.objects.select_related("issue", "reviewed_by").filter(data_scope=scope)
    st = request.GET.get("status", "open")
    if st in IssueAlert.Status.values:
        qs = qs.filter(status=st)
    if request.GET.get("stale") != "1":
        qs = qs.filter(is_stale=False)
    svc = TrendService(scope="demo" if scope == "demo" else "real")
    return render(request, "feedback/alerts.html", {
        "page": _page(request, qs), "status": st, "statuses": IssueAlert.Status.choices, "scope": scope,
        "show_stale": request.GET.get("stale") == "1", "scopes": IssueAlert.Scope.choices,
        "detect_form": forms.DetectForm(initial={"scope": "demo" if scope == "demo" else "real"}),
        "rules": svc.rules, "latest_end": svc.latest_window_end(),
        "query": f"scope={scope}&status={st}" + ("&stale=1" if request.GET.get("stale") == "1" else "")})


@login_required
@permission_required("feedback.run_pipeline", raise_exception=True)
@require_POST
def detect_alerts(request):
    form = forms.DetectForm(request.POST)
    if not form.is_valid():
        flash.error(request, "Invalid date.")
        return redirect("alerts")
    svc = TrendService(scope=form.cleaned_data["scope"])
    if form.cleaned_data["all_windows"]:
        results = svc.detect_all()
    else:
        end = form.cleaned_data["window_end"] or svc.latest_window_end()
        if end is None:
            flash.error(request, "No analysed issue mentions yet — run analysis first.")
            return redirect("alerts")
        results = [svc.detect(end)]
    created = sum(len(r.created) for r in results)
    existing = sum(len(r.existing) for r in results)
    staled = sum(len(r.staled) for r in results)
    insufficient = [r.current.label() for r in results if not r.sufficient_data]
    msg = (f"[{svc.scope} data] Evaluated {len(results)} window(s): {created} new alert(s), {existing} already "
           f"existed, {staled} marked stale.")
    if insufficient:
        msg += f" Insufficient data for: {', '.join(insufficient)}."
    flash.info(request, msg)
    return redirect(f"{reverse('alerts')}?scope={svc.scope}")


@login_required
def alert_detail(request, pk):
    alert = get_object_or_404(IssueAlert.objects.select_related("issue", "reviewed_by", "superseded_by"), pk=pk)
    svc = TrendService(scope="demo" if alert.data_scope == "demo" else "real")
    days = timedelta(days=svc.rules["WINDOW_DAYS"])
    end = max(alert.window_end, svc.latest_window_end() or alert.window_end)
    series = svc.weekly_series(alert.issue, alert.baseline_start, min(end, alert.window_end + 3 * days))
    return render(request, "feedback/alert_detail.html", {
        "alert": alert, "evidence": alert.evidence.select_related("message").order_by("-message_created_at"),
        "trend": chart_bars(series, (alert.window_start, alert.window_end)), "trend_rows": series,
        "review_form": forms.AlertReviewForm(), "actions": alert.actions.all()})


@login_required
@require_POST
def alert_review(request, pk):
    alert = get_object_or_404(IssueAlert, pk=pk)
    form = forms.AlertReviewForm(request.POST)
    if form.is_valid():
        try:
            review.review_alert(alert, request.user, form.cleaned_data["status"], form.cleaned_data["note"])
            flash.success(request, "Alert review recorded.")
        except action_service.ActionError as exc:
            flash.error(request, str(exc))
    return redirect("alert-detail", pk=pk)


# ---------------- actions ----------------
@login_required
def action_list(request):
    qs = ActionItem.objects.select_related("issue", "assignee", "created_by")
    st = request.GET.get("status", "")
    if st in ActionItem.Status.values:
        qs = qs.filter(status=st)
    if request.GET.get("mine"):
        qs = qs.filter(assignee=request.user)
    return render(request, "feedback/actions.html", {
        "page": _page(request, qs), "status": st, "statuses": ActionItem.Status.choices,
        "mine": request.GET.get("mine"), "query": _query_without_page(request)})


@login_required
@require_POST
def action_create(request, issue_pk):
    issue = get_object_or_404(Issue, pk=issue_pk)
    alert = None
    if request.POST.get("alert", "").isdigit():
        alert = get_object_or_404(IssueAlert, pk=int(request.POST["alert"]))
    form = forms.ActionCreateForm(request.POST)
    if not form.is_valid():
        flash.error(request, "Title and description are required.")
        return redirect("issue-detail", pk=issue.pk)
    try:
        act = action_service.create_manual(user=request.user, issue=issue, alert=alert, **form.cleaned_data)
    except action_service.ActionError as exc:
        flash.error(request, str(exc))
        return redirect("issue-detail", pk=issue.pk)
    flash.success(request, "Action proposed. It must be approved before work starts.")
    return redirect("action-detail", pk=act.pk)


@login_required
@require_POST
def action_suggest(request, issue_pk):
    issue = get_object_or_404(Issue, pk=issue_pk)
    alert = None
    if request.POST.get("alert", "").isdigit():
        alert = get_object_or_404(IssueAlert, pk=int(request.POST["alert"]))
    back = redirect("alert-detail", pk=alert.pk) if alert else redirect("issue-detail", pk=issue.pk)
    try:
        act = action_service.suggest_with_ai(user=request.user, issue=issue, alert=alert)
    except LLMConfigError as exc:
        flash.error(request, f"LLM is not configured: {exc.detail}")
        return back
    except action_service.SuggestionError as exc:
        flash.error(request, exc.detail)
        return back
    except action_service.ActionError as exc:
        flash.error(request, str(exc))
        return back
    flash.success(request, "AI proposal created with status 'Proposed'. Review it before approving.")
    return redirect("action-detail", pk=act.pk)


EVENT_LABELS = {"created": "Proposed", "ai_suggested": "AI proposal created", "status_changed": "Status changed",
                "assigned": "Assigned", "edited": "Proposal edited"}


def _label_event(event):
    event.event_label = EVENT_LABELS.get(event.event, event.event)
    return event


@login_required
def action_detail(request, pk):
    act = get_object_or_404(ActionItem.objects.select_related("issue", "alert", "assignee", "decided_by",
                                                              "created_by"), pk=pk)
    svc = TrendService()
    days = timedelta(days=svc.rules["WINDOW_DAYS"])
    end = svc.latest_window_end()
    series, hl = [], None
    if end:
        anchor_end = act.alert.window_end if act.alert else end
        start = (act.alert.baseline_start if act.alert else end - 8 * days)
        series = svc.weekly_series(act.issue, start, min(end, anchor_end + 4 * days) if act.alert else end)
        hl = (act.alert.window_start, act.alert.window_end) if act.alert else None
    return render(request, "feedback/action_detail.html", {
        "act": act, "events": [_label_event(e) for e in act.events.select_related("actor")],
        "evidence": act.evidence.select_related("message").order_by("-message_created_at"),
        "can_assign": act.status in (ActionItem.Status.APPROVED, ActionItem.Status.IN_PROGRESS),
        "allowed": action_service.allowed_transitions(act, request.user),
        "transition_form": forms.TransitionForm(), "assign_form": forms.AssignForm(
            initial={"assignee": act.assignee_id, "due_date": act.due_date}),
        "edit_form": forms.EditProposalForm(initial={"title": act.title, "description": act.description}),
        "trend": chart_bars(series, hl), "trend_rows": series,
        "status_labels": dict(ActionItem.Status.choices),
    })


@login_required
@require_POST
def action_update(request, pk):
    act = get_object_or_404(ActionItem, pk=pk)
    kind = request.POST.get("kind")
    try:
        if kind == "transition":
            form = forms.TransitionForm(request.POST)
            if form.is_valid():
                action_service.transition(act, form.cleaned_data["status"], request.user, form.cleaned_data["note"])
                flash.success(request, "Status updated.")
        elif kind == "assign":
            form = forms.AssignForm(request.POST)
            if form.is_valid():
                action_service.assign(act, request.user, form.cleaned_data["assignee"], form.cleaned_data["due_date"])
                flash.success(request, "Assignment saved.")
        elif kind == "edit":
            form = forms.EditProposalForm(request.POST)
            if form.is_valid():
                action_service.edit_proposal(act, request.user, **form.cleaned_data)
                flash.success(request, "Proposal updated.")
        else:
            flash.error(request, "Unknown update.")
            return redirect("action-detail", pk=pk)
        if not form.is_valid():
            flash.error(request, "Please correct the form: " + "; ".join(
                f"{k}: {' '.join(v)}" for k, v in form.errors.items()))
    except action_service.ActionError as exc:
        flash.error(request, str(exc))
    return redirect("action-detail", pk=pk)


def models_q_current_or_suggested(issue):
    return Q(current_issue=issue) | Q(suggested_issue=issue)


# ---------------- analysis jobs ----------------
@login_required
def job_list(request):
    form = forms.JobForm()
    if request.method == "POST":
        if not request.user.has_perm("feedback.run_pipeline"):
            raise PermissionDenied
        form = forms.JobForm(request.POST)
        if form.is_valid():
            try:
                job = jobs.create_job(user=request.user, selection=form.cleaned_data["selection"],
                                      max_messages=form.cleaned_data["max_messages"],
                                      message_ids=form.parsed_message_ids(),
                                      max_requests=form.cleaned_data.get("max_requests"),
                                      max_cost_usd=form.cleaned_data.get("max_cost_usd"))
                flash.success(request, f"Job #{job.pk} created as a draft. A manager must authorise it before it runs.")
                return redirect("job-detail", pk=job.pk)
            except LLMConfigError as exc:
                flash.error(request, f"LLM is not configured: {exc.detail}")
            except action_service.ActionError as exc:
                flash.error(request, str(exc))
    provider = settings.LLM["PROVIDER"]
    return render(request, "feedback/jobs.html", {
        "form": form, "page": _page(request, AnalysisJob.objects.select_related("created_by", "authorized_by")),
        "budget": llm_budget.status(), "per_message": jobs.per_message_estimate(provider),
        "provenance": stats.provenance()})


@login_required
def job_detail(request, pk):
    job = get_object_or_404(AnalysisJob.objects.select_related("created_by", "authorized_by"), pk=pk)
    return render(request, "feedback/job_detail.html", {
        "job": job, "estimate": jobs.estimate(job.provider, job.max_requests), "budget": llm_budget.status(),
        "requests": job.requests.order_by("-created_at")[:25],
        "auto_refresh": job.status in ("authorized", "running")})


@login_required
@require_POST
def job_update(request, pk):
    job = get_object_or_404(AnalysisJob, pk=pk)
    ops = {"authorize": jobs.authorize, "cancel": jobs.cancel, "pause": jobs.pause, "resume": jobs.resume}
    op = ops.get(request.POST.get("op"))
    if op is None:
        flash.error(request, "Unknown operation.")
    else:
        try:
            op(job, request.user)
            flash.success(request, f"Job #{job.pk}: {request.POST['op']} done.")
        except action_service.ActionError as exc:
            flash.error(request, str(exc))
    return redirect("job-detail", pk=pk)


# ---------------- grouping review ----------------
@login_required
def grouping_review(request):
    st = request.GET.get("status", "pending")
    qs = GroupingSuggestion.objects.select_related("current_issue", "suggested_issue", "mention", "mention__message",
                                                   "reviewed_by")
    if st in GroupingSuggestion.Status.values:
        qs = qs.filter(status=st)
    page = _page(request, qs)
    nearest_ids = {i for sgg in page for i in sgg.nearest_mentions}
    nearest = IssueMention.objects.in_bulk(list(nearest_ids))
    for sgg in page:
        sgg.nearest = [nearest[i] for i in sgg.nearest_mentions if i in nearest]
    return render(request, "feedback/grouping.html", {"page": page, "status": st,
                                                       "statuses": GroupingSuggestion.Status.choices,
                                                       "method": settings.GROUPING["SIMILARITY"]})


@login_required
@require_POST
def grouping_decide(request, pk):
    sugg = get_object_or_404(GroupingSuggestion, pk=pk)
    fn = grouping.accept_suggestion if request.POST.get("decision") == "accept" else grouping.reject_suggestion
    try:
        fn(sugg, request.user, request.POST.get("note", ""))
        flash.success(request, "Grouping decision recorded.")
    except action_service.ActionError as exc:
        flash.error(request, str(exc))
    return redirect("grouping")


# ---------------- blind labelling ----------------
@login_required
def label_next(request):
    if not evaluation.can_label(request.user):
        raise PermissionDenied
    if request.method == "POST":
        message_id = request.POST.get("message", "")
        if not message_id.isdigit():
            raise Http404("Unknown message.")
        msg = get_object_or_404(CustomerMessage, pk=int(message_id))
        form = forms.LabelForm(request.POST)
        if form.is_valid():
            try:
                evaluation.save_label(request.user, msg, form.cleaned_data["message_type"],
                                      form.cleaned_data["issue_keys"], form.cleaned_data["notes"])
                flash.success(request, "Label saved.")
            except ValueError as exc:
                flash.error(request, str(exc))
        else:
            flash.error(request, "Please correct the form: " + "; ".join(
                f"{k}: {' '.join(v)}" for k, v in form.errors.items()))
        return redirect("label")
    msg = evaluation.next_for_labelling(request.user)
    total = len(evaluation.eval_sample_ids())
    mine = request.user and evaluation.EvaluationLabel.objects.filter(annotator=request.user, role="annotator").count()
    return render(request, "feedback/label.html", {"msg": msg, "form": forms.LabelForm(), "total": total,
                                                    "mine": mine, "taxonomy": taxonomy.CATEGORIES})
