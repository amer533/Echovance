"""DRF API. Views only translate HTTP <-> services; business rules live in feedback.services."""
from __future__ import annotations

from datetime import timedelta

import django_filters
from django.conf import settings
from django.core.exceptions import PermissionDenied as DjangoPermissionDenied
from django.db.models import Prefetch
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from feedback.ai.client import LLMConfigError
from feedback.models import (ActionItem, AnalysisJob, AnalysisRun, CustomerMessage, GroupingSuggestion, ImportBatch,
                             IssueAlert, IssueMention, MessageAnalysis)
from feedback.services import actions as action_service
from feedback.services import corrections, grouping, jobs, review, stats
from feedback.services.analysis import AnalysisService
from feedback.services.imports import import_csv
from feedback.services.trends import SCOPE_SOURCES, TrendService

from . import serializers as s


class HasPerm(permissions.BasePermission):
    perm = ""

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.has_perm(self.perm))


class CanRunPipeline(HasPerm):
    perm = "feedback.run_pipeline"


class CanManage(HasPerm):
    perm = "feedback.manage_actions"


def _error(exc, code=status.HTTP_400_BAD_REQUEST):
    return Response({"detail": str(exc)}, status=code)


# ---------------- imports ----------------
class ImportBatchViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin,
                         viewsets.GenericViewSet):
    queryset = ImportBatch.objects.select_related("created_by")
    serializer_class = s.ImportBatchSerializer
    parser_classes = [MultiPartParser, FormParser]
    filterset_fields = ["status", "source"]

    def get_permissions(self):
        if self.action == "create":
            return [permissions.IsAuthenticated(), CanRunPipeline()]
        return super().get_permissions()

    def create(self, request, *args, **kwargs):
        upload = s.ImportUploadSerializer(data=request.data)
        upload.is_valid(raise_exception=True)
        f = upload.validated_data["file"]
        batch = import_csv(f, f.name, upload.validated_data["brand_account"], user=request.user,
                           data_classification=upload.validated_data["data_classification"])
        code = status.HTTP_400_BAD_REQUEST if batch.status == ImportBatch.Status.FAILED else status.HTTP_201_CREATED
        return Response(s.ImportBatchSerializer(batch).data, status=code)


# ---------------- messages ----------------
class MessageFilter(django_filters.FilterSet):
    created_after = django_filters.IsoDateTimeFilter(field_name="created_at", lookup_expr="gte")
    created_before = django_filters.IsoDateTimeFilter(field_name="created_at", lookup_expr="lt")
    issue = django_filters.NumberFilter(method="filter_issue")
    q = django_filters.CharFilter(field_name="text", lookup_expr="icontains")

    class Meta:
        model = CustomerMessage
        fields = ["author_role", "analysis_status", "is_thread_root", "import_batch"]

    def filter_issue(self, qs, name, value):
        return qs.filter(mentions__issue_id=value, mentions__is_active=True).distinct()


class MessageViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = CustomerMessage.objects.all()
    filterset_class = MessageFilter
    ordering_fields = ["created_at", "source_message_id"]

    def get_serializer_class(self):
        return s.CustomerMessageDetailSerializer if self.action == "retrieve" else s.CustomerMessageSerializer

    @action(detail=True, methods=["post"], permission_classes=[permissions.IsAuthenticated, CanRunPipeline])
    def correct(self, request, pk=None):
        body = s.CorrectionSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        d = body.validated_data
        try:
            corrections.correct_analysis(self.get_object(), request.user, message_type=d["message_type"],
                                         aspects=[dict(a) for a in d["aspects"]], note=d["note"])
        except action_service.ActionError as exc:
            return _error(exc)
        msg = self.get_queryset().get(pk=pk)
        return Response(s.CustomerMessageDetailSerializer(msg).data, status=status.HTTP_201_CREATED)

    def get_queryset(self):
        qs = super().get_queryset()
        if self.action == "retrieve":
            qs = qs.prefetch_related(Prefetch("analyses", MessageAnalysis.objects.prefetch_related("mentions__issue")))
        return qs


# ---------------- analysis ----------------
class AnalysisRunViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AnalysisRun.objects.all()
    serializer_class = s.AnalysisRunSerializer

    @action(detail=False, methods=["post"], permission_classes=[permissions.IsAuthenticated, CanRunPipeline])
    def run(self, request):
        body = s.RunAnalysisSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        cap = settings.ANALYSIS_MAX_BATCH_PER_REQUEST
        limit = body.validated_data["limit"]
        if limit > cap:
            return _error(f"limit may not exceed {cap} per request; use `manage.py run_analysis` for bulk runs.")
        try:
            service = AnalysisService()
        except LLMConfigError as exc:
            return _error(exc.detail, status.HTTP_503_SERVICE_UNAVAILABLE)
        if not service.client.is_mock:  # real provider: no bulk analysis through the API
            limit = min(limit, settings.LLM["REAL_MAX_PER_REQUEST"])
        run = service.run(limit=limit, message_ids=body.validated_data.get("message_ids"), user=request.user,
                          retry_failed=body.validated_data["retry_failed"])
        return Response(s.AnalysisRunSerializer(run).data, status=status.HTTP_201_CREATED)


# ---------------- issues ----------------
class IssueViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.UpdateModelMixin,
                   viewsets.GenericViewSet):
    serializer_class = s.IssueSerializer
    filterset_fields = ["category", "status"]
    ordering_fields = ["mention_count", "last_seen", "key"]
    ordering = ["-mention_count"]
    http_method_names = ["get", "patch", "head", "options"]

    def get_queryset(self):
        # lists hide issues without mentions in the scope; detail/update must still find every issue
        return stats.issues_with_counts(stats.clean_scope(self.request.query_params.get("scope")),
                                        include_empty=self.action != "list")

    def retrieve(self, request, *args, **kwargs):
        scope = stats.clean_scope(request.query_params.get("scope"))
        issue = self.get_object()
        data = self.get_serializer(issue).data
        data["scope"] = scope
        svc = TrendService(scope=scope)
        end = svc.latest_window_end()
        if end is not None:
            start = end - timedelta(days=svc.rules["WINDOW_DAYS"] * 8)
            data["weekly_trend"] = [{"start": w["start"], "end": w["end"], "count": w["count"],
                                     "analysed_volume": w["volume"]} for w in svc.weekly_series(issue, start, end)]
        else:
            data["weekly_trend"] = []
        recent = (IssueMention.objects.filter(issue=issue, is_active=True, source__in=SCOPE_SOURCES[scope])
                  .select_related("message", "issue")[:20])
        data["recent_mentions"] = s.IssueMentionSerializer(recent, many=True).data
        return Response(data)

    def partial_update(self, request, *args, **kwargs):
        body = s.IssueStatusSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        issue = self.get_object()
        try:
            review.set_issue_status(issue, request.user, body.validated_data["status"])
        except DjangoPermissionDenied as exc:
            raise PermissionDenied(str(exc))
        return Response(self.get_serializer(self.get_queryset().get(pk=issue.pk)).data)

    @action(detail=True, methods=["get"])
    def mentions(self, request, pk=None):
        issue = self.get_object()
        scope = stats.clean_scope(request.query_params.get("scope"))
        qs = IssueMention.objects.filter(issue=issue, is_active=True, source__in=SCOPE_SOURCES[scope]) \
            .select_related("message", "issue")
        page = self.paginate_queryset(qs)
        return self.get_paginated_response(s.IssueMentionSerializer(page, many=True).data)


# ---------------- alerts ----------------
class AlertViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = IssueAlert.objects.select_related("issue", "reviewed_by")
    filterset_fields = ["status", "rule", "issue", "data_scope", "is_stale"]
    ordering_fields = ["window_end", "current_count", "created_at"]

    def get_serializer_class(self):
        return s.IssueAlertDetailSerializer if self.action == "retrieve" else s.IssueAlertSerializer

    @action(detail=False, methods=["post"], permission_classes=[permissions.IsAuthenticated, CanRunPipeline])
    def detect(self, request):
        body = s.DetectSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        svc = TrendService(scope=body.validated_data["scope"])
        if body.validated_data["all_windows"]:
            results = svc.detect_all()
        else:
            end = body.validated_data.get("window_end") or svc.latest_window_end()
            if end is None:
                return _error("No analysed issue mentions yet; run analysis first.")
            results = [svc.detect(end)]
        return Response({"windows": [{
            "window_start": r.current.start, "window_end": r.current.end,
            "sufficient_data": r.sufficient_data, "insufficient_reason": r.insufficient_reason,
            "current_volume": r.current_volume, "baseline_volumes": r.baseline_volumes,
            "alerts_created": [a.pk for a in r.created], "alerts_existing": [a.pk for a in r.existing],
            "alerts_marked_stale": [a.pk for a in r.staled], "scope": r.scope,
        } for r in results]}, status=status.HTTP_200_OK)

    @action(detail=True, methods=["post"], permission_classes=[permissions.IsAuthenticated, CanManage])
    def review(self, request, pk=None):
        body = s.AlertReviewSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        try:
            alert = review.review_alert(self.get_object(), request.user, body.validated_data["status"],
                                        body.validated_data["note"])
        except action_service.ActionError as exc:
            return _error(exc)
        return Response(s.IssueAlertSerializer(alert).data)


# ---------------- actions ----------------
class ActionViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin,
                    mixins.UpdateModelMixin, viewsets.GenericViewSet):
    queryset = ActionItem.objects.select_related("issue", "created_by", "assignee", "decided_by")
    filterset_fields = ["status", "origin", "issue", "assignee"]
    ordering_fields = ["created_at", "updated_at", "due_date"]
    http_method_names = ["get", "post", "patch", "head", "options"]

    def get_serializer_class(self):
        return s.ActionDetailSerializer if self.action == "retrieve" else s.ActionItemSerializer

    def get_permissions(self):
        # check before body validation so non-managers learn nothing about which issues/alerts exist
        if self.action in ("create", "suggest"):
            return [permissions.IsAuthenticated(), CanManage()]
        return super().get_permissions()

    def create(self, request, *args, **kwargs):
        body = s.ActionCreateSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        try:
            act = action_service.create_manual(user=request.user, **body.validated_data)
        except DjangoPermissionDenied as exc:
            raise PermissionDenied(str(exc))
        except action_service.ActionError as exc:
            return _error(exc)
        return Response(s.ActionDetailSerializer(act).data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=["post"])
    def suggest(self, request):
        body = s.ActionSuggestSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        try:
            act = action_service.suggest_with_ai(user=request.user, issue=body.validated_data["issue"],
                                                 alert=body.validated_data.get("alert"))
        except DjangoPermissionDenied as exc:
            raise PermissionDenied(str(exc))
        except LLMConfigError as exc:
            return _error(exc.detail, status.HTTP_503_SERVICE_UNAVAILABLE)
        except action_service.SuggestionError as exc:
            return Response({"detail": exc.detail, "code": exc.code}, status=status.HTTP_502_BAD_GATEWAY)
        except action_service.ActionError as exc:
            return _error(exc)
        return Response(s.ActionDetailSerializer(act).data, status=status.HTTP_201_CREATED)

    def partial_update(self, request, *args, **kwargs):
        act = self.get_object()
        body = s.ActionUpdateSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        d = body.validated_data
        try:
            if "status" in d:
                act = action_service.transition(act, d["status"], request.user, d.get("note", ""))
            elif "assignee" in d:
                act = action_service.assign(act, request.user, d["assignee"], d.get("due_date"))
            else:
                act = action_service.edit_proposal(act, request.user, title=d.get("title"),
                                                   description=d.get("description"))
        except DjangoPermissionDenied as exc:
            raise PermissionDenied(str(exc))
        except action_service.TransitionError as exc:
            return _error(exc, status.HTTP_409_CONFLICT)
        except action_service.ActionError as exc:
            return _error(exc)
        return Response(s.ActionDetailSerializer(act).data)

    def update(self, request, *args, **kwargs):
        if not kwargs.get("partial"):
            return _error("Use PATCH.", status.HTTP_405_METHOD_NOT_ALLOWED)
        return self.partial_update(request, *args, **kwargs)


class DashboardSummaryView(APIView):
    def get(self, request):
        scope = stats.clean_scope(request.query_params.get("scope"))
        o = stats.overview(scope)
        lo, hi = o["date_range"]
        return Response({
            "scope": scope, "provenance": o["provenance"],
            "analysed_in_scope": o["analysed_in_scope"], "alerts_stale": o["alerts_stale"],
            "alerts_legacy": o["alerts_legacy"], "jobs_open": o["jobs_open"],
            "grouping_suggestions_pending": o["grouping_suggestions_pending"],
            "messages_total": o["messages_total"], "customer_roots": o["customer_roots"],
            "agent_replies": o["agent_replies"], "customer_followups": o["customer_followups"],
            "analysis_status": o["analysis_status"], "message_types": o["message_types"],
            "current_analyses_mock": o["current_analyses_mock"], "current_analyses_real": o["current_analyses_real"],
            "issues_with_mentions": o["issues_with_mentions"], "categories": o["categories"],
            "top_issues": s.IssueSerializer(o["top_issues"], many=True).data,
            "alerts_open": o["alerts_open"], "alerts_total": o["alerts_total"], "actions": o["actions"],
            "data_from": lo, "data_to": hi,
            "weekly_trend": [{"start": w["start"], "end": w["end"], "count": w["count"],
                              "analysed_volume": w["volume"]} for w in stats.overall_weekly_trend(scope=scope)],
        })


# ---------------- analysis jobs ----------------
class AnalysisJobViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin,
                         viewsets.GenericViewSet):
    queryset = AnalysisJob.objects.select_related("created_by", "authorized_by")
    serializer_class = s.AnalysisJobSerializer
    filterset_fields = ["status", "selection"]

    def get_permissions(self):
        if self.action in ("create", "cancel", "pause"):
            return [permissions.IsAuthenticated(), CanRunPipeline()]
        if self.action in ("authorize", "resume"):
            return [permissions.IsAuthenticated(), CanManage()]
        return super().get_permissions()

    def create(self, request, *args, **kwargs):
        body = s.JobCreateSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        try:
            job = jobs.create_job(user=request.user, **body.validated_data)
        except LLMConfigError as exc:
            return _error(exc.detail, status.HTTP_503_SERVICE_UNAVAILABLE)
        except action_service.ActionError as exc:
            return _error(exc)
        return Response(s.AnalysisJobSerializer(job).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["get"])
    def estimate(self, request, pk=None):
        job = self.get_object()
        est = jobs.estimate(job.provider, job.max_requests)
        est["cost_usd"] = None if est["cost_usd"] is None else str(est["cost_usd"])
        return Response(est)

    def _transition(self, fn):
        try:
            job = fn(self.get_object(), self.request.user)
        except DjangoPermissionDenied as exc:
            raise PermissionDenied(str(exc))
        except action_service.ActionError as exc:
            return _error(exc, status.HTTP_409_CONFLICT)
        return Response(s.AnalysisJobSerializer(job).data)

    @action(detail=True, methods=["post"])
    def authorize(self, request, pk=None):
        return self._transition(jobs.authorize)

    @action(detail=True, methods=["post"])
    def cancel(self, request, pk=None):
        return self._transition(jobs.cancel)

    @action(detail=True, methods=["post"])
    def pause(self, request, pk=None):
        return self._transition(jobs.pause)

    @action(detail=True, methods=["post"])
    def resume(self, request, pk=None):
        return self._transition(jobs.resume)


# ---------------- hybrid grouping review ----------------
class GroupingSuggestionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = GroupingSuggestion.objects.select_related("current_issue", "suggested_issue", "mention", "reviewed_by")
    serializer_class = s.GroupingSuggestionSerializer
    filterset_fields = ["status", "current_issue", "suggested_issue"]

    def _review(self, fn):
        body = s.ReviewNoteSerializer(data=self.request.data)
        body.is_valid(raise_exception=True)
        try:
            sugg = fn(self.get_object(), self.request.user, body.validated_data["note"])
        except DjangoPermissionDenied as exc:
            raise PermissionDenied(str(exc))
        except action_service.ActionError as exc:
            return _error(exc, status.HTTP_409_CONFLICT)
        return Response(s.GroupingSuggestionSerializer(sugg).data)

    @action(detail=True, methods=["post"], permission_classes=[permissions.IsAuthenticated, CanRunPipeline])
    def accept(self, request, pk=None):
        return self._review(grouping.accept_suggestion)

    @action(detail=True, methods=["post"], permission_classes=[permissions.IsAuthenticated, CanRunPipeline])
    def reject(self, request, pk=None):
        return self._review(grouping.reject_suggestion)
