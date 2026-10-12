from django.contrib.auth import get_user_model
from rest_framework import serializers

from feedback import taxonomy
from feedback.models import (ActionEvent, ActionItem, AnalysisJob, AnalysisRun, CustomerMessage, GroupingSuggestion,
                             ImportBatch, Issue, IssueAlert, IssueMention, MessageAnalysis)

User = get_user_model()


class UserRefSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username"]


class ImportBatchSerializer(serializers.ModelSerializer):
    created_by = UserRefSerializer(read_only=True)

    class Meta:
        model = ImportBatch
        fields = ["id", "source", "original_filename", "file_sha256", "file_size_bytes", "brand_account",
                  "data_classification",
                  "status", "total_rows", "accepted_count", "duplicate_count", "rejected_count",
                  "customer_root_count", "agent_count", "errors", "failure_reason", "created_by",
                  "started_at", "finished_at"]
        read_only_fields = fields


class ImportUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    brand_account = serializers.CharField(max_length=64, default="SpotifyCares")
    data_classification = serializers.ChoiceField(choices=ImportBatch.DataClass.choices,
                                                  default=ImportBatch.DataClass.INTERNAL)


class IssueMentionSerializer(serializers.ModelSerializer):
    issue_key = serializers.CharField(source="issue.key", read_only=True)
    issue_title = serializers.CharField(source="issue.title", read_only=True)
    message_source_id = serializers.IntegerField(source="message.source_message_id", read_only=True)
    message_text = serializers.CharField(source="message.text", read_only=True)

    class Meta:
        model = IssueMention
        fields = ["id", "issue", "issue_key", "issue_title", "message", "message_source_id", "message_created_at",
                  "message_text", "sentiment", "summary", "evidence_text", "evidence_start", "evidence_end",
                  "is_active", "source", "original_issue", "regrouped_at"]


class MessageAnalysisSerializer(serializers.ModelSerializer):
    mentions = IssueMentionSerializer(many=True, read_only=True)

    class Meta:
        model = MessageAnalysis
        fields = ["id", "provider", "model_name", "prompt_version", "is_mock", "source", "job", "reused_from",
                  "redaction", "status", "is_current",
                  "message_type", "is_complaint", "review_reasons", "validated_output", "error_code",
                  "error_detail", "attempts", "latency_ms", "input_tokens", "output_tokens", "created_at",
                  "mentions"]


class CustomerMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerMessage
        fields = ["id", "source_message_id", "import_batch", "author_role", "author_hash", "brand_account",
                  "created_at", "text", "in_response_to_id", "is_thread_root", "analysis_status"]


class CustomerMessageDetailSerializer(CustomerMessageSerializer):
    analyses = MessageAnalysisSerializer(many=True, read_only=True)
    replies = serializers.SerializerMethodField()

    class Meta(CustomerMessageSerializer.Meta):
        fields = CustomerMessageSerializer.Meta.fields + ["response_ids", "analyses", "replies"]

    def get_replies(self, obj):
        qs = CustomerMessage.objects.filter(in_response_to_id=obj.source_message_id).order_by("created_at")[:10]
        return CustomerMessageSerializer(qs, many=True).data


class AnalysisRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisRun
        fields = ["id", "provider", "model_name", "prompt_version", "is_mock", "status", "requested",
                  "succeeded", "needs_review", "failed", "skipped", "stopped_reason", "started_at", "finished_at"]


class AnalysisJobSerializer(serializers.ModelSerializer):
    created_by = UserRefSerializer(read_only=True)
    authorized_by = UserRefSerializer(read_only=True)

    class Meta:
        model = AnalysisJob
        fields = ["id", "selection", "message_ids", "provider", "model_name", "prompt_version", "is_mock",
                  "max_messages", "max_requests", "max_input_tokens", "max_output_tokens", "max_cost_usd", "status",
                  "processed", "succeeded", "needs_review", "failed", "reused", "privacy_blocked", "requests_used",
                  "input_tokens_used", "output_tokens_used", "cost_usd", "stop_reason", "created_by",
                  "authorized_by", "authorized_at", "created_at", "started_at", "heartbeat_at", "finished_at"]
        read_only_fields = fields


class JobCreateSerializer(serializers.Serializer):
    selection = serializers.ChoiceField(choices=AnalysisJob.Selection.choices)
    max_messages = serializers.IntegerField(min_value=1, max_value=10_000)
    message_ids = serializers.ListField(child=serializers.IntegerField(min_value=1), required=False, max_length=1000)
    max_requests = serializers.IntegerField(min_value=1, required=False)
    max_cost_usd = serializers.DecimalField(max_digits=10, decimal_places=4, min_value=0, required=False)


class GroupingSuggestionSerializer(serializers.ModelSerializer):
    current_issue_key = serializers.CharField(source="current_issue.key", read_only=True)
    suggested_issue_key = serializers.CharField(source="suggested_issue.key", read_only=True)
    evidence_text = serializers.CharField(source="mention.evidence_text", read_only=True)
    reviewed_by = UserRefSerializer(read_only=True)

    class Meta:
        model = GroupingSuggestion
        fields = ["id", "mention", "evidence_text", "current_issue", "current_issue_key", "suggested_issue",
                  "suggested_issue_key", "method", "similarity_current", "similarity_suggested", "nearest_mentions",
                  "status", "reviewed_by", "reviewed_at", "review_note", "created_at"]
        read_only_fields = fields


class ReviewNoteSerializer(serializers.Serializer):
    note = serializers.CharField(max_length=300, required=False, allow_blank=True, default="")


class CorrectionAspectSerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=taxonomy.category_codes())
    subtype = serializers.ChoiceField(choices=taxonomy.subtype_codes())
    sentiment = serializers.ChoiceField(choices=list(taxonomy.SENTIMENTS), default="negative")
    evidence = serializers.CharField(max_length=300)
    summary = serializers.CharField(max_length=200, required=False, allow_blank=True, default="")


class CorrectionSerializer(serializers.Serializer):
    message_type = serializers.ChoiceField(choices=list(taxonomy.MESSAGE_TYPES))
    aspects = CorrectionAspectSerializer(many=True, required=False, default=list)
    note = serializers.CharField(max_length=200, required=False, allow_blank=True, default="")


class RunAnalysisSerializer(serializers.Serializer):
    limit = serializers.IntegerField(min_value=1, default=10)
    message_ids = serializers.ListField(child=serializers.IntegerField(min_value=1), required=False,
                                        max_length=100)
    retry_failed = serializers.BooleanField(default=True)


class IssueSerializer(serializers.ModelSerializer):
    mention_count = serializers.IntegerField(read_only=True)
    negative_count = serializers.IntegerField(read_only=True)
    first_seen = serializers.DateTimeField(read_only=True)
    last_seen = serializers.DateTimeField(read_only=True)
    open_alerts = serializers.IntegerField(read_only=True)

    class Meta:
        model = Issue
        fields = ["id", "key", "category", "subtype", "title", "status", "status_changed_at", "mention_count",
                  "negative_count", "first_seen", "last_seen", "open_alerts"]
        read_only_fields = [f for f in fields if f != "status"]


class IssueStatusSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=Issue.Status.choices)


class IssueAlertSerializer(serializers.ModelSerializer):
    issue_key = serializers.CharField(source="issue.key", read_only=True)
    issue_title = serializers.CharField(source="issue.title", read_only=True)
    reviewed_by = UserRefSerializer(read_only=True)

    class Meta:
        model = IssueAlert
        fields = ["id", "issue", "issue_key", "issue_title", "rule", "rule_version", "data_scope", "revision",
                  "is_stale", "stale_reason", "stale_at", "superseded_by", "input_fingerprint",
                  "window_start", "window_end",
                  "baseline_start", "baseline_end", "current_count", "baseline_total", "baseline_mean",
                  "growth_ratio", "current_share", "baseline_share", "current_window_volume",
                  "baseline_window_volumes", "thresholds", "explanation", "status", "reviewed_by", "reviewed_at",
                  "review_note", "created_at"]


class IssueAlertDetailSerializer(IssueAlertSerializer):
    evidence = IssueMentionSerializer(many=True, read_only=True)

    class Meta(IssueAlertSerializer.Meta):
        fields = IssueAlertSerializer.Meta.fields + ["evidence"]


class AlertReviewSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=[IssueAlert.Status.ACKNOWLEDGED, IssueAlert.Status.DISMISSED])
    note = serializers.CharField(max_length=500, required=False, allow_blank=True, default="")


class DetectSerializer(serializers.Serializer):
    window_end = serializers.DateField(required=False, help_text="Exclusive end date (UTC). Default: latest data.")
    all_windows = serializers.BooleanField(default=False)
    scope = serializers.ChoiceField(choices=["real", "demo"], default="real",
                                    help_text="real = LLM + human analyses; demo = mock analyses (never mixed)")


class ActionEventSerializer(serializers.ModelSerializer):
    actor = UserRefSerializer(read_only=True)

    class Meta:
        model = ActionEvent
        fields = ["id", "actor", "event", "from_status", "to_status", "details", "created_at"]


class ActionItemSerializer(serializers.ModelSerializer):
    issue_key = serializers.CharField(source="issue.key", read_only=True)
    created_by = UserRefSerializer(read_only=True)
    assignee = UserRefSerializer(read_only=True)
    decided_by = UserRefSerializer(read_only=True)

    class Meta:
        model = ActionItem
        fields = ["id", "issue", "issue_key", "alert", "title", "description", "steps", "rationale", "limitations",
                  "uncertainty", "evidence", "origin",
                  "ai_provider", "ai_model", "ai_prompt_version", "ai_is_mock", "status", "created_by",
                  "assignee", "due_date", "decided_by", "decided_at", "decision_note", "completed_at",
                  "completion_note", "created_at", "updated_at"]
        read_only_fields = fields


class ActionDetailSerializer(ActionItemSerializer):
    events = ActionEventSerializer(many=True, read_only=True)

    class Meta(ActionItemSerializer.Meta):
        fields = ActionItemSerializer.Meta.fields + ["events"]
        read_only_fields = fields


class ActionCreateSerializer(serializers.Serializer):
    issue = serializers.PrimaryKeyRelatedField(queryset=Issue.objects.all())
    alert = serializers.PrimaryKeyRelatedField(queryset=IssueAlert.objects.all(), required=False, allow_null=True)
    title = serializers.CharField(max_length=160)
    description = serializers.CharField(max_length=2000)
    rationale = serializers.CharField(max_length=1000, required=False, allow_blank=True, default="")


class ActionSuggestSerializer(serializers.Serializer):
    issue = serializers.PrimaryKeyRelatedField(queryset=Issue.objects.all())
    alert = serializers.PrimaryKeyRelatedField(queryset=IssueAlert.objects.all(), required=False, allow_null=True)


class ActionUpdateSerializer(serializers.Serializer):
    """PATCH body. Exactly one kind of change per request: status | assignment | proposal edit."""
    status = serializers.ChoiceField(choices=ActionItem.Status.choices, required=False)
    note = serializers.CharField(max_length=1000, required=False, allow_blank=True, default="")
    assignee = serializers.PrimaryKeyRelatedField(queryset=User.objects.filter(is_active=True), required=False)
    due_date = serializers.DateField(required=False, allow_null=True)
    title = serializers.CharField(max_length=160, required=False)
    description = serializers.CharField(max_length=2000, required=False)

    def validate(self, attrs):
        kinds = [bool("status" in attrs), bool("assignee" in attrs),
                 bool("title" in attrs or "description" in attrs)]
        if sum(kinds) != 1:
            raise serializers.ValidationError(
                "Send exactly one change: 'status' (+note), 'assignee' (+due_date), or 'title'/'description'.")
        return attrs
