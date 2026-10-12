from django.contrib import admin

from .models import (ActionEvent, ActionItem, AnalysisRun, CustomerMessage, ImportBatch, Issue, IssueAlert,
                     IssueMention, MessageAnalysis)


@admin.register(ImportBatch)
class ImportBatchAdmin(admin.ModelAdmin):
    list_display = ["id", "original_filename", "status", "total_rows", "accepted_count", "duplicate_count",
                    "rejected_count", "started_at"]
    readonly_fields = [f.name for f in ImportBatch._meta.fields]


@admin.register(CustomerMessage)
class CustomerMessageAdmin(admin.ModelAdmin):
    list_display = ["source_message_id", "author_role", "is_thread_root", "analysis_status", "created_at"]
    list_filter = ["author_role", "analysis_status", "is_thread_root"]
    search_fields = ["text", "source_message_id"]


@admin.register(MessageAnalysis)
class MessageAnalysisAdmin(admin.ModelAdmin):
    list_display = ["id", "message", "provider", "model_name", "prompt_version", "is_mock", "status", "is_current"]
    list_filter = ["status", "is_mock", "provider", "prompt_version"]


admin.site.register(AnalysisRun)
admin.site.register(Issue)
admin.site.register(IssueMention)
admin.site.register(IssueAlert)


class ActionEventInline(admin.TabularInline):
    model = ActionEvent
    extra = 0
    readonly_fields = ["actor", "event", "from_status", "to_status", "details", "created_at"]
    can_delete = False


@admin.register(ActionItem)
class ActionItemAdmin(admin.ModelAdmin):
    list_display = ["id", "title", "status", "origin", "assignee", "created_at"]
    list_filter = ["status", "origin"]
    # Status changes must go through ActionService so the audit trail and rules apply.
    readonly_fields = ["status", "decided_by", "decided_at", "completed_at", "origin", "ai_is_mock"]
    inlines = [ActionEventInline]
