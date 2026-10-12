from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("imports/", views.import_list, name="imports"),
    path("imports/<int:pk>/", views.import_detail, name="import-detail"),
    path("messages/", views.message_list, name="messages"),
    path("messages/<int:pk>/", views.message_detail, name="message-detail"),
    path("messages/<int:pk>/correct/", views.message_correct, name="message-correct"),
    path("jobs/", views.job_list, name="jobs"),
    path("jobs/<int:pk>/", views.job_detail, name="job-detail"),
    path("jobs/<int:pk>/update/", views.job_update, name="job-update"),
    path("grouping/", views.grouping_review, name="grouping"),
    path("grouping/<int:pk>/decide/", views.grouping_decide, name="grouping-decide"),
    path("evaluation/label/", views.label_next, name="label"),
    path("analysis/run/", views.run_analysis, name="run-analysis"),
    path("issues/", views.issue_list, name="issues"),
    path("issues/<int:pk>/", views.issue_detail, name="issue-detail"),
    path("issues/<int:pk>/status/", views.issue_status, name="issue-status"),
    path("issues/<int:issue_pk>/actions/new/", views.action_create, name="action-create"),
    path("issues/<int:issue_pk>/actions/suggest/", views.action_suggest, name="action-suggest"),
    path("alerts/", views.alert_list, name="alerts"),
    path("alerts/detect/", views.detect_alerts, name="detect-alerts"),
    path("alerts/<int:pk>/", views.alert_detail, name="alert-detail"),
    path("alerts/<int:pk>/review/", views.alert_review, name="alert-review"),
    path("actions/", views.action_list, name="actions"),
    path("actions/<int:pk>/", views.action_detail, name="action-detail"),
    path("actions/<int:pk>/update/", views.action_update, name="action-update"),
]
