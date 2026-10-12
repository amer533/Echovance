from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("imports", views.ImportBatchViewSet, basename="import")
router.register("messages", views.MessageViewSet, basename="message")
router.register("analyses", views.AnalysisRunViewSet, basename="analysis")
router.register("issues", views.IssueViewSet, basename="issue")
router.register("alerts", views.AlertViewSet, basename="alert")
router.register("actions", views.ActionViewSet, basename="action")
router.register("jobs", views.AnalysisJobViewSet, basename="job")
router.register("grouping-suggestions", views.GroupingSuggestionViewSet, basename="grouping-suggestion")

urlpatterns = [
    path("dashboard/summary/", views.DashboardSummaryView.as_view(), name="api-dashboard-summary"),
    path("", include(router.urls)),
]
