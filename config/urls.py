from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/login/", auth_views.LoginView.as_view(template_name="feedback/login.html"), name="login"),
    path("accounts/logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("api/", include("feedback.api.urls")),
    path("api/auth/", include("rest_framework.urls")),
    path("", include("feedback.web.urls")),
]
