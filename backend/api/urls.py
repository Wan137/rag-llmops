from django.urls import path

from .views import AskQuestionView, DocumentUploadView, HealthView

urlpatterns = [
    path("health/", HealthView.as_view()),
    path("ask/", AskQuestionView.as_view()),
    path("documents/", DocumentUploadView.as_view()),
]
