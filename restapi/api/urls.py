from django.urls import path

from .views import *

urlpatterns = [
    path('add/', AnalysisCreateView.as_view()),
    path('list/', AnalysisListView.as_view()),
    path('<int:pk>/', AnalysisDetailView.as_view()),
]
