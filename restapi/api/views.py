from django.shortcuts import render
from rest_framework import generics

from .serializers import *


class AnalysisCreateView(generics.CreateAPIView):
    serializer_class = AnalysisDetailSerializer


class AnalysisListView(generics.ListAPIView):
    serializer_class = AnalysisListSerializer
    queryset = Analysis.objects.all()


class AnalysisDetailView(generics.RetrieveDestroyAPIView):
    serializer_class = AnalysisDetailSerializer
    queryset = Analysis.objects.all()
