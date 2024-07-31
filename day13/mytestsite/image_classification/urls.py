from django.urls import path
from . import views

urlpatterns = [
    path("", views.index),
    path("myfirst/", views.myfirst),
    path("classification/", views.classification),
]