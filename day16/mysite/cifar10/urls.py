from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path("cifar10/", views.cifar10, name="cifar10"),
]
