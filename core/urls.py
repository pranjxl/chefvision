from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("recipes/", views.recipe_list, name="recipe_list"),
    path("recipes/<int:pk>/", views.recipe_detail, name="recipe_detail"),
    path("cook/", views.cook, name="cook"),
    path("api/match/", views.api_match, name="api_match"),
    path("health/", views.health, name="health"),
]
