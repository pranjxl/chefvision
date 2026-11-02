import redis
from django.conf import settings
from django.core.paginator import Paginator
from django.db import connection
from django.db.models import Case, IntegerField, Q, Value, When
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.vary import vary_on_headers
from rest_framework import serializers
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .filters import apply_recipe_filters
from .forms import CookForm, RecipeFilterForm
from .models import Recipe
from .retrieval import rank_recipes

PAGE_SIZE = 12


def home(request):
    stored = Recipe.objects.filter(source=Recipe.Source.STORED)
    return render(request, "home.html", {"recipe_count": stored.count()})


def filter_recipes(params):
    """Apply the discovery-page filters to the stored-recipe collection."""
    form = RecipeFilterForm(params)
    qs = Recipe.objects.filter(source=Recipe.Source.STORED)
    if not form.is_valid():
        return form, qs.order_by("name")

    data = form.cleaned_data
    q = data["q"].strip()
    if q:
        for term in q.split()[:6]:
            qs = qs.filter(Q(name__icontains=term) | Q(ingredients__name__icontains=term))
        qs = qs.distinct()
    qs = apply_recipe_filters(
        qs, diet=data["diet"], high_protein=data["high_protein"], snack=data["snack"], max_minutes=data["max_minutes"],
    )

    sort = data["sort"] or "best"
    if sort == "quickest":
        qs = qs.order_by("minutes", "name")
    elif sort == "protein":
        qs = qs.order_by("-protein_g", "name")
    elif sort == "calories":
        qs = qs.order_by("calories", "name")
    elif sort == "name" or not q:
        qs = qs.order_by("name")
    else:  # "best" with a search term: recipes whose *name* matches come first
        qs = qs.annotate(
            name_hit=Case(When(name__icontains=q.split()[0], then=Value(1)), default=Value(0), output_field=IntegerField())
        ).order_by("-name_hit", "name")
    return form, qs


@vary_on_headers("HX-Request")
def recipe_list(request):
    form, qs = filter_recipes(request.GET)
    page = Paginator(qs, PAGE_SIZE).get_page(request.GET.get("page"))
    params = request.GET.copy()
    params.pop("page", None)
    context = {"form": form, "page": page, "querystring": params.urlencode(), "total": page.paginator.count}
    htmx = request.htmx and not request.htmx.history_restore_request
    return render(request, "recipes/_results.html" if htmx else "recipes/list.html", context)


def recipe_detail(request, pk):
    recipe = get_object_or_404(Recipe, pk=pk)
    return render(request, "recipes/detail.html", {"recipe": recipe})


@vary_on_headers("HX-Request")
def cook(request):
    """'Cook with my ingredients' (manual entry; photo / camera input arrives in Phase 3)."""
    searched = bool(request.GET)
    form = CookForm(request.GET or None)
    result = None
    if searched and form.is_valid() and form.cleaned_data["ingredients"].strip():
        data = form.cleaned_data
        result = rank_recipes(
            data["ingredients"], diet=data["diet"], high_protein=data["high_protein"], snack=data["snack"],
            max_minutes=data["max_minutes"], staples=form.assume_staples, limit=8,
        )
    context = {"form": form, "result": result, "searched": searched}
    htmx = request.htmx and not request.htmx.history_restore_request
    return render(request, "cook/_results.html" if htmx else "cook/index.html", context)


class MatchRequestSerializer(serializers.Serializer):
    ingredients = serializers.ListField(child=serializers.CharField(max_length=80), min_length=1, max_length=40)
    diet = serializers.ChoiceField(choices=["veg", "nonveg"], required=False, allow_blank=True, default="")
    high_protein = serializers.BooleanField(default=False)
    snack = serializers.BooleanField(default=False)
    max_minutes = serializers.IntegerField(required=False, allow_null=True, min_value=1, max_value=1000)
    staples = serializers.BooleanField(default=True)
    limit = serializers.IntegerField(default=8, min_value=1, max_value=20)


@api_view(["POST"])
def api_match(request):
    """POST {"ingredients": ["chicken", "tomato"], "diet": "veg", ...} -> ranked recipes."""
    body = MatchRequestSerializer(data=request.data)
    body.is_valid(raise_exception=True)
    d = body.validated_data
    result = rank_recipes(
        d["ingredients"], diet=d["diet"], high_protein=d["high_protein"], snack=d["snack"],
        max_minutes=d.get("max_minutes"), staples=d["staples"], limit=d["limit"],
    )
    return Response({
        "ingredients": result.ingredients,
        "semantic_used": result.semantic_used,
        "candidates": result.candidates,
        "results": [
            {
                "id": item.recipe.pk, "name": item.recipe.name, "diet": item.recipe.diet,
                "minutes": item.recipe.minutes, "url": reverse("recipe_detail", args=[item.recipe.pk]),
                "score": round(item.score, 4), "coverage": round(item.coverage, 4), "usage": round(item.usage, 4),
                "similarity": round(item.similarity, 4), "have": item.have, "missing": item.missing,
            }
            for item in result.items
        ],
    })


def _check_database() -> dict:
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
        # pgvector is required from Phase 2 onward, so a missing extension is a failure.
        return {"ok": row is not None, "pgvector": row[0] if row else None}
    except Exception as exc:  # noqa: BLE001 - health endpoint reports any failure
        return {"ok": False, "error": str(exc)}


def _check_redis() -> dict:
    try:
        client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        return {"ok": bool(client.ping())}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def health(request):
    checks = {"database": _check_database(), "redis": _check_redis()}
    healthy = all(c["ok"] for c in checks.values())
    return JsonResponse(
        {"status": "ok" if healthy else "degraded", "checks": checks},
        status=200 if healthy else 503,
    )
