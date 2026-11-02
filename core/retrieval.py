"""Hybrid recipe retrieval: ingredient overlap + semantic similarity (pgvector) + hard filters."""
import logging
from dataclasses import dataclass

from django.conf import settings
from django.db.models import Prefetch
from pgvector.django import CosineDistance

from .embeddings import EmbedderUnavailable, get_embedder, query_text
from .filters import apply_recipe_filters
from .matching import Weights, hybrid_score, match_credit, parse_user_ingredients, score_overlap
from .models import Ingredient, Recipe, RecipeIngredient

logger = logging.getLogger(__name__)


@dataclass
class RankedRecipe:
    recipe: Recipe
    score: float
    coverage: float
    usage: float
    similarity: float
    have: list[str]
    missing: list[str]

    @property
    def coverage_pct(self) -> int:
        return round(self.coverage * 100)

    @property
    def considered(self) -> int:
        return len(self.have) + len(self.missing)


@dataclass
class RankResult:
    items: list[RankedRecipe]
    semantic_used: bool
    candidates: int
    ingredients: list[str]


def default_weights() -> Weights:
    return Weights(**settings.RANKING_WEIGHTS)


def _matching_ingredient_ids(user: list[str]) -> list[int]:
    return [
        pk for pk, name in Ingredient.objects.values_list("id", "name")
        if any(match_credit(u, name) > 0 for u in user)
    ]


def rank_recipes(
    ingredients,
    *,
    diet="",
    high_protein=False,
    snack=False,
    max_minutes=None,
    staples=True,
    limit: int | None = 8,
    weights: Weights | None = None,
    candidates: str = "matching",
    embedder=None,
    source: str = Recipe.Source.STORED,
) -> RankResult:
    """Rank recipes for a pantry.

    candidates="matching": only recipes sharing at least one ingredient (the normal mode).
    candidates="all": every recipe passing the filters (used for the semantic-only baseline).
    If the embedding backend is unavailable the ranking degrades to overlap-only instead of failing.
    """
    user = parse_user_ingredients(ingredients)
    weights = weights or default_weights()
    if not user:
        return RankResult([], False, 0, [])

    qs = apply_recipe_filters(
        Recipe.objects.filter(source=source),
        diet=diet, high_protein=high_protein, snack=snack, max_minutes=max_minutes,
    )
    if candidates == "matching":
        ids = _matching_ingredient_ids(user)
        if not ids:
            return RankResult([], False, 0, user)
        qs = qs.filter(pk__in=RecipeIngredient.objects.filter(ingredient_id__in=ids).values("recipe_id"))

    semantic_used = False
    if weights.similarity > 0 or candidates == "all":
        try:
            embedder = embedder or get_embedder()
            vector = embedder.embed([query_text(user)])[0]
            qs = qs.annotate(distance=CosineDistance("embedding", vector))
            semantic_used = True
        except EmbedderUnavailable:
            if candidates == "all":
                raise
            logger.warning("Embedding backend unavailable; ranking by ingredient overlap only.", exc_info=True)

    qs = qs.prefetch_related(
        Prefetch("recipe_ingredients", queryset=RecipeIngredient.objects.select_related("ingredient").order_by("position"))
    )

    ranked: list[RankedRecipe] = []
    for recipe in qs:
        names = [ri.ingredient.name for ri in recipe.recipe_ingredients.all()]
        overlap = score_overlap(user, names, staples=staples)
        distance = getattr(recipe, "distance", None)
        similarity = 1.0 - float(distance) if distance is not None else 0.0
        ranked.append(RankedRecipe(
            recipe=recipe,
            score=hybrid_score(overlap.coverage, overlap.usage, similarity, len(overlap.missing), weights),
            coverage=overlap.coverage, usage=overlap.usage, similarity=similarity, have=overlap.have, missing=overlap.missing,
        ))
    ranked.sort(key=lambda r: (-r.score, r.recipe.name))
    return RankResult(
        items=ranked if limit is None else ranked[:limit],
        semantic_used=semantic_used, candidates=len(ranked), ingredients=user,
    )
