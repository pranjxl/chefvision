import time

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Prefetch

from core.embeddings import EmbedderUnavailable, get_embedder, recipe_embedding_text
from core.models import Recipe, RecipeIngredient


class Command(BaseCommand):
    help = "Compute and store embeddings for recipes (only the ones missing one, unless --all)."

    def add_arguments(self, parser):
        parser.add_argument("--all", action="store_true", help="Re-embed every recipe, not just those without an embedding.")
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--backend", default=None, help="Override EMBEDDER_BACKEND ('sentence-transformers' or 'hash').")

    def handle(self, *args, all, batch_size, backend, **options):
        try:
            embedder = get_embedder(backend)
        except EmbedderUnavailable as exc:
            raise CommandError(str(exc)) from exc

        qs = Recipe.objects.all() if all else Recipe.objects.filter(embedding__isnull=True)
        qs = qs.prefetch_related(
            Prefetch("recipe_ingredients", queryset=RecipeIngredient.objects.select_related("ingredient").order_by("position"))
        ).order_by("pk")
        recipes = list(qs)
        if not recipes:
            self.stdout.write("Nothing to embed: every recipe already has an embedding.")
            return

        started = time.perf_counter()
        for start in range(0, len(recipes), batch_size):
            batch = recipes[start:start + batch_size]
            texts = [
                recipe_embedding_text(r.name, [ri.ingredient.name for ri in r.recipe_ingredients.all()], r.display_tags[:5])
                for r in batch
            ]
            for recipe, vector in zip(batch, embedder.embed(texts)):
                recipe.embedding = vector
            Recipe.objects.bulk_update(batch, ["embedding"])
            self.stdout.write(f"  embedded {min(start + batch_size, len(recipes))}/{len(recipes)}")
        elapsed = time.perf_counter() - started
        self.stdout.write(self.style.SUCCESS(
            f"Embedded {len(recipes)} recipes with '{embedder.name}' in {elapsed:.1f}s."
        ))
