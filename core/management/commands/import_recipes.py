from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Count

from core.importer import read_candidates, select_recipes
from core.models import Ingredient, Recipe, RecipeIngredient


class Command(BaseCommand):
    help = "Import a quality-filtered, balanced sample of recipes from Food.com RAW_recipes.csv."

    def add_arguments(self, parser):
        parser.add_argument("--path", default=str(Path(settings.BASE_DIR) / "data" / "RAW_recipes.csv"))
        parser.add_argument("--limit", type=int, default=500, help="How many recipes to keep (default 500).")
        parser.add_argument("--seed", type=int, default=42, help="Sampling seed, for reproducible imports.")
        parser.add_argument("--reset", action="store_true", help="Delete existing stored recipes first.")

    def handle(self, *args, path, limit, seed, reset, **options):
        if not Path(path).exists():
            raise CommandError(f"CSV not found: {path}")

        self.stdout.write("Reading and filtering rows...")
        candidates = read_candidates(path)
        self.stdout.write(f"  {len(candidates)} rows passed the quality checks")
        chosen = select_recipes(candidates, limit, seed)

        with transaction.atomic():
            if reset:
                deleted, _ = Recipe.objects.filter(source=Recipe.Source.STORED).delete()
                self.stdout.write(f"  removed {deleted} existing rows")

            existing = set(Recipe.objects.filter(source_id__isnull=False).values_list("source_id", flat=True))
            fresh = [r for r in chosen if r.source_id not in existing]

            Recipe.objects.bulk_create(
                [
                    Recipe(
                        source=Recipe.Source.STORED, source_id=r.source_id, name=r.name,
                        description=r.description, minutes=r.minutes, ingredients_raw=r.ingredients_raw,
                        steps=r.steps, tags=r.tags, nutrition=r.nutrition, diet=r.diet,
                        calories=r.calories, protein_g=r.protein_g,
                        is_high_protein=r.is_high_protein, is_snack=r.is_snack,
                    )
                    for r in fresh
                ],
                batch_size=500,
            )
            recipe_ids = dict(
                Recipe.objects.filter(source_id__in=[r.source_id for r in fresh]).values_list("source_id", "id")
            )

            names = {n for r in fresh for n in r.ingredients}
            Ingredient.objects.bulk_create([Ingredient(name=n) for n in names], ignore_conflicts=True, batch_size=1000)
            ingredient_ids = dict(Ingredient.objects.filter(name__in=names).values_list("name", "id"))

            RecipeIngredient.objects.bulk_create(
                [
                    RecipeIngredient(recipe_id=recipe_ids[r.source_id], ingredient_id=ingredient_ids[n], position=i)
                    for r in fresh
                    for i, n in enumerate(r.ingredients)
                ],
                batch_size=2000,
            )

        stored = Recipe.objects.filter(source=Recipe.Source.STORED)
        self.stdout.write(self.style.SUCCESS(f"Imported {len(fresh)} new recipes ({stored.count()} stored in total)."))
        for row in stored.values("diet").annotate(n=Count("id")).order_by("diet"):
            self.stdout.write(f"  {row['diet']:<7} {row['n']}")
        self.stdout.write(f"  high-protein {stored.filter(is_high_protein=True).count()}")
        self.stdout.write(f"  snacks       {stored.filter(is_snack=True).count()}")
        self.stdout.write(f"  ingredients  {Ingredient.objects.count()} distinct")
