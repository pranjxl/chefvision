from django.db import models
from pgvector.django import HnswIndex, VectorField

# Tags from the Food.com dataset that describe timing/bookkeeping rather than the dish.
_NOISY_TAGS = {
    "time-to-make", "course", "preparation", "main-ingredient", "cuisine", "occasion",
    "dietary", "equipment", "taste-mood", "number-of-servings", "main-dish", "easy",
    "healthy", "low-in-something", "North-American", "north-american",
}


class Ingredient(models.Model):
    """A normalized (lowercase, singularized) ingredient name used for matching."""

    name = models.CharField(max_length=120, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Recipe(models.Model):
    class Source(models.TextChoices):
        STORED = "stored", "Stored"
        GENERATED = "generated", "AI-generated"

    class Diet(models.TextChoices):
        VEG = "veg", "Vegetarian"
        EGG = "egg", "Contains egg"
        NONVEG = "nonveg", "Non-vegetarian"

    # stored = imported from the dataset, generated = created by the LLM in Phase 4
    source = models.CharField(max_length=10, choices=Source.choices, default=Source.STORED, db_index=True)
    # Original dataset id, so re-imports are idempotent. Null for generated recipes.
    source_id = models.IntegerField(null=True, blank=True, unique=True)

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    minutes = models.PositiveIntegerField(default=0)

    # Display copy of the ingredient lines, as written in the source.
    ingredients_raw = models.JSONField(default=list)
    steps = models.JSONField(default=list)
    tags = models.JSONField(default=list)
    # Raw Food.com vector: [calories, fat %DV, sugar %DV, sodium %DV, protein %DV, sat fat %DV, carbs %DV]
    nutrition = models.JSONField(default=list)

    # Derived at import time so filtering and sorting stay plain indexed SQL.
    diet = models.CharField(max_length=6, choices=Diet.choices, default=Diet.VEG, db_index=True)
    calories = models.FloatField(null=True, blank=True)
    protein_g = models.FloatField(null=True, blank=True, db_index=True)
    is_high_protein = models.BooleanField(default=False, db_index=True)
    is_snack = models.BooleanField(default=False, db_index=True)

    # Normalized ingredients for matching (Phases 2-4).
    ingredients = models.ManyToManyField(Ingredient, through="RecipeIngredient", related_name="recipes")

    # Filled in Phase 2 (all-MiniLM-L6-v2 produces 384 dimensions).
    embedding = VectorField(dimensions=384, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        indexes = [
            # Approximate nearest-neighbour index for cosine similarity search (Phase 2).
            HnswIndex(
                name="recipe_embedding_hnsw", fields=["embedding"], m=16, ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self):
        return self.name

    @property
    def display_tags(self):
        return [
            t for t in self.tags
            if t not in _NOISY_TAGS and not t.endswith("-or-less") and not t[:1].isdigit()
        ][:6]

    @property
    def is_veg(self):
        return self.diet == self.Diet.VEG

    @property
    def nutrition_facts(self):
        """Human-readable nutrition panel, or an empty dict if the data is missing."""
        if len(self.nutrition) != 7:
            return {}
        calories, fat, sugar, sodium, protein, sat_fat, carbs = self.nutrition
        return {
            "Calories": f"{calories:.0f} kcal",
            "Protein": f"{protein * 0.5:.0f} g ({protein:.0f}% DV)",
            "Total fat": f"{fat:.0f}% DV",
            "Saturated fat": f"{sat_fat:.0f}% DV",
            "Carbohydrates": f"{carbs:.0f}% DV",
            "Sugar": f"{sugar:.0f}% DV",
            "Sodium": f"{sodium:.0f}% DV",
        }


class RecipeIngredient(models.Model):
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE, related_name="recipe_ingredients")
    ingredient = models.ForeignKey(Ingredient, on_delete=models.CASCADE, related_name="recipe_ingredients")
    position = models.PositiveSmallIntegerField(default=0)
    # Empty for stored recipes (the dataset has no quantities); LLM-generated recipes fill this.
    quantity = models.CharField(max_length=60, blank=True)

    class Meta:
        ordering = ["recipe_id", "position"]
        constraints = [
            models.UniqueConstraint(fields=["recipe", "ingredient"], name="unique_ingredient_per_recipe"),
        ]

    def __str__(self):
        return f"{self.recipe_id}: {self.ingredient_id}"
