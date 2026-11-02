import csv
import io
import tempfile
from pathlib import Path

from django.core.management import call_command
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .importer import ParsedRecipe, parse_row, select_recipes
from .ingredients import classify_diet, normalize_ingredient
from .models import Ingredient, Recipe

FIELDS = ["name", "id", "minutes", "contributor_id", "submitted", "tags", "nutrition",
          "n_steps", "steps", "description", "ingredients", "n_ingredients"]


def make_row(i, name=None, minutes=30, ingredients=("onion", "tomato", "rice"), tags=("easy",),
             protein_pdv=10.0, calories=250.0, n_steps=4, **over):
    row = {
        "name": name or f"test   recipe {i}", "id": str(1000 + i), "minutes": str(minutes),
        "contributor_id": "1", "submitted": "2020-01-01", "tags": repr(list(tags)),
        "nutrition": repr([calories, 10.0, 5.0, 8.0, protein_pdv, 6.0, 12.0]),
        "n_steps": str(n_steps), "steps": repr([f"do step {n} carefully" for n in range(n_steps)]),
        "description": "a tasty dish", "ingredients": repr(list(ingredients)), "n_ingredients": str(len(ingredients)),
    }
    row.update(over)
    return row


class IngredientTests(SimpleTestCase):
    def test_normalization(self):
        cases = {
            "Fresh Tomatoes (diced)": "fresh tomato", "potatoes": "potato", "berries": "berry",
            "hummus": "hummus", "molasses": "molasses", "green onions": "green onion", "  Olive   Oil ": "olive oil",
            "peaches": "peach", "leaves": "leaf",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_ingredient(raw), expected, raw)

    def test_diet_classification(self):
        self.assertEqual(classify_diet(["eggplant", "onion"]), "veg")  # eggplant is not an egg
        self.assertEqual(classify_diet(["egg", "flour"]), "egg")
        self.assertEqual(classify_diet(["chicken breast", "egg"]), "nonveg")  # meat wins over egg
        self.assertEqual(classify_diet(["hamburger bun", "tomato"]), "veg")
        self.assertEqual(classify_diet(["fish sauce", "rice"]), "nonveg")
        self.assertEqual(classify_diet(["paneer", "spinach"]), "veg")


class ImporterTests(SimpleTestCase):
    def test_valid_row_is_cleaned(self):
        r = parse_row(make_row(1, ingredients=("Eggs", "flour", "milk"), protein_pdv=45.0))
        self.assertEqual(r.name, "Test Recipe 1")
        self.assertEqual(r.diet, "egg")
        self.assertAlmostEqual(r.protein_g, 22.5)
        self.assertTrue(r.is_high_protein)
        self.assertTrue(r.steps[0].startswith("Do step"))

    def test_quality_filters(self):
        self.assertIsNone(parse_row(make_row(1, minutes=2)))            # too quick to be real
        self.assertIsNone(parse_row(make_row(2, minutes=100000)))       # absurd duration
        self.assertIsNone(parse_row(make_row(3, n_steps=1)))
        self.assertIsNone(parse_row(make_row(4, ingredients=("salt", "water"))))
        self.assertIsNone(parse_row(make_row(5, nutrition="[1, 2]")))   # malformed nutrition
        self.assertIsNone(parse_row(make_row(6, steps="not a list")))

    def test_selection_meets_quotas(self):
        def cand(i, veg, protein, snack):
            return ParsedRecipe(
                source_id=i, name=f"r{i}", description="", minutes=20, ingredients_raw=[], ingredients=[], steps=[],
                tags=[], nutrition=[], diet="veg" if veg else "nonveg", calories=200, protein_g=30 if protein else 5,
                is_high_protein=protein, is_snack=snack,
            )
        pool = [cand(i, veg=i % 5 == 0, protein=i % 8 == 0, snack=i % 7 == 0) for i in range(400)]
        chosen = select_recipes(pool, limit=50, seed=1)
        self.assertEqual(len(chosen), 50)
        self.assertEqual(len({c.source_id for c in chosen}), 50)
        self.assertGreaterEqual(sum(c.diet == "veg" for c in chosen), 20)
        self.assertGreaterEqual(sum(c.is_high_protein for c in chosen), 10)
        self.assertGreaterEqual(sum(c.is_snack for c in chosen), 10)
        self.assertEqual([c.source_id for c in chosen], [c.source_id for c in select_recipes(pool, 50, seed=1)])


class DiscoveryTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        rows = [
            make_row(1, name="paneer butter masala", ingredients=("paneer", "tomato", "butter"), minutes=25, protein_pdv=45, calories=400),
            make_row(2, name="chicken curry", ingredients=("chicken breast", "onion", "tomato"), minutes=50, protein_pdv=60, calories=350),
            make_row(3, name="egg bhurji", ingredients=("eggs", "onion", "chilli"), minutes=10, protein_pdv=42, calories=200),
            make_row(4, name="masala peanuts", ingredients=("peanuts", "chilli powder", "lemon juice"), minutes=12, tags=("snacks",), calories=300),
            make_row(5, name="plain khichdi", ingredients=("rice", "lentils", "ghee"), minutes=40, protein_pdv=15, calories=450),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipes.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            call_command("import_recipes", path=str(path), limit=10, stdout=io.StringIO())
        cls.url = reverse("recipe_list")

    def names(self, **params):
        response = self.client.get(self.url, params)
        self.assertEqual(response.status_code, 200)
        return [r.name for r in response.context["page"].object_list]

    def test_import_builds_normalized_ingredients(self):
        self.assertEqual(Recipe.objects.count(), 5)
        self.assertTrue(Ingredient.objects.filter(name="lentil").exists())  # 'lentils' singularized
        paneer = Recipe.objects.get(name="Paneer Butter Masala")
        self.assertEqual([i.name for i in paneer.ingredients.order_by("recipe_ingredients__position")], ["paneer", "tomato", "butter"])

    def test_import_is_idempotent(self):
        before = Recipe.objects.count()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "again.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(make_row(1, name="paneer butter masala", ingredients=("paneer", "tomato", "butter")))
            call_command("import_recipes", path=str(path), limit=10, stdout=io.StringIO())
        self.assertEqual(Recipe.objects.count(), before)

    def test_search_matches_name_and_ingredients(self):
        self.assertEqual(self.names(q="paneer"), ["Paneer Butter Masala"])
        self.assertCountEqual(self.names(q="tomato"), ["Paneer Butter Masala", "Chicken Curry"])
        self.assertEqual(self.names(q="chicken tomato"), ["Chicken Curry"])  # all terms must match

    def test_diet_filter_treats_egg_as_nonveg(self):
        self.assertCountEqual(self.names(diet="veg"), ["Paneer Butter Masala", "Masala Peanuts", "Plain Khichdi"])
        self.assertCountEqual(self.names(diet="nonveg"), ["Chicken Curry", "Egg Bhurji"])

    def test_protein_snack_and_time_filters(self):
        self.assertCountEqual(self.names(high_protein="on"), ["Paneer Butter Masala", "Chicken Curry", "Egg Bhurji"])
        self.assertEqual(self.names(snack="on"), ["Masala Peanuts"])
        self.assertCountEqual(self.names(max_minutes="15"), ["Egg Bhurji", "Masala Peanuts"])

    def test_sorting(self):
        self.assertEqual(self.names(sort="quickest")[0], "Egg Bhurji")
        self.assertEqual(self.names(sort="protein")[0], "Chicken Curry")
        self.assertEqual(self.names(sort="calories")[0], "Egg Bhurji")

    def test_best_match_ranks_name_hits_first(self):
        # 'masala' is in two names; 'tomato' only in ingredients elsewhere -> name hits first
        self.assertEqual(self.names(q="masala")[:2], ["Masala Peanuts", "Paneer Butter Masala"])

    def test_invalid_params_do_not_crash(self):
        self.assertEqual(len(self.names(diet="bogus", sort="nope", max_minutes="x")), 5)

    def test_htmx_returns_partial_and_full_page_otherwise(self):
        partial = self.client.get(self.url, HTTP_HX_REQUEST="true")
        self.assertTemplateUsed(partial, "recipes/_results.html")
        self.assertNotContains(partial, "<html")
        self.assertEqual(partial["Vary"], "HX-Request")
        self.assertTemplateUsed(self.client.get(self.url), "recipes/list.html")

    def test_pagination(self):
        for i in range(20, 40):
            Recipe.objects.create(source_id=5000 + i, name=f"Filler {i:02d}", minutes=20, diet="veg")
        response = self.client.get(self.url, {"page": 2})
        self.assertEqual(response.context["page"].number, 2)
        self.assertEqual(response.context["total"], 25)

    def test_detail_page(self):
        recipe = Recipe.objects.get(name="Chicken Curry")
        response = self.client.get(reverse("recipe_detail", args=[recipe.pk]))
        self.assertContains(response, "Chicken Curry")
        self.assertContains(response, "Non-vegetarian")
        self.assertEqual(self.client.get(reverse("recipe_detail", args=[999999])).status_code, 404)
