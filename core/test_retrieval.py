import io
import json

from django.core.management import CommandError, call_command
from django.test import TestCase, override_settings
from django.urls import reverse

from .embeddings import EmbedderUnavailable, HashEmbedder
from .matching import Weights
from .models import Ingredient, Recipe, RecipeIngredient
from .retrieval import rank_recipes


def make_recipe(source_id, name, ingredients, diet="veg", minutes=30, **extra):
    recipe = Recipe.objects.create(source_id=source_id, name=name, minutes=minutes, diet=diet, **extra)
    for i, ing in enumerate(ingredients):
        obj, _ = Ingredient.objects.get_or_create(name=ing)
        RecipeIngredient.objects.create(recipe=recipe, ingredient=obj, position=i)
    return recipe


@override_settings(EMBEDDER_BACKEND="hash")
class RankingTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.tomato_soup = make_recipe(1, "Tomato Soup", ["tomato", "onion", "garlic", "salt", "water"])
        cls.chicken_rice = make_recipe(2, "Chicken Rice", ["chicken breast", "rice", "onion", "garlic"], diet="nonveg")
        cls.chicken_tomato = make_recipe(3, "Chicken Tomato Curry", ["chicken breast", "tomato", "onion", "garlic", "cream"], diet="nonveg", minutes=50)
        cls.pasta = make_recipe(4, "Plain Pasta", ["pasta", "butter", "salt"], is_snack=True)
        cls.dessert = make_recipe(5, "Chocolate Cake", ["chocolate", "sugar", "flour", "egg", "butter"], diet="egg", minutes=90)
        call_command("embed_recipes", stdout=io.StringIO())

    def names(self, *args, **kwargs):
        return [i.recipe.name for i in rank_recipes(*args, **kwargs).items]

    def test_best_match_ranks_first_and_unrelated_recipes_are_excluded(self):
        names = self.names("chicken, tomato, onion, garlic, cream")
        self.assertEqual(names[0], "Chicken Tomato Curry")  # uses all five, beating the smaller Tomato Soup
        self.assertNotIn("Chocolate Cake", names)
        self.assertNotIn("Plain Pasta", names)

    def test_have_and_missing_lists(self):
        result = rank_recipes("chicken, tomato, onion")
        top = next(i for i in result.items if i.recipe == self.chicken_tomato)
        self.assertCountEqual(top.have, ["chicken breast", "tomato", "onion"])
        self.assertCountEqual(top.missing, ["garlic", "cream"])
        self.assertAlmostEqual(top.coverage, (0.75 + 1 + 1) / 5)

    def test_staples_toggle_changes_what_counts(self):
        with_staples = rank_recipes("tomato, onion, garlic", staples=True).items[0]
        without = rank_recipes("tomato, onion, garlic", staples=False).items[0]
        self.assertEqual(with_staples.recipe, self.tomato_soup)
        self.assertEqual(with_staples.coverage, 1.0)
        self.assertCountEqual(without.missing, ["salt", "water"])
        self.assertLess(without.coverage, 1.0)

    def test_hard_filters_are_applied_before_ranking(self):
        self.assertNotIn("Chicken Tomato Curry", self.names("chicken, tomato, onion", diet="veg"))
        self.assertEqual(self.names("tomato, onion, chicken", diet="nonveg")[0], "Chicken Tomato Curry")
        self.assertEqual(self.names("butter, pasta, flour", snack=True), ["Plain Pasta"])
        self.assertNotIn("Chicken Tomato Curry", self.names("chicken, tomato", max_minutes=45))

    def test_no_match_and_empty_input(self):
        self.assertEqual(rank_recipes("dragonfruit").items, [])
        self.assertEqual(rank_recipes("").items, [])
        self.assertEqual(rank_recipes(None).items, [])

    def test_limit_and_candidate_count(self):
        result = rank_recipes("onion", limit=1)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.candidates, 3)

    def test_usage_stops_tiny_recipes_from_winning_on_coverage_alone(self):
        pantry = "chicken breast, tomato, onion, garlic, cream"
        naive = rank_recipes(pantry, weights=Weights(coverage=1.0, usage=0.0, similarity=0.0, missing=0.0)).items
        self.assertEqual({naive[0].coverage, naive[1].coverage}, {1.0})  # soup and curry tie on coverage
        self.assertEqual(rank_recipes(pantry).items[0].recipe, self.chicken_tomato)

    def test_semantic_used_and_similarity_reported(self):
        result = rank_recipes("chicken, rice")
        self.assertTrue(result.semantic_used)
        self.assertGreater(result.items[0].similarity, 0)

    def test_overlap_only_weights_skip_the_embedder(self):
        result = rank_recipes("chicken", weights=Weights(coverage=1.0, usage=0.0, similarity=0.0, missing=0.0))
        self.assertFalse(result.semantic_used)

    def test_semantic_only_scores_all_recipes(self):
        result = rank_recipes("chicken tomato", weights=Weights(coverage=0.0, usage=0.0, similarity=1.0, missing=0.0), candidates="all", limit=None)
        self.assertEqual(result.candidates, 5)

    def test_degrades_to_overlap_when_embedder_unavailable(self):
        from unittest import mock

        with mock.patch("core.retrieval.get_embedder", side_effect=EmbedderUnavailable("nope")):
            with self.assertLogs("core.retrieval", level="WARNING"):
                result = rank_recipes("chicken breast, tomato, onion, garlic, cream")
        self.assertFalse(result.semantic_used)
        self.assertEqual(result.items[0].recipe, self.chicken_tomato)

    def test_ranking_is_deterministic(self):
        self.assertEqual(self.names("onion, garlic"), self.names("onion, garlic"))


@override_settings(EMBEDDER_BACKEND="hash")
class EmbedCommandTests(TestCase):
    def test_embeds_only_missing_and_is_idempotent(self):
        a = make_recipe(1, "A", ["tomato", "onion"])
        out = io.StringIO()
        call_command("embed_recipes", stdout=out)
        a.refresh_from_db()
        self.assertEqual(len(a.embedding), 384)
        out = io.StringIO()
        call_command("embed_recipes", stdout=out)
        self.assertIn("Nothing to embed", out.getvalue())
        b = make_recipe(2, "B", ["rice"])
        call_command("embed_recipes", stdout=io.StringIO())
        b.refresh_from_db()
        self.assertEqual(len(b.embedding), 384)

    def test_all_flag_reembeds(self):
        make_recipe(1, "A", ["tomato"])
        call_command("embed_recipes", stdout=io.StringIO())
        out = io.StringIO()
        call_command("embed_recipes", "--all", stdout=out)
        self.assertIn("Embedded 1 recipes", out.getvalue())

    def test_unknown_backend_is_a_clean_error(self):
        with self.assertRaises(CommandError):
            call_command("embed_recipes", backend="nonsense", stdout=io.StringIO())


@override_settings(EMBEDDER_BACKEND="hash")
class CookPageAndApiTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.curry = make_recipe(1, "Chicken Tomato Curry", ["chicken breast", "tomato", "onion", "cream"], diet="nonveg")
        cls.soup = make_recipe(2, "Tomato Soup", ["tomato", "onion", "garlic"])
        call_command("embed_recipes", stdout=io.StringIO())

    def test_empty_page_shows_prompt(self):
        r = self.client.get(reverse("cook"))
        self.assertContains(r, "What is in your kitchen?")

    def test_results_show_have_and_missing(self):
        r = self.client.get(reverse("cook"), {"ingredients": "chicken, tomato", "staples": "yes"})
        self.assertContains(r, "Chicken Tomato Curry")
        self.assertContains(r, "Missing")
        self.assertContains(r, "cream")
        self.assertContains(r, "% match")

    def test_filters_apply(self):
        r = self.client.get(reverse("cook"), {"ingredients": "tomato", "diet": "veg"})
        self.assertContains(r, "Tomato Soup")
        self.assertNotContains(r, "Chicken Tomato Curry")

    def test_no_results_state_and_blank_submit(self):
        self.assertContains(self.client.get(reverse("cook"), {"ingredients": "dragonfruit"}), "No recipes use those")
        self.assertContains(self.client.get(reverse("cook"), {"ingredients": "  "}), "Add at least one ingredient")

    def test_garbage_params_do_not_crash(self):
        r = self.client.get(reverse("cook"), {"ingredients": "x" * 5000, "diet": "bogus", "max_minutes": "zzz"})
        self.assertEqual(r.status_code, 200)

    def test_htmx_partial(self):
        r = self.client.get(reverse("cook"), {"ingredients": "tomato"}, HTTP_HX_REQUEST="true")
        self.assertTemplateUsed(r, "cook/_results.html")
        self.assertNotContains(r, "<html")

    def test_api_returns_ranked_json(self):
        r = self.client.post(
            reverse("api_match"), data=json.dumps({"ingredients": ["Chicken", "tomatoes"], "limit": 5}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["ingredients"], ["chicken", "tomato"])
        self.assertTrue(body["semantic_used"])
        self.assertEqual(body["results"][0]["name"], "Chicken Tomato Curry")
        self.assertEqual(body["results"][0]["missing"], ["onion", "cream"])
        self.assertEqual(body["results"][0]["url"], reverse("recipe_detail", args=[self.curry.pk]))

    def test_api_validates_input(self):
        post = lambda payload: self.client.post(reverse("api_match"), data=json.dumps(payload), content_type="application/json")  # noqa: E731
        self.assertEqual(post({"ingredients": []}).status_code, 400)
        self.assertEqual(post({}).status_code, 400)
        self.assertEqual(post({"ingredients": ["a"], "diet": "bogus"}).status_code, 400)
        self.assertEqual(post({"ingredients": ["a"], "limit": 500}).status_code, 400)
        self.assertEqual(self.client.get(reverse("api_match")).status_code, 405)


@override_settings(EMBEDDER_BACKEND="hash")
class EvaluationCommandTests(TestCase):
    def test_benchmark_runs_and_prints_all_methods(self):
        pools = [
            ["tomato", "onion", "garlic", "basil", "pasta"], ["chicken breast", "rice", "onion", "garlic", "ginger"],
            ["potato", "pea", "cumin", "ginger", "tomato"], ["lentil", "onion", "garlic", "cumin", "spinach"],
            ["chickpea", "tomato", "onion", "cumin", "garlic"], ["paneer", "spinach", "onion", "cream", "garlic"],
        ]
        for i, ings in enumerate(pools):
            make_recipe(i + 1, f"Recipe {i}", ings)
        call_command("embed_recipes", stdout=io.StringIO())
        out = io.StringIO()
        call_command("evaluate_matching", n=6, stdout=out)
        text = out.getvalue()
        for label in ("Coverage only", "Overlap (coverage", "Semantic only", "Hybrid", "Recall@5", "MRR"):
            self.assertIn(label, text)
        self.assertIn("stand-in embedder", text)

    def test_requires_embeddings(self):
        make_recipe(1, "A", ["tomato"])
        with self.assertRaises(CommandError):
            call_command("evaluate_matching", stdout=io.StringIO())
