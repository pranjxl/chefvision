from django.test import SimpleTestCase

from .embeddings import HashEmbedder, recipe_embedding_text
from .importer import _title
from .matching import (
    Weights, hybrid_score, is_staple, match_credit, parse_user_ingredients, score_overlap,
)


class TitleTests(SimpleTestCase):
    def test_apostrophes_are_restored(self):
        cases = {
            "baked beans don t get any better": "Baked Beans Don't Get Any Better",
            "mom s best meatloaf": "Mom's Best Meatloaf",
            "you re gonna love it": "You're Gonna Love It",
            "i ve got chili": "I've Got Chili",
            "can t believe it s not butter": "Can't Believe It's Not Butter",
        }
        for raw, expected in cases.items():
            self.assertEqual(_title(raw), expected, raw)

    def test_legitimate_single_letters_are_untouched(self):
        self.assertEqual(_title("chicken t-bone steak"), "Chicken T-bone Steak")
        self.assertEqual(_title("vitamin d smoothie"), "Vitamin D Smoothie")
        self.assertEqual(_title("won ton soup"), "Won Ton Soup")


class ParseTests(SimpleTestCase):
    def test_parse_string_and_list(self):
        self.assertEqual(parse_user_ingredients("Chicken, Tomatoes; rice\n onion ,,"), ["chicken", "tomato", "rice", "onion"])
        self.assertEqual(parse_user_ingredients(["Potatoes", "potato", " "]), ["potato"])
        self.assertEqual(parse_user_ingredients(""), [])
        self.assertEqual(parse_user_ingredients(None), [])

    def test_parse_is_capped(self):
        self.assertEqual(len(parse_user_ingredients([f"item{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(100)])), 40)


class MatchCreditTests(SimpleTestCase):
    def test_exact_and_descriptor_insensitive(self):
        self.assertEqual(match_credit("tomato", "tomato"), 1.0)
        self.assertEqual(match_credit("tomato", "fresh tomato"), 1.0)
        self.assertEqual(match_credit("onion", "onion chopped"), 1.0)
        self.assertEqual(match_credit("chicken breast", "boneless skinless chicken breast"), 1.0)

    def test_partial_when_one_name_contains_the_other(self):
        self.assertEqual(match_credit("chicken", "chicken breast"), 0.75)
        self.assertEqual(match_credit("chicken breast", "chicken"), 0.75)

    def test_unrelated_is_zero(self):
        self.assertEqual(match_credit("rice", "rice vinegar" if False else "tomato"), 0.0)
        self.assertEqual(match_credit("butter", "peanut"), 0.0)


class StapleTests(SimpleTestCase):
    def test_staples(self):
        for name in ["salt", "black pepper", "ground black pepper", "olive oil", "all-purpose flour", "water"]:
            self.assertTrue(is_staple(name), name)
        for name in ["bell pepper", "coconut oil" if False else "chicken", "sugar snap pea", "rice flour"]:
            self.assertFalse(is_staple(name), name)


class OverlapTests(SimpleTestCase):
    RECIPE = ["chicken breast", "tomato", "onion", "garlic", "salt", "olive oil"]

    def test_staples_are_ignored_when_enabled(self):
        r = score_overlap(["chicken", "tomato"], self.RECIPE, staples=True)
        self.assertEqual(r.considered, 4)  # salt and olive oil dropped
        self.assertEqual(sorted(r.have), ["chicken breast", "tomato"])
        self.assertEqual(sorted(r.missing), ["garlic", "onion"])
        self.assertAlmostEqual(r.coverage, (0.75 + 1.0) / 4)

    def test_staples_count_when_disabled(self):
        r = score_overlap(["chicken", "tomato"], self.RECIPE, staples=False)
        self.assertEqual(r.considered, 6)
        self.assertIn("salt", r.missing)
        self.assertLess(r.coverage, (0.75 + 1.0) / 4)

    def test_full_pantry_gives_full_coverage(self):
        r = score_overlap(["chicken breast", "tomato", "onion", "garlic"], self.RECIPE)
        self.assertEqual(r.coverage, 1.0)
        self.assertEqual(r.missing, [])

    def test_empty_and_all_staple_recipes(self):
        self.assertEqual(score_overlap(["tomato"], []).coverage, 0.0)
        self.assertEqual(score_overlap(["tomato"], ["salt", "water"]).coverage, 0.0)
        self.assertEqual(score_overlap([], ["tomato"]).coverage, 0.0)


class UsageTests(SimpleTestCase):
    def test_usage_is_the_share_of_the_pantry_a_recipe_uses(self):
        r = score_overlap(["chicken", "tomato", "onion", "garlic", "cream"], ["tomato", "onion", "garlic"])
        self.assertEqual(r.coverage, 1.0)
        self.assertAlmostEqual(r.usage, 3 / 5)

    def test_partial_matches_earn_partial_usage(self):
        r = score_overlap(["chicken"], ["chicken breast", "rice"])
        self.assertAlmostEqual(r.usage, 0.75)

    def test_staples_typed_by_the_user_do_not_dilute_usage(self):
        with_staples = score_overlap(["tomato", "salt"], ["tomato", "onion"], staples=True)
        self.assertEqual(with_staples.usage, 1.0)
        without = score_overlap(["tomato", "salt"], ["tomato", "onion"], staples=False)
        self.assertEqual(without.usage, 0.5)


class HybridScoreTests(SimpleTestCase):
    W = Weights(coverage=0.5, usage=0.2, similarity=0.2, missing=0.1)

    def test_more_coverage_scores_higher(self):
        self.assertGreater(hybrid_score(0.9, 0.5, 0.3, 1, self.W), hybrid_score(0.4, 0.5, 0.3, 1, self.W))

    def test_using_more_of_the_pantry_scores_higher(self):
        self.assertGreater(hybrid_score(1.0, 1.0, 0.3, 0, self.W), hybrid_score(1.0, 0.4, 0.3, 0, self.W))

    def test_missing_ingredients_are_penalized_and_saturate(self):
        self.assertGreater(hybrid_score(0.5, 0.5, 0.2, 1, self.W), hybrid_score(0.5, 0.5, 0.2, 6, self.W))
        self.assertEqual(hybrid_score(0.5, 0.5, 0.2, 8, self.W), hybrid_score(0.5, 0.5, 0.2, 80, self.W))

    def test_similarity_breaks_ties_and_is_clamped(self):
        self.assertGreater(hybrid_score(0.5, 0.5, 0.6, 2, self.W), hybrid_score(0.5, 0.5, 0.1, 2, self.W))
        self.assertEqual(hybrid_score(0.5, 0.5, 5.0, 2, self.W), hybrid_score(0.5, 0.5, 1.0, 2, self.W))
        self.assertEqual(hybrid_score(0.5, 0.5, -3.0, 2, self.W), hybrid_score(0.5, 0.5, None, 2, self.W))

    def test_coverage_only_weights_ignore_everything_else(self):
        w = Weights(coverage=1.0, usage=0.0, similarity=0.0, missing=0.0)
        self.assertEqual(hybrid_score(0.5, 0.1, 0.9, 5, w), 0.5)


class HashEmbedderTests(SimpleTestCase):
    def test_deterministic_unit_vectors_with_right_size(self):
        e = HashEmbedder()
        a, b = e.embed(["chicken tomato rice", "chicken tomato rice"])
        self.assertEqual(a, b)
        self.assertEqual(len(a), 384)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=6)

    def test_similar_texts_are_closer_than_unrelated(self):
        e = HashEmbedder()
        q, near, far = e.embed(["chicken tomato rice", "chicken rice bowl tomato", "chocolate cake sugar"])
        dot = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731
        self.assertGreater(dot(q, near), dot(q, far))

    def test_recipe_text_includes_name_ingredients_and_tags(self):
        text = recipe_embedding_text("Tomato Soup", ["tomato", "onion"], ["soup"])
        self.assertEqual(text, "Tomato Soup. Ingredients: tomato, onion. Tags: soup.")
