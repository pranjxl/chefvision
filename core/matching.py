"""Pure ingredient-matching and scoring logic (no Django, no database).

A recipe is described by normalized ingredient names (see core.ingredients). A user's pantry is
a list of normalized names too. For every recipe ingredient we ask: does the user have it?
"""
import re
from dataclasses import dataclass

from .ingredients import normalize_ingredient

MAX_USER_INGREDIENTS = 40
PARTIAL_CREDIT = 0.75  # "chicken" vs "chicken breast": probably fine, but not certain
MAX_MISSING = 8        # missing-ingredient penalty saturates here

# Preparation / size words that don't change which ingredient you need to own.
DESCRIPTORS = frozenset({
    "fresh", "frozen", "dried", "dry", "ground", "chopped", "minced", "diced", "sliced", "shredded",
    "grated", "crushed", "large", "small", "medium", "boneless", "skinless", "lean", "whole", "raw",
    "cooked", "unsalted", "salted", "extra", "virgin", "plain", "pure", "finely", "thinly", "coarsely",
    "optional", "peeled", "softened", "melted", "cold", "warm", "hot", "ripe", "packed", "divided",
    "toasted", "roasted", "canned",
})

# Assumed to be in every kitchen when the user turns "pantry staples" on.
STAPLES = frozenset({
    "salt", "kosher salt", "sea salt", "table salt", "pepper", "black pepper", "salt and pepper",
    "water", "ice water", "oil", "cooking oil", "vegetable oil", "olive oil", "canola oil",
    "sugar", "white sugar", "granulated sugar", "flour", "all-purpose flour", "plain flour",
})

_SPLIT = re.compile(r"[,;\n]+")


def canonical(name: str) -> frozenset[str]:
    """Token set for comparing two ingredient names, ignoring descriptor words."""
    tokens = frozenset(name.split())
    core = tokens - DESCRIPTORS
    return core or tokens


_STAPLE_CANON = frozenset(canonical(s) for s in STAPLES)


def is_staple(name: str) -> bool:
    return canonical(name) in _STAPLE_CANON


def parse_user_ingredients(raw) -> list[str]:
    """Accept 'Chicken, Tomatoes; rice' or a list of strings; return normalized, unique names."""
    items = _SPLIT.split(raw) if isinstance(raw, str) else list(raw or [])
    seen: list[str] = []
    for item in items:
        name = normalize_ingredient(str(item))
        if name and name not in seen:
            seen.append(name)
    return seen[:MAX_USER_INGREDIENTS]


def match_credit(user_ingredient: str, recipe_ingredient: str) -> float:
    """1.0 for the same ingredient, PARTIAL_CREDIT if one name contains the other, else 0."""
    u, r = canonical(user_ingredient), canonical(recipe_ingredient)
    if u == r:
        return 1.0
    if u <= r or r <= u:
        return PARTIAL_CREDIT
    return 0.0


@dataclass(frozen=True)
class Overlap:
    have: list[str]
    missing: list[str]
    coverage: float  # share of the recipe's ingredients the user has, in [0, 1] ("can I make it?")
    usage: float     # share of the user's ingredients the recipe uses, in [0, 1] ("does it use what I have?")

    @property
    def considered(self) -> int:
        return len(self.have) + len(self.missing)


def score_overlap(user: list[str], recipe_ingredients: list[str], staples: bool = True) -> Overlap:
    """Compare a pantry with one recipe. With `staples`, salt/oil/water/... are ignored entirely."""
    have: list[str] = []
    missing: list[str] = []
    earned = 0.0
    for name in recipe_ingredients:
        if staples and is_staple(name):
            continue
        credit = max((match_credit(u, name) for u in user), default=0.0)
        if credit > 0:
            have.append(name)
            earned += credit
        else:
            missing.append(name)
    considered = len(have) + len(missing)
    return Overlap(
        have=have, missing=missing,
        coverage=earned / considered if considered else 0.0,
        usage=_usage(user, have + missing, staples),
    )


def _usage(user: list[str], considered: list[str], staples: bool) -> float:
    """Average best-match credit over the user's ingredients, ignoring staples the user typed."""
    pantry = [u for u in user if not (staples and is_staple(u))]
    if not pantry:
        return 0.0
    return sum(max((match_credit(u, r) for r in considered), default=0.0) for u in pantry) / len(pantry)


@dataclass(frozen=True)
class Weights:
    """Coverage alone favors tiny recipes (3 ingredients, all owned), so usage balances it."""

    coverage: float = 0.45
    usage: float = 0.20
    similarity: float = 0.20
    missing: float = 0.15


def hybrid_score(coverage: float, usage: float, similarity: float | None, missing_count: int, w: Weights) -> float:
    """Higher is better: reward coverage, pantry usage and semantic similarity; penalize missing items."""
    sim = min(max(similarity or 0.0, 0.0), 1.0)
    penalty = min(missing_count, MAX_MISSING) / MAX_MISSING
    return w.coverage * coverage + w.usage * usage + w.similarity * sim - w.missing * penalty
