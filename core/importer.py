"""Parsing, filtering and sampling for the Food.com RAW_recipes.csv dataset (no DB access)."""
import ast
import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .ingredients import classify_diet, normalize_ingredient

SNACK_TAGS = {"snacks", "appetizers", "finger-food", "snacks-sweet", "snacks-kid-friendly"}
HIGH_PROTEIN_MIN_G = 20.0  # protein %DV is relative to 50 g, so 40% DV == 20 g

_SPACES = re.compile(r"\s+")


@dataclass
class ParsedRecipe:
    source_id: int
    name: str
    description: str
    minutes: int
    ingredients_raw: list[str]
    ingredients: list[str]  # normalized, de-duplicated, order preserved
    steps: list[str]
    tags: list[str]
    nutrition: list[float]
    diet: str
    calories: float
    protein_g: float
    is_high_protein: bool
    is_snack: bool
    extra: dict = field(default_factory=dict)


# Food.com strips apostrophes ("don't" -> "don t"); put them back after title-casing.
_NOT_STEMS = "Don|Doesn|Didn|Isn|Aren|Wasn|Weren|Hasn|Haven|Hadn|Won|Wouldn|Couldn|Shouldn|Can|Ain|Mustn|Needn"
_END = r"(?![-\w])"  # don't touch things like "T-Bone"
_APOSTROPHES = [
    (re.compile(rf"\b({_NOT_STEMS}) T\b{_END}"), r"\1't"),
    (re.compile(rf"\b(I) M\b{_END}"), r"\1'm"),
    (re.compile(rf"\b(I|You|We|They) Ve\b{_END}"), r"\1've"),
    (re.compile(rf"\b(You|We|They) Re\b{_END}"), r"\1're"),
    (re.compile(rf"\b(I|You|We|They|He|She|It|That|There) Ll\b{_END}"), r"\1'll"),
    (re.compile(rf"\b([A-Za-z]{{2,}}) S\b{_END}"), r"\1's"),
]


def _title(text: str) -> str:
    title = " ".join(w.capitalize() for w in _SPACES.split(text.strip()))
    for pattern, replacement in _APOSTROPHES:
        title = pattern.sub(replacement, title)
    return title


def _sentence(text: str) -> str:
    text = _SPACES.sub(" ", text).strip()
    return text[:1].upper() + text[1:]


def parse_row(row: dict) -> ParsedRecipe | None:
    """Return a clean recipe, or None if the row fails a quality check."""
    try:
        minutes = int(row["minutes"])
        n_steps = int(row["n_steps"])
        n_ingredients = int(row["n_ingredients"])
    except (KeyError, ValueError):
        return None
    # Cheap checks first so we only literal_eval rows that can pass.
    if not (5 <= minutes <= 180 and 3 <= n_steps <= 15 and 3 <= n_ingredients <= 15):
        return None

    name = _title(row.get("name") or "")
    if len(name) < 3:
        return None

    try:
        tags = ast.literal_eval(row["tags"])
        nutrition = [float(x) for x in ast.literal_eval(row["nutrition"])]
        steps = ast.literal_eval(row["steps"])
        raw_ingredients = ast.literal_eval(row["ingredients"])
    except (ValueError, SyntaxError, KeyError):
        return None
    if len(nutrition) != 7 or not (50 <= nutrition[0] <= 1500):
        return None

    normalized: list[str] = []
    for text in raw_ingredients:
        n = normalize_ingredient(text)
        if n and n not in normalized:
            normalized.append(n)
    if len(normalized) < 3:
        return None

    protein_g = nutrition[4] * 0.5
    return ParsedRecipe(
        source_id=int(row["id"]),
        name=name,
        description=_sentence(row.get("description") or ""),
        minutes=minutes,
        ingredients_raw=[_SPACES.sub(" ", t).strip() for t in raw_ingredients],
        ingredients=normalized,
        steps=[_sentence(s) for s in steps if s.strip()],
        tags=tags,
        nutrition=nutrition,
        diet=classify_diet(normalized),
        calories=nutrition[0],
        protein_g=protein_g,
        is_high_protein="high-protein" in tags or protein_g >= HIGH_PROTEIN_MIN_G,
        is_snack=bool(SNACK_TAGS & set(tags)),
    )


def read_candidates(path) -> list[ParsedRecipe]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in (parse_row(row) for row in csv.DictReader(f)) if r]


def select_recipes(candidates: list[ParsedRecipe], limit: int, seed: int = 42) -> list[ParsedRecipe]:
    """Pick `limit` recipes, guaranteeing a minimum share of veg / high-protein / snack dishes.

    A plain "first N rows" import would leave the discovery filters nearly empty.
    """
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)
    quotas = {"veg": int(limit * 0.4), "high_protein": int(limit * 0.2), "snack": int(limit * 0.2)}
    counts: dict[str, int] = defaultdict(int)
    chosen: list[ParsedRecipe] = []
    chosen_ids: set[int] = set()

    def flags(r: ParsedRecipe) -> dict[str, bool]:
        return {"veg": r.diet == "veg", "high_protein": r.is_high_protein, "snack": r.is_snack}

    def take(r: ParsedRecipe) -> None:
        chosen.append(r)
        chosen_ids.add(r.source_id)
        for key, on in flags(r).items():
            counts[key] += on

    for r in pool:  # pass 1: satisfy quotas
        if len(chosen) >= limit:
            break
        f = flags(r)
        if any(f[k] and counts[k] < q for k, q in quotas.items()):
            take(r)
    for r in pool:  # pass 2: fill remaining slots
        if len(chosen) >= limit:
            break
        if r.source_id not in chosen_ids:
            take(r)
    return chosen
