"""Pure helpers for ingredient normalization and diet classification (no Django imports)."""
import re

_NON_ALPHA = re.compile(r"[^a-z\s'-]")
_SPACES = re.compile(r"\s+")

# Words that look plural but are not; singularizing them would corrupt the name.
_KEEP_AS_IS = {
    "hummus", "asparagus", "molasses", "couscous", "swiss", "grits", "greens", "watercress",
    "lemongrass", "cress", "bass", "citrus", "octopus", "series", "species", "chickpeas",
}
_IRREGULAR = {"leaves": "leaf", "loaves": "loaf", "halves": "half", "knives": "knife"}

MEAT_TERMS = {
    "chicken", "beef", "pork", "lamb", "mutton", "veal", "turkey", "duck", "goose", "venison",
    "bacon", "ham", "sausage", "salami", "pepperoni", "prosciutto", "chorizo", "bologna",
    "brisket", "steak", "hamburger", "wiener", "hotdog", "gelatin", "lard", "tallow", "liver",
    "fish", "salmon", "tuna", "cod", "tilapia", "halibut", "trout", "sardine", "sardines",
    "anchovy", "anchovies", "shrimp", "prawn", "crab", "lobster", "clam", "oyster", "mussel",
    "scallop", "squid", "calamari",
}
MEAT_PHRASES = {"hot dog", "ground meat", "ribs", "spareribs"}
# Phrases that contain a meat word but are vegetarian.
MEAT_EXCEPTIONS = {"hamburger bun", "hamburger roll", "hamburger buns", "hamburger rolls"}
EGG_TERMS = {"egg", "eggs"}


def normalize_ingredient(text: str) -> str:
    """'Fresh Tomatoes (diced)' -> 'fresh tomato'. Used for matching, not for display."""
    text = text.lower().split("(")[0]
    text = _NON_ALPHA.sub(" ", text)
    text = _SPACES.sub(" ", text).strip(" -'")
    if not text:
        return ""
    words = text.split(" ")
    words[-1] = _singular(words[-1])
    return " ".join(words)


def _singular(word: str) -> str:
    if word in _KEEP_AS_IS:
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


_MEAT_ALL = MEAT_TERMS | {_singular(t) for t in MEAT_TERMS}
_EGG_ALL = EGG_TERMS | {_singular(t) for t in EGG_TERMS}


def classify_diet(normalized_ingredients: list[str]) -> str:
    """Return 'nonveg', 'egg' or 'veg' using the Indian convention (egg is not 'veg').

    This is a keyword heuristic over ingredient names, good enough for filtering and
    spot-checkable; it does not catch hidden animal products (e.g. fish sauce is caught,
    worcestershire sauce is not).
    """
    has_egg = False
    for name in normalized_ingredients:
        if name.startswith(tuple(MEAT_EXCEPTIONS)):
            continue
        tokens = set(name.split(" "))
        if tokens & _MEAT_ALL or any(p in name for p in MEAT_PHRASES):
            return "nonveg"
        if tokens & _EGG_ALL:
            has_egg = True
    return "egg" if has_egg else "veg"
