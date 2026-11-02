"""Recipe filters shared by the discovery page and the ingredient matcher."""
from .models import Recipe


def apply_recipe_filters(qs, *, diet="", high_protein=False, snack=False, max_minutes=None):
    if diet == "veg":
        qs = qs.filter(diet=Recipe.Diet.VEG)
    elif diet == "nonveg":  # eggs count as non-veg, following Indian food labelling
        qs = qs.exclude(diet=Recipe.Diet.VEG)
    if high_protein:
        qs = qs.filter(is_high_protein=True)
    if snack:
        qs = qs.filter(is_snack=True)
    if max_minutes:
        qs = qs.filter(minutes__lte=int(max_minutes))
    return qs
