from django import forms


class RecipeFilterForm(forms.Form):
    SORTS = [
        ("best", "Best match"),
        ("quickest", "Quickest"),
        ("protein", "Most protein"),
        ("calories", "Fewest calories"),
        ("name", "A to Z"),
    ]
    DIETS = [("", "Any diet"), ("veg", "Vegetarian"), ("nonveg", "Non-vegetarian")]
    TIMES = [("", "Any time"), ("15", "Under 15 min"), ("30", "Under 30 min"), ("60", "Under 1 hour")]

    q = forms.CharField(required=False, max_length=100, label="Search",
                        widget=forms.TextInput(attrs={"placeholder": "Search recipes or ingredients, e.g. paneer, lentil"}))
    diet = forms.ChoiceField(required=False, choices=DIETS)
    high_protein = forms.BooleanField(required=False, label="High protein")
    snack = forms.BooleanField(required=False, label="Snacks")
    max_minutes = forms.ChoiceField(required=False, choices=TIMES)
    sort = forms.ChoiceField(required=False, choices=SORTS)


class CookForm(forms.Form):
    """'Cook with my ingredients': what the user has plus the constraints to respect."""

    STAPLES = [("yes", "Assume salt, oil, water, sugar, flour"), ("no", "Don't assume any staples")]

    ingredients = forms.CharField(
        required=False, max_length=600, label="Ingredients",
        widget=forms.TextInput(attrs={"placeholder": "What do you have? e.g. chicken, tomato, rice, onion", "autocomplete": "off"}),
    )
    diet = forms.ChoiceField(required=False, choices=RecipeFilterForm.DIETS)
    max_minutes = forms.ChoiceField(required=False, choices=RecipeFilterForm.TIMES)
    high_protein = forms.BooleanField(required=False, label="High protein")
    snack = forms.BooleanField(required=False, label="Snacks")
    staples = forms.ChoiceField(required=False, choices=STAPLES)

    @property
    def assume_staples(self) -> bool:
        return self.cleaned_data.get("staples") != "no"
