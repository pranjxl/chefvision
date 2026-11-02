from django.contrib import admin

from .models import Ingredient, Recipe, RecipeIngredient


class RecipeIngredientInline(admin.TabularInline):
    model = RecipeIngredient
    extra = 0
    autocomplete_fields = ["ingredient"]


@admin.register(Recipe)
class RecipeAdmin(admin.ModelAdmin):
    list_display = ["name", "source", "diet", "minutes", "calories", "protein_g", "is_high_protein", "is_snack"]
    list_filter = ["source", "diet", "is_high_protein", "is_snack"]
    search_fields = ["name"]
    exclude = ["embedding"]
    inlines = [RecipeIngredientInline]


@admin.register(Ingredient)
class IngredientAdmin(admin.ModelAdmin):
    search_fields = ["name"]
