"""Self-retrieval benchmark for the ingredient matcher.

Take a stored recipe, reveal only a fraction of its (non-staple) ingredients as the "pantry", and
check whether the ranker puts that recipe near the top. Compare naive coverage, overlap-only,
semantic-only and hybrid ranking. This is a proxy for relevance, not a human judgement, and it says so in its output.
"""
import json
import math
import random
import statistics
import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core.embeddings import EmbedderUnavailable, get_embedder
from core.matching import Weights, is_staple
from core.models import Recipe
from core.retrieval import default_weights, rank_recipes

CUTOFF = 50  # a recipe not in the top 50 counts as a miss


class Command(BaseCommand):
    help = "Benchmark naive coverage vs overlap vs semantic-only vs hybrid ranking with a self-retrieval test."

    def add_arguments(self, parser):
        parser.add_argument("--n", type=int, default=200, help="Number of query recipes (default 200).")
        parser.add_argument("--keep", type=float, default=0.6, help="Fraction of ingredients revealed (default 0.6).")
        parser.add_argument("--min-ingredients", type=int, default=4, help="Skip recipes with fewer non-staple ingredients.")
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--backend", default=None, help="Embedding backend override.")
        parser.add_argument("--markdown", default=None, help="Also write the results table to this file.")
        parser.add_argument("--json", dest="json_path", default=None, help="Also write raw metrics to this file.")

    def handle(self, *args, n, keep, min_ingredients, seed, backend, markdown, json_path, **options):
        try:
            embedder = get_embedder(backend)
        except EmbedderUnavailable as exc:
            raise CommandError(str(exc)) from exc
        if not Recipe.objects.filter(source=Recipe.Source.STORED, embedding__isnull=False).exists():
            raise CommandError("No recipe embeddings found. Run: python manage.py embed_recipes")

        rng = random.Random(seed)
        pool = []
        for recipe in Recipe.objects.filter(source=Recipe.Source.STORED).prefetch_related("ingredients"):
            names = [i.name for i in recipe.ingredients.all() if not is_staple(i.name)]
            if len(names) >= min_ingredients:
                pool.append((recipe.pk, sorted(names)))
        pool.sort()
        rng.shuffle(pool)
        queries = []
        for pk, names in pool[:n]:
            k = max(2, math.ceil(len(names) * keep))
            queries.append((pk, rng.sample(names, k)))
        if not queries:
            raise CommandError("Not enough recipes with enough ingredients to evaluate.")

        base = default_weights()
        methods = {
            "Coverage only (naive)": dict(weights=Weights(coverage=1.0, usage=0.0, similarity=0.0, missing=0.0), candidates="matching"),
            "Overlap (coverage + usage + missing)": dict(
                weights=Weights(coverage=base.coverage, usage=base.usage, similarity=0.0, missing=base.missing), candidates="matching"),
            "Semantic only": dict(weights=Weights(coverage=0.0, usage=0.0, similarity=1.0, missing=0.0), candidates="all"),
            "Hybrid (all signals)": dict(weights=base, candidates="matching"),
        }

        rows, raw = [], {}
        for label, options_ in methods.items():
            ranks, times = [], []
            for target, pantry in queries:
                t0 = time.perf_counter()
                result = rank_recipes(pantry, limit=CUTOFF, embedder=embedder, **options_)
                times.append((time.perf_counter() - t0) * 1000)
                ids = [item.recipe.pk for item in result.items]
                ranks.append(ids.index(target) + 1 if target in ids else None)
            hit = lambda k: sum(1 for r in ranks if r is not None and r <= k) / len(ranks)  # noqa: E731
            mrr = sum(1 / r for r in ranks if r) / len(ranks)
            rows.append((label, hit(1), hit(5), hit(10), mrr, statistics.mean(times)))
            raw[label] = {"recall@1": hit(1), "recall@5": hit(5), "recall@10": hit(10), "mrr": mrr, "ms_per_query": statistics.mean(times)}

        lines = [
            f"Self-retrieval benchmark: {len(queries)} queries, {int(keep * 100)}% of ingredients revealed, "
            f"backend '{embedder.name}', weights {base}.",
            "",
            "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |",
            "|---|---|---|---|---|---|",
        ]
        lines += [f"| {m} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {e:.1f} |" for m, a, b, c, d, e in rows]
        lines += ["", f"Proxy metric: the target recipe is in the corpus; ranks beyond {CUTOFF} count as misses."]
        if embedder.name == "hash":
            lines.append("WARNING: 'hash' is a stand-in embedder, not a semantic model. Do not quote these numbers.")
        report = "\n".join(lines)
        self.stdout.write(report)
        if markdown:
            Path(markdown).write_text(report + "\n", encoding="utf-8")
        if json_path:
            Path(json_path).write_text(json.dumps({"queries": len(queries), "keep": keep, "methods": raw}, indent=2), encoding="utf-8")
