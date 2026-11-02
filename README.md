# ChefVision

An end-to-end AI cooking assistant: recognize ingredients from a photo, let the user correct the list,
recommend recipes (stored + AI-generated), and generate a picture of the finished dish with a diffusion model.

**Stack:** Django, DRF, HTMX, PostgreSQL + pgvector, Celery + Redis, diffusion model (local GPU).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit values

docker compose up -d          # Postgres (pgvector) + Redis
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Check that everything is wired: open <http://localhost:8000/health/> — expect `"status": "ok"` with the
database, pgvector version, and Redis all reporting `ok`.

In a second terminal, start the worker:

```bash
celery -A config worker -l info
```

Verify the queue end to end:

```bash
python manage.py shell -c "from core.tasks import ping; print(ping.delay().get(timeout=10))"
# -> pong
```

## Project layout

```
config/      Django project (settings, urls, celery app)
core/        Models, importer, matching + ranking, embeddings, views, tests
templates/   Base layout, recipe pages, "Cook with my ingredients" page
```

## Recipe data (Phase 1)

Download `RAW_recipes.csv` from the Food.com Recipes and Interactions dataset on Kaggle into `data/`
(git-ignored), then:

```bash
python manage.py migrate
python manage.py import_recipes --limit 500 --reset
python manage.py test core
```

The importer drops low-quality rows (absurd durations, malformed fields), normalizes ingredient names into
an `Ingredient` table, derives `diet` (veg / egg / nonveg), protein in grams and high-protein / snack flags,
and samples with quotas so every discovery filter has results. Browse at <http://localhost:8000/recipes/>.

## Ingredient matching and retrieval (Phase 2)

Install the ML dependency once (if you have an NVIDIA GPU, install a CUDA build of torch first):

```bash
pip install -r requirements-ml.txt
python manage.py migrate                 # adds the pgvector HNSW index
python manage.py embed_recipes           # ~90 MB model download on first run
python manage.py evaluate_matching --markdown docs/eval.md
```

Try it at <http://localhost:8000/cook/> or through the API:

```bash
curl -X POST localhost:8000/api/match/ -H "Content-Type: application/json" \
     -d '{"ingredients": ["chicken", "tomato", "rice"], "diet": "nonveg", "limit": 5}'
```

How ranking works. Hard filters (diet, time, snack, high-protein) are applied first. Each remaining
recipe that shares an ingredient with the user's pantry is then scored from four signals:

| Signal | Meaning | Default weight |
|---|---|---|
| Coverage | share of the recipe's ingredients you have ("can I make it?") | 0.45 |
| Usage | share of *your* ingredients the recipe uses ("does it use what I have?") | 0.20 |
| Semantic similarity | cosine similarity of MiniLM embeddings (pgvector) | 0.20 |
| Missing penalty | penalty for each ingredient you lack, saturating at 8 | 0.15 |

Matching is word-based and ignores descriptors ("fresh", "chopped"); "chicken" earns partial credit
against "chicken breast". Salt, oil, water, sugar and flour are assumed available unless turned off.
Weights live in `RANKING_WEIGHTS` in `config/settings.py`. If the embedding model is unavailable the
app falls back to overlap-only ranking instead of failing.

`evaluate_matching` is a self-retrieval benchmark: reveal 60% of a stored recipe's ingredients and check
whether the ranker finds that recipe. It compares naive coverage, overlap, semantic-only and hybrid
ranking. It is a proxy metric, not human judgement. Tests use a built-in `hash` stand-in embedder
(`EMBEDDER_BACKEND=hash`) so they run offline; its numbers are not meaningful.

## Roadmap

0. Foundation (done)
1. Recipe library + discovery page (done)
2. Embedding-based retrieval (pgvector) (done)
3. Ingredient recognition with a user-correction step
4. Recommendations + AI recipe generation
5. Dish image generation (async)
6. Evaluation and polish
7. Packaging and deployment
