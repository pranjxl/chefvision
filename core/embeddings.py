"""Text embedding backends for recipe retrieval.

Production uses sentence-transformers (all-MiniLM-L6-v2, 384 dims, matching Recipe.embedding).
`HashEmbedder` is a dependency-free stand-in: deterministic, same dimensionality, used by the
test-suite and as a smoke-test backend. It is NOT a semantic model; do not report its metrics.
"""
import hashlib
import logging
import math
import re
from functools import lru_cache

from django.conf import settings

logger = logging.getLogger(__name__)

DIMENSIONS = 384
_WORD = re.compile(r"[a-z]+")


class EmbedderUnavailable(RuntimeError):
    """The configured embedding backend can't be used (missing package, model not downloadable)."""


def recipe_embedding_text(name: str, ingredient_names: list[str], tags: list[str] | None = None) -> str:
    """The text we embed for a recipe: title + normalized ingredients + a few descriptive tags.

    Steps are deliberately excluded: they are long and mostly cooking verbs, which drowns the
    signal that matters for "what can I make with these ingredients".
    """
    parts = [f"{name}.", f"Ingredients: {', '.join(ingredient_names)}."]
    if tags:
        parts.append(f"Tags: {', '.join(tags)}.")
    return " ".join(parts)


def query_text(user_ingredients: list[str]) -> str:
    """Query-side counterpart of recipe_embedding_text."""
    return f"Ingredients: {', '.join(user_ingredients)}."


class HashEmbedder:
    """Signed feature-hashing bag-of-words, L2-normalized. Deterministic and offline."""

    name = "hash"
    dimensions = DIMENSIONS

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * DIMENSIONS
        for token in _WORD.findall(text.lower()):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "little") % DIMENSIONS
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 (or any 384-dim sentence-transformers model). Loaded lazily, once."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EmbedderUnavailable(
                "sentence-transformers is not installed. Run: pip install -r requirements-ml.txt"
            ) from exc
        self.name = model_name
        logger.info("Loading embedding model %s", model_name)
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - network / cache problems
            raise EmbedderUnavailable(f"Could not load embedding model {model_name!r}: {exc}") from exc
        self.dimensions = self._model.get_sentence_embedding_dimension()
        if self.dimensions != DIMENSIONS:
            raise EmbedderUnavailable(
                f"{model_name} produces {self.dimensions}-dim vectors but Recipe.embedding stores {DIMENSIONS}."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        return [[float(x) for x in v] for v in vectors]


@lru_cache(maxsize=4)
def _build(backend: str, model_name: str):
    if backend == "hash":
        return HashEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise EmbedderUnavailable(f"Unknown EMBEDDER_BACKEND {backend!r} (use 'sentence-transformers' or 'hash').")


def get_embedder(backend: str | None = None):
    """Return the configured embedder (cached, so the model loads once per process)."""
    return _build(backend or settings.EMBEDDER_BACKEND, settings.EMBEDDING_MODEL)
