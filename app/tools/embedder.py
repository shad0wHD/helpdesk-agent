"""Local text embeddings via fastembed (no API key required).

Model: BAAI/bge-small-en-v1.5 — 384 dimensions, ~130MB, CPU-only.
Downloaded automatically on first use and cached in ~/.cache/fastembed.
"""

from functools import lru_cache

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)


async def embed_text(text: str) -> list[float]:
    """Return a 384-dim embedding vector. Runs synchronously (CPU-bound, fast enough)."""
    embeddings = list(_model().embed([text]))
    return embeddings[0].tolist()


async def embed_batch(texts: list[str]) -> list[list[float]]:
    embeddings = list(_model().embed(texts))
    return [e.tolist() for e in embeddings]
