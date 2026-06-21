"""Text embedding via Anthropic's voyage-3 model through the anthropic SDK."""

import anthropic

from app.config import settings

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


async def embed_text(text: str) -> list[float]:
    """Return a 1536-dim embedding vector for the given text."""
    response = await _client.embeddings.create(
        model="voyage-3",
        input=text,
    )
    return response.embeddings[0].values


async def embed_batch(texts: list[str]) -> list[list[float]]:
    response = await _client.embeddings.create(
        model="voyage-3",
        input=texts,
    )
    return [r.values for r in response.embeddings]
