"""RAG tool: semantic search over the pgvector knowledge base.

Requires ANTHROPIC_API_KEY (for Voyage-3 embeddings via the voyageai package).
When the key is absent the tool returns a graceful message and the agent continues
without KB context — Jira ticket and Slack reply still work normally.
"""

import logging

from langchain_core.tools import tool

log = logging.getLogger(__name__)


@tool
async def search_knowledge_base(query: str, k: int = 4) -> str:
    """Search the company knowledge base for documentation relevant to the query.

    Returns the top-k most relevant passages with their source titles.
    Use this before creating a Jira ticket to find existing solutions or runbooks.
    """
    from app.config import settings

    if not settings.anthropic_api_key:
        log.warning("RAG search skipped: ANTHROPIC_API_KEY not configured.")
        return (
            "Knowledge base search unavailable (no embedding API key configured). "
            "Proceed with HR lookup and ticket creation based on the request description."
        )

    try:
        from app.db.session import AsyncSessionLocal
        from app.db.models import Document
        from app.tools.embedder import embed_text
        from sqlalchemy import select

        embedding = await embed_text(query)

        async with AsyncSessionLocal() as session:
            results = await session.execute(
                select(Document.title, Document.content, Document.source)
                .order_by(Document.embedding.cosine_distance(embedding))
                .limit(k)
            )
            rows = results.all()

        if not rows:
            return "No relevant documentation found in the knowledge base."

        passages = [f"**{title}** (source: {source})\n{content}" for title, content, source in rows]
        return "\n\n---\n\n".join(passages)

    except Exception as exc:
        log.error("RAG search failed: %s", exc)
        return f"Knowledge base search failed: {exc}. Proceed with ticket creation."
