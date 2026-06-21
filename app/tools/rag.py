"""RAG tool: semantic search over the pgvector knowledge base."""

import logging

from langchain_core.tools import tool

log = logging.getLogger(__name__)


@tool
async def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for documentation relevant to the query.

    Returns the top-k most relevant passages with their source titles.
    Use this before creating a Jira ticket to find existing solutions or runbooks.
    """
    try:
        from sqlalchemy import select

        from app.db.models import Document
        from app.db.session import AsyncSessionLocal
        from app.tools.embedder import embed_text

        embedding = await embed_text(query)

        async with AsyncSessionLocal() as session:
            results = await session.execute(
                select(Document.title, Document.content, Document.source)
                .order_by(Document.embedding.cosine_distance(embedding))
                .limit(2)
            )
            rows = results.all()

        if not rows:
            return "No relevant documentation found in the knowledge base."

        passages = [
            f"[{title}]\n{content[:400]}"
            for title, content, source in rows
        ]
        return "\n---\n".join(passages)

    except Exception as exc:
        log.error("RAG search failed: %s", exc)
        return f"Knowledge base search failed: {exc}. Proceed with ticket creation."
