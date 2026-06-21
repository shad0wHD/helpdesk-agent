"""One-shot DB initialiser: creates tables and seeds the knowledge base."""

import asyncio
import json
import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Base, Document
from app.db.session import AsyncSessionLocal, engine

log = logging.getLogger(__name__)


async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables ready.")


async def seed_knowledge_base(session: AsyncSession) -> None:
    from app.config import settings

    if not settings.anthropic_api_key:
        log.warning(
            "ANTHROPIC_API_KEY not set — skipping knowledge base seeding. "
            "RAG will return no results until you seed the DB. "
            "Run `python -m app.db.init_db` after setting the key."
        )
        return

    from app.tools.embedder import embed_text

    seed_path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.json"
    docs = json.loads(seed_path.read_text())

    existing = (await session.execute(text("SELECT COUNT(*) FROM documents"))).scalar()
    if existing:
        log.info("Knowledge base already seeded (%d docs), skipping.", existing)
        return

    for doc in docs:
        embedding = await embed_text(doc["content"])
        session.add(
            Document(
                title=doc["title"],
                content=doc["content"],
                source=doc.get("source", "manual"),
                embedding=embedding,
            )
        )
    await session.commit()
    log.info("Seeded %d documents into knowledge base.", len(docs))


async def init_db() -> None:
    await create_tables()
    async with AsyncSessionLocal() as session:
        await seed_knowledge_base(session)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_db())
