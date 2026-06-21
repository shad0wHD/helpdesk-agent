"""Document ingestion CLI — add PDFs, text files, or markdown to the knowledge base.

Usage:
    python scripts/ingest.py path/to/runbook.pdf
    python scripts/ingest.py docs/                    # ingest a whole folder
    python scripts/ingest.py runbook.pdf --source confluence

Supported formats: .pdf, .txt, .md
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("SLACK_BOT_TOKEN", "ingest")
os.environ.setdefault("SLACK_SIGNING_SECRET", "ingest")
os.environ.setdefault("SLACK_APP_TOKEN", "ingest")
os.environ.setdefault("JIRA_BASE_URL", "https://ingest.atlassian.net")
os.environ.setdefault("JIRA_EMAIL", "ingest@ingest.com")
os.environ.setdefault("JIRA_API_TOKEN", "ingest")

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


async def ingest_file(path: Path, source: str) -> int:
    from sqlalchemy import text as sql_text

    from app.db.models import Document
    from app.db.session import AsyncSessionLocal, engine
    from app.tools.embedder import embed_batch

    # Ensure tables exist
    from app.db.models import Base
    async with engine.begin() as conn:
        await conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    raw_text = _extract_text(path)
    if not raw_text.strip():
        log.warning("No text extracted from %s — skipping.", path.name)
        return 0

    chunks = _chunk(raw_text)
    log.info("%s → %d chunks", path.name, len(chunks))

    embeddings = await embed_batch(chunks)

    async with AsyncSessionLocal() as session:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            title = f"{path.stem} (chunk {i + 1}/{len(chunks)})"
            session.add(Document(
                title=title,
                content=chunk,
                source=source,
                embedding=embedding,
            ))
        await session.commit()

    return len(chunks)


async def main(paths: list[Path], source: str) -> None:
    total = 0
    for path in paths:
        if path.is_dir():
            files = [f for f in path.rglob("*") if f.suffix.lower() in (".pdf", ".txt", ".md")]
            log.info("Found %d files in %s", len(files), path)
            for f in files:
                total += await ingest_file(f, source)
        elif path.exists():
            total += await ingest_file(path, source)
        else:
            log.error("Not found: %s", path)

    log.info("Done. Ingested %d total chunks.", total)


if __name__ == "__main__":
    import selectors

    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or folders to ingest")
    parser.add_argument("--source", default="upload", help="Source label (e.g. confluence, gdrive)")
    args = parser.parse_args()
    # psycopg (async) requires SelectorEventLoop on Windows
    asyncio.run(main(args.paths, args.source), loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))
