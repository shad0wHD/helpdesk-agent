from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Company knowledge base documents stored with embeddings for RAG."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(256))  # e.g. "confluence", "gdrive"
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TicketLog(Base):
    """Audit log of every Jira ticket the agent creates."""

    __tablename__ = "ticket_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    jira_key: Mapped[str] = mapped_column(String(64))
    summary: Mapped[str] = mapped_column(String(512))
    requester: Mapped[str] = mapped_column(String(256))
    slack_channel: Mapped[str] = mapped_column(String(128))
    slack_ts: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
