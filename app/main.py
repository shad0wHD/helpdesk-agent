"""FastAPI entrypoint.

Routes:
    POST /agent/run  — Direct HTTP API for testing without Slack
    GET  /health     — Liveness probe

Slack events arrive via Socket Mode (WebSocket), not HTTP — no /slack/events route needed.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from app.agents.graph import service_desk_graph
from app.db.init_db import init_db
from app.slack_handler import create_socket_handler

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Pre-load the fastembed ONNX model so the first Slack mention isn't slow
    from app.tools.embedder import embed_text
    await embed_text("warmup")
    socket_handler = create_socket_handler()
    # start_async() runs the WebSocket listener loop — must be a background task
    # so it doesn't block the lifespan and keeps running for the app's lifetime
    def _on_slack_task_done(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            log.error("Slack Socket Mode task crashed: %s", task.exception())

    slack_task = asyncio.create_task(socket_handler.start_async())
    slack_task.add_done_callback(_on_slack_task_done)
    log.info("Service desk agent ready — Slack Socket Mode listening.")
    yield
    slack_task.cancel()
    try:
        await slack_task
    except asyncio.CancelledError:
        pass
    log.info("Slack Socket Mode disconnected.")


app = FastAPI(title="Service Desk Agent", version="0.1.0", lifespan=lifespan)


@app.post("/agent/run")
async def run_agent(req: Request) -> JSONResponse:
    """Direct HTTP endpoint — test the agent without going through Slack."""
    body = await req.json()
    message: str = body.get("message", "")
    channel: str = body.get("channel", "demo")
    ts: str = body.get("ts", "0")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=422)

    result = await service_desk_graph.ainvoke(
        {
            "messages": [HumanMessage(content=message)],
            "slack_channel": channel,
            "slack_ts": ts,
        },
        config={"recursion_limit": 12},
    )

    last_ai = next(
        (m for m in reversed(result["messages"]) if m.type == "ai" and not m.tool_calls),
        None,
    )
    return JSONResponse(
        {
            "response": last_ai.content if last_ai else "Agent completed with no final message.",
            "message_count": len(result["messages"]),
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
