"""Slack Bolt app — listens for @mentions via Socket Mode and routes to the LangGraph agent.

Socket Mode (SLACK_APP_TOKEN) means Slack opens an outbound WebSocket to us —
no public URL or ngrok required for local development.
"""

import logging
import re

from langchain_core.messages import HumanMessage
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from app.agents.graph import service_desk_graph
from app.config import settings

log = logging.getLogger(__name__)

slack_app = AsyncApp(
    token=settings.slack_bot_token,
    signing_secret=settings.slack_signing_secret,
)

# Instantiated lazily inside the FastAPI lifespan (requires a running event loop)
def create_socket_handler() -> AsyncSocketModeHandler:
    return AsyncSocketModeHandler(slack_app, settings.slack_app_token)

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def _clean(text: str) -> str:
    return _MENTION_RE.sub("", text).strip()


@slack_app.event("app_mention")
async def handle_mention(event: dict, say, ack) -> None:
    await ack()

    channel = event["channel"]
    ts = event["ts"]
    text = _clean(event.get("text", ""))

    if not text:
        await say(text="I didn't catch that — what do you need help with?", thread_ts=ts)
        return

    log.info("Received mention in %s: %s", channel, text[:120])

    await say(text=":hourglass_flowing_sand: On it — running the service desk agent...", thread_ts=ts)

    try:
        await service_desk_graph.ainvoke(
            {
                "messages": [HumanMessage(content=text)],
                "slack_channel": channel,
                "slack_ts": ts,
            },
            config={"recursion_limit": 12},
        )
    except Exception:
        log.exception("Agent failed for message: %s", text)
        await say(
            text=":x: The agent encountered an error. Please try again or contact IT directly.",
            thread_ts=ts,
        )
