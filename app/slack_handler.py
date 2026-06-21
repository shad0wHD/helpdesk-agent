"""Slack Bolt app — listens for @mentions via Socket Mode and routes to the LangGraph agent.

Socket Mode (SLACK_APP_TOKEN) means Slack opens an outbound WebSocket to us —
no public URL or ngrok required for local development.
"""

import json
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

    # Resolve the Slack user's display name so the agent can personalise the ticket
    try:
        user_info = await slack_app.client.users_info(user=event["user"])
        profile = user_info["user"]["profile"]
        requester = profile.get("display_name") or profile.get("real_name") or ""
    except Exception:
        requester = ""

    if requester:
        text = f"[Requester: {requester}] {text}"

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


@slack_app.action("create_ticket")
async def handle_create_ticket(ack, body, client) -> None:
    await ack()

    action = body["actions"][0]
    ctx = json.loads(action["value"])
    channel = ctx["channel"]
    thread_ts = ctx["ts"]
    summary = ctx.get("summary", "IT support request")
    requester_name = ctx.get("name", "")
    requester_email = ctx.get("email", "")
    bot_message_ts = body["container"]["message_ts"]

    try:
        from app.tools.jira import _post_jira
        from app.config import settings

        description_text = summary
        if requester_name:
            description_text += f"\n\nRequester: {requester_name}"
            if requester_email:
                description_text += f" <{requester_email}>"

        adf = {
            "version": 1,
            "type": "doc",
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": description_text}]}],
        }
        payload = {
            "fields": {
                "project": {"key": settings.jira_project_key},
                "summary": summary,
                "description": adf,
                "issuetype": {"name": "Task"},
                "priority": {"name": "Medium"},
            }
        }

        data = await _post_jira(payload)
        key = data["key"]
        url = f"{settings.jira_base_url}/browse/{key}"

        log.info("Created Jira ticket %s from Slack button", key)

        # Post ticket link in the thread
        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=f":white_check_mark: Ticket created: <{url}|{key}>. Our team will follow up with you soon.",
        )

        # Replace the button with a "ticket created" context line
        original_blocks = body["message"].get("blocks", [])
        answer_block = next((b for b in original_blocks if b.get("type") == "section"), None)
        updated_blocks = []
        if answer_block:
            updated_blocks.append(answer_block)
        updated_blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f":white_check_mark: Ticket raised: <{url}|{key}>"}],
        })
        await client.chat_update(
            channel=channel,
            ts=bot_message_ts,
            text=body["message"].get("text", summary),
            blocks=updated_blocks,
        )

    except Exception:
        log.exception("Failed to create ticket from button click")
        await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=":x: Failed to create the ticket. Please contact IT directly.",
        )
