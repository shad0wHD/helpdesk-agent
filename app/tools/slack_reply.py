"""Slack tool: post threaded replies back to the originating channel."""

import logging

from langchain_core.tools import tool
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from app.config import settings

log = logging.getLogger(__name__)

_slack = AsyncWebClient(token=settings.slack_bot_token)


@tool
async def post_slack_reply(channel: str, thread_ts: str, message: str) -> str:
    """Post a reply in the Slack thread where the request originated.

    Always call this as the final step to close the loop with the user.

    Args:
        channel: Slack channel ID (e.g. C01234ABC).
        thread_ts: Timestamp of the original message to reply to.
        message: The formatted response to send, including ticket link and next steps.
    """
    # When called from the HTTP test endpoint there's no real channel
    if channel in ("", "demo", "none"):
        log.info("Slack reply (dry-run, no channel): %s", message)
        return f"[DRY RUN — no Slack channel] Message:\n{message}"

    try:
        await _slack.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=message,
            unfurl_links=False,
        )
        return "Reply posted to Slack."
    except SlackApiError as exc:
        log.error("Slack API error: %s", exc.response["error"])
        return f"Failed to post Slack reply: {exc.response['error']}"
