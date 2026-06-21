"""Slack tool: post answers with an optional ticket-creation button."""

import json
import logging

from langchain_core.tools import tool
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from app.config import settings

log = logging.getLogger(__name__)

_slack = AsyncWebClient(token=settings.slack_bot_token)


@tool
async def post_answer(
    channel: str,
    thread_ts: str,
    answer: str,
    question_summary: str,
    requester_name: str = "",
    requester_email: str = "",
) -> str:
    """Post the answer to the Slack thread with a button so the user can escalate to a Jira ticket if needed.

    Args:
        channel: Slack channel ID (e.g. C01234ABC).
        thread_ts: Timestamp of the original message to reply to.
        answer: The answer with 2-3 specific actionable steps from the knowledge base.
        question_summary: One-line summary of the question (used as Jira title if a ticket is created).
        requester_name: Employee's name from the HR lookup.
        requester_email: Employee's email from the HR lookup.
    """
    button_value = json.dumps({
        "channel": channel,
        "ts": thread_ts,
        "summary": question_summary[:200],
        "name": requester_name,
        "email": requester_email,
    })

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": answer},
        },
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🎫 Issue not resolved — create ticket"},
                    "style": "primary",
                    "action_id": "create_ticket",
                    "value": button_value,
                }
            ],
        },
    ]

    if channel in ("", "demo", "none"):
        log.info("Slack answer (dry-run): %s", answer)
        return "Answer posted (test mode)."

    try:
        await _slack.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=answer,
            blocks=blocks,
            unfurl_links=False,
        )
        return "Answer posted to Slack with ticket escalation button."
    except SlackApiError as exc:
        log.error("Slack API error: %s", exc.response["error"])
        return f"Failed to post answer: {exc.response['error']}"
