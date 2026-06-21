"""Jira tool: creates and updates service desk tickets via the Jira REST API v3."""

import logging
from base64 import b64encode

import httpx
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.db.models import TicketLog
from app.db.session import AsyncSessionLocal

log = logging.getLogger(__name__)


def _auth_header() -> str:
    token = b64encode(
        f"{settings.jira_email}:{settings.jira_api_token}".encode()
    ).decode()
    return f"Basic {token}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _post_jira(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{settings.jira_base_url}/rest/api/3/issue",
            json=payload,
            headers={
                "Authorization": _auth_header(),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()


@tool
async def create_jira_ticket(
    summary: str,
    description: str,
    priority: str = "Medium",
    requester_name: str = "",
    requester_email: str = "",
    slack_channel: str = "",
    slack_ts: str = "",
) -> str:
    """Create a Jira service desk ticket and return the ticket key and URL.

    Args:
        summary: One-line ticket title.
        description: Full description of the issue including context from RAG and HR lookup.
        priority: One of Highest, High, Medium, Low, Lowest.
        requester_name: Name of the employee who needs help.
        requester_email: Email of the employee.
        slack_channel: Slack channel ID where the request originated.
        slack_ts: Slack message timestamp for threading replies.
    """
    adf_description = {
        "version": 1,
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": description}],
            }
        ],
    }

    if requester_name or requester_email:
        adf_description["content"].append(
            {
                "type": "paragraph",
                "content": [
                    {
                        "type": "text",
                        "text": f"Requester: {requester_name} <{requester_email}>",
                        "marks": [{"type": "em"}],
                    }
                ],
            }
        )

    payload = {
        "fields": {
            "project": {"key": settings.jira_project_key},
            "summary": summary,
            "description": adf_description,
            "issuetype": {"name": "Task"},
            "priority": {"name": priority},
        }
    }

    try:
        data = await _post_jira(payload)
    except httpx.HTTPStatusError as exc:
        log.error("Jira API error: %s", exc.response.text)
        return f"Failed to create Jira ticket: {exc.response.status_code} — {exc.response.text}"

    key = data["key"]
    url = f"{settings.jira_base_url}/browse/{key}"

    async with AsyncSessionLocal() as session:
        session.add(
            TicketLog(
                jira_key=key,
                summary=summary,
                requester=requester_name or "unknown",
                slack_channel=slack_channel,
                slack_ts=slack_ts,
            )
        )
        await session.commit()

    log.info("Created Jira ticket %s", key)
    return f"Jira ticket created: {key}\nURL: {url}"
