"""
Demo script: run the service desk agent without Slack or Jira credentials.

This is the 5-minute recruiter demo. It:
  - Spins up a real LangGraph agent with Claude
  - Searches the local knowledge base (no DB needed — uses in-memory vector search)
  - Looks up a mock employee
  - Simulates Jira ticket creation and Slack reply (prints instead of calling APIs)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo.py
"""

import asyncio
import os
import sys

# Simple in-memory vector search for the demo (no postgres needed)
from dataclasses import dataclass

# ── minimal stubs so the demo works without .env ─────────────────────────────
os.environ.setdefault("SLACK_BOT_TOKEN", "demo")
os.environ.setdefault("SLACK_SIGNING_SECRET", "demo")
os.environ.setdefault("SLACK_APP_TOKEN", "demo")
os.environ.setdefault("JIRA_BASE_URL", "https://demo.atlassian.net")
os.environ.setdefault("JIRA_EMAIL", "demo@demo.com")
os.environ.setdefault("JIRA_API_TOKEN", "demo")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://demo:demo@localhost/demo")
os.environ.setdefault("GROQ_API_KEY", "")

import json
import math
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: Set GROQ_API_KEY environment variable.")
    sys.exit(1)

# ── In-memory RAG (no pgvector needed for demo) ───────────────────────────────
_DOCS = json.loads((Path(__file__).parent / "data" / "knowledge_base.json").read_text())


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def _simple_embed(text: str) -> list[float]:
    """TF-style bag-of-words embedding (demo only — replace with real embeddings)."""
    words = re.findall(r"\w+", text.lower())
    vocab = ["vpn", "access", "password", "laptop", "software", "slack", "employee",
             "onboard", "reset", "jira", "ticket", "google", "drive", "license",
             "hardware", "escalat", "priority", "acme", "network", "install"]
    return [float(words.count(w)) for w in vocab]


# ── Demo tools (real logic, fake API calls) ───────────────────────────────────
_EMPLOYEES = {
    "john smith": {
        "name": "John Smith", "email": "john.smith@acme.com",
        "department": "Engineering", "manager": "Jane Doe",
        "employee_id": "EMP-1042", "vpn_group": "engineering-vpn",
    },
}

_ticket_counter = {"n": 41}

SYSTEM_PROMPT = """You are an enterprise IT service desk AI agent. Handle support requests from Slack.

For every request follow this exact sequence:
1. search_knowledge_base — find relevant runbooks or docs
2. lookup_employee — if a person is named, get their profile
3. create_jira_ticket — always create a tracking ticket
4. post_slack_reply — reply in the thread with ticket link and next steps

Be concise and professional. Never skip steps 3 or 4."""


@tool
async def search_knowledge_base(query: str, k: int = 3) -> str:
    """Search the company knowledge base for documentation relevant to the query."""
    q_vec = await _simple_embed(query)
    scored = []
    for doc in _DOCS:
        d_vec = await _simple_embed(doc["content"])
        scored.append((_cosine_sim(q_vec, d_vec), doc))
    scored.sort(key=lambda x: -x[0])
    top = scored[:k]
    passages = [f"**{d['title']}**\n{d['content'][:600]}..." for _, d in top]
    return "\n\n---\n\n".join(passages)


@tool
async def lookup_employee(name: str) -> str:
    """Look up an employee by name in the HR directory."""
    key = name.strip().lower()
    emp = _EMPLOYEES.get(key)
    if not emp:
        for k, v in _EMPLOYEES.items():
            if key in k:
                emp = v
                break
    if not emp:
        return f"Employee '{name}' not found."
    return "\n".join(f"{k.title()}: {v}" for k, v in emp.items())


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
    """Create a Jira service desk ticket. Returns the ticket key and URL."""
    _ticket_counter["n"] += 1
    key = f"SD-{_ticket_counter['n']}"
    url = f"https://demo.atlassian.net/browse/{key}"
    print(f"\n  {'─'*60}")
    print(f"  [JIRA] Created ticket {key}: {summary}")
    print(f"  Priority: {priority} | Requester: {requester_name or 'unknown'}")
    print(f"  {'─'*60}\n")
    return f"Jira ticket created: {key}\nURL: {url}"


@tool
async def post_slack_reply(channel: str, thread_ts: str, message: str) -> str:
    """Post a reply in the Slack thread where the request originated."""
    print(f"\n  {'─'*60}")
    print(f"  [SLACK] Reply to #{channel} (thread {thread_ts}):")
    print(f"  {message}")
    print(f"  {'─'*60}\n")
    return "Reply posted to Slack."


TOOLS = [search_knowledge_base, lookup_employee, create_jira_ticket, post_slack_reply]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    slack_channel: str
    slack_ts: str


def agent_node(state: AgentState) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=4096,
    ).bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_graph():
    tool_node = ToolNode(TOOLS)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_demo(message: str) -> None:
    print(f"\n{'='*64}")
    print(f"  SERVICE DESK AGENT — DEMO")
    print(f"{'='*64}")
    print(f"\n  [SLACK] @servicedesk {message}\n")

    graph = build_graph()
    result = await graph.ainvoke({
        "messages": [HumanMessage(content=message)],
        "slack_channel": "it-support",
        "slack_ts": "1750000000.000001",
    })

    last_ai = next(
        (m for m in reversed(result["messages"]) if m.type == "ai" and not m.tool_calls),
        None,
    )
    if last_ai:
        print(f"\n  [AGENT FINAL RESPONSE]\n  {last_ai.content}\n")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    scenario = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "VPN access isn't working for John Smith"
    )
    asyncio.run(run_demo(scenario))
