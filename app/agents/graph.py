"""LangGraph agent graph for the service desk workflow.

Flow:
    START → agent (tool-calling loop) → END

The agent has three tools:
  1. search_knowledge_base  — RAG over company docs
  2. lookup_employee        — HR directory
  3. post_answer            — Slack reply with optional ticket escalation button
"""

import logging

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from app.config import settings
from app.tools.hr_lookup import lookup_employee
from app.tools.rag import search_knowledge_base
from app.tools.slack_reply import post_answer

log = logging.getLogger(__name__)

TOOLS = [search_knowledge_base, lookup_employee, post_answer]

SYSTEM_PROMPT = """You are an IT service desk agent. Follow these steps in order — do not skip any:
1. Call search_knowledge_base with a relevant query.
2. Call lookup_employee with the requester's name (if mentioned; use "unknown" if not).
3. Call post_answer with: a helpful answer containing 2-3 specific steps sourced from the knowledge base (not generic advice), a one-line question_summary, requester_name and requester_email from your HR lookup, and the slack_channel and thread_ts values.
You MUST call post_answer as the final step. Never call the same tool twice."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    slack_channel: str
    slack_ts: str


def _make_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=1024,
    ).bind_tools(TOOLS)


async def agent_node(state: AgentState) -> dict:
    llm = _make_llm()
    prompt = SYSTEM_PROMPT + f"\n\nslack_channel={state['slack_channel']} slack_ts={state['slack_ts']}"
    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = await llm.ainvoke(messages)
    tool_names = [tc["name"] for tc in (response.tool_calls or [])]
    log.info("Agent turn %d — tools called: %s | text: %s",
             len(state["messages"]), tool_names or "none (final)", str(response.content)[:120])
    return {"messages": [response]}


def build_graph() -> StateGraph:
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


# Module-level compiled graph — reused across requests
service_desk_graph = build_graph()
