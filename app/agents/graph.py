"""LangGraph agent graph for the service desk workflow.

Flow:
    START → agent (tool-calling loop) → END

The agent has four tools:
  1. search_knowledge_base  — RAG over company docs
  2. lookup_employee        — HR directory
  3. create_jira_ticket     — Jira REST API
  4. post_slack_reply       — Slack thread reply

The system prompt enforces a strict execution order so the agent always
gathers context before creating tickets and always closes with a Slack reply.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from app.config import settings
from app.tools.hr_lookup import lookup_employee
from app.tools.jira import create_jira_ticket
from app.tools.rag import search_knowledge_base
from app.tools.slack_reply import post_slack_reply

TOOLS = [search_knowledge_base, lookup_employee, create_jira_ticket, post_slack_reply]

SYSTEM_PROMPT = """You are an enterprise IT service desk AI agent. Your job is to handle
support requests that come from Slack and resolve them efficiently.

For every request you MUST follow this sequence:
1. **Search the knowledge base** — find existing documentation, runbooks, or solutions.
2. **Look up the employee** — if a person is named, retrieve their HR profile for context.
3. **Create a Jira ticket** — always create a ticket to track the work. Write a clear,
   detailed description using what you learned from steps 1 and 2.
4. **Post a Slack reply** — reply in the original thread with:
   - A brief summary of the issue
   - The Jira ticket key and link
   - Recommended next steps (from the knowledge base)
   - ETA if the runbook mentions one

Be concise, professional, and action-oriented. Never skip steps 3 or 4.
Always pass the slack_channel and slack_ts to create_jira_ticket and post_slack_reply.
If post_slack_reply returns a [DRY RUN] message, that counts as success — summarise what was done and stop.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    slack_channel: str
    slack_ts: str


def _make_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=4096,
    ).bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    llm = _make_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
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
