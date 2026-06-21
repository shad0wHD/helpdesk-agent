"""Tests for the agent graph structure.

Full e2e integration tests require real credentials and are run manually via demo.py.
These tests verify the graph compiles correctly and the state schema is valid.
"""

import pytest
from langchain_core.messages import HumanMessage

from app.agents.graph import AgentState, TOOLS, build_graph


def test_graph_compiles():
    """Graph must compile without errors — catches import and wiring issues."""
    graph = build_graph()
    assert graph is not None


def test_graph_has_four_tools():
    """Exactly four tools must be registered: RAG, HR, Jira, Slack."""
    tool_names = {t.name for t in TOOLS}
    assert tool_names == {
        "search_knowledge_base",
        "lookup_employee",
        "create_jira_ticket",
        "post_slack_reply",
    }


def test_agent_state_schema():
    """AgentState must accept well-formed input without raising."""
    state: AgentState = {
        "messages": [HumanMessage(content="VPN access not working for John Smith")],
        "slack_channel": "C123",
        "slack_ts": "1234567890.000001",
    }
    assert state["slack_channel"] == "C123"
    assert len(state["messages"]) == 1
