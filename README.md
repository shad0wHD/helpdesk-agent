# AI Service Desk Agent

A production-grade multi-agent system that automates IT support workflows across Slack, Jira, and a RAG-powered knowledge base.

**Demo in 30 seconds:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python demo.py "VPN access isn't working for John Smith"
```

## What it does

A user @mentions the bot in Slack:
> "VPN access isn't working for John Smith"

The agent autonomously:
1. **Searches the knowledge base** (RAG via pgvector) — retrieves the VPN troubleshooting runbook
2. **Looks up the employee** (HR directory) — finds John's department, manager, and VPN group
3. **Creates a Jira ticket** — structured description with context from steps 1–2
4. **Replies in the Slack thread** — ticket link, recommended fix, and ETA

Total time: ~10 seconds.

## Architecture

```
Slack Bolt
    │
    ▼
FastAPI  ──── /slack/events
              /agent/run (HTTP API for testing)
    │
    ▼
LangGraph Agent (ReAct loop)
    ├── search_knowledge_base  → pgvector (cosine similarity)
    ├── lookup_employee        → HR directory (mock / SCIM)
    ├── create_jira_ticket     → Jira REST API v3
    └── post_slack_reply       → Slack Web API
    │
    ▼
PostgreSQL + pgvector
(documents, ticket_log)
```

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Claude claude-sonnet-4-6 (Anthropic) |
| Agent framework | LangGraph |
| API | FastAPI + uvicorn |
| Vector store | PostgreSQL + pgvector |
| Embeddings | Voyage-3 (via Anthropic API) |
| Integrations | Slack Bolt, Jira REST API v3 |
| Retry / resilience | Tenacity (exponential backoff) |
| Packaging | pyproject.toml / hatch |
| Deployment | Docker Compose |

## Quick start

### Demo mode (no Slack/Jira needed)

```bash
git clone ...
cd service-desk-agent
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
python demo.py
# Or try a different scenario:
python demo.py "New employee Alice Chen needs laptop and software access"
python demo.py "John Smith can't log into Google Workspace"
```

### Full deployment

1. Copy `.env.example` to `.env` and fill in credentials
2. Start PostgreSQL and seed the knowledge base:
   ```bash
   docker compose up postgres -d
   python -m app.db.init_db
   ```
3. Run the app:
   ```bash
   uvicorn app.main:app --reload
   # or
   docker compose up
   ```
4. Expose your local server to Slack (for development):
   ```bash
   ngrok http 8000
   # Set https://<ngrok-url>/slack/events as your Slack app's event URL
   ```

### Slack app setup

1. Create a Slack app at api.slack.com/apps
2. Enable Socket Mode (for development) or HTTP Events
3. Subscribe to `app_mention` events
4. Add bot scopes: `chat:write`, `app_mentions:read`
5. Install to workspace and copy `SLACK_BOT_TOKEN`

### Jira setup

1. Create a free Jira Software project
2. Generate an API token at id.atlassian.com/manage-profile/security
3. Create a project with key `SD` (or set `JIRA_PROJECT_KEY` in `.env`)

## Adding to the knowledge base

Add documents to `data/knowledge_base.json` and re-run the seeder:

```bash
python -m app.db.init_db
```

In production, connect a Confluence or Google Drive importer — the `Document` model and embedding pipeline are ready.

## Tests

```bash
pytest tests/ -v
```

## Resume bullet

> Built a multi-agent IT service desk assistant integrating Slack, Jira, enterprise knowledge bases, and HR data sources to automate support workflows end-to-end. Used LangGraph for agent orchestration, pgvector for semantic retrieval, and the Anthropic API for reasoning — reducing mean ticket resolution time from manual triage to ~10 seconds.
