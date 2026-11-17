# n8n integration

The `n8n/` folder has five workflows. Import them in n8n with **Workflows → Import from file**.

| Workflow | What it does | Needs |
|---|---|---|
| `01-research-webhook.json` | `POST /webhook/research-question {"question": "..."}` → research agent → answer with sources | Research API |
| `02-weekly-policy-digest.json` | Every Monday 09:00: "What changed in the travel policy?" → formatted digest → Slack | Research API, Slack (node disabled until you add credentials) |
| `03-chat-agent-with-mcp.json` | n8n chat → **AI Agent** that uses the research agent as an **MCP tool** | MCP server, an OpenAI (or compatible) model |
| `04-approval-before-external-share.json` | Answer for an external request → **Slack approval (send and wait)** → send or log rejection | Research API, Slack |
| `05-mcp-direct-call.json` | Calls the `research` MCP tool directly, no LLM (connection test) | MCP server |

## Setup

1. Start the agent (on the machine running n8n, or reachable from it):

   ```bash
   python -m research_agent serve-api --port 8000   # for workflows 01, 02, 04
   python -m research_agent serve-mcp --port 8001   # for workflows 03, 05
   ```

   If n8n runs in Docker, start both with `--host 0.0.0.0` and change `localhost` in the workflows to
   `host.docker.internal`.

2. In n8n, create a credential of type **Bearer Auth** named `Secure Research Agent token`, and paste a token from
   `data/tokens.json` (demo tokens; each one maps to a sample user, so the answers change with the token).

3. For workflow 03, add an **OpenAI** credential (or any OpenAI-compatible endpoint) to the chat model node.

4. Publish the workflow.

## How each workflow was tested

All five were imported into a real **n8n 2.48.4** instance (Node.js 24). Then:

- **05 (MCP direct call):** executed with `n8n execute`. The MCP Client node connected over Streamable HTTP with the
  bearer token and returned the verified report.
- **02 (weekly digest):** executed with `n8n execute` through the "Run now" trigger. The Slack node is disabled, so
  it passed the data through.
- **01 (webhook):** published, then called with `curl`. It returned the formatted answer with sources.
- **03 (AI Agent + MCP tool):** run as a test copy with a webhook trigger, and with a scripted fake
  OpenAI-compatible model in place of OpenAI. n8n offered the model exactly the two allowed MCP tools
  (`research`, `list_documents`), executed the model's tool call against the MCP server, and returned the result.
  With a real model, the answer wording will differ.
- **04 (approval):** imported only. It needs a Slack workspace to wait for the approval click.
