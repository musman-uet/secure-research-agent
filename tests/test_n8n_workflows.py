"""Static checks for the n8n workflows. (They were also imported and executed in a
real n8n 2.48 instance; see docs/n8n.md.)"""

import json
from pathlib import Path

import pytest

N8N = Path(__file__).resolve().parent.parent / "n8n"
WORKFLOWS = sorted(N8N.glob("*.json"))


def test_there_are_five_workflows():
    assert len(WORKFLOWS) == 5


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_structure(path):
    wf = json.loads(path.read_text())
    names = [n["name"] for n in wf["nodes"]]
    assert len(names) == len(set(names)), "node names must be unique"
    for source, outputs in wf["connections"].items():
        assert source in names
        for branches in outputs.values():
            for branch in branches:
                for target in branch:
                    assert target["node"] in names
    for n in wf["nodes"]:
        assert {"id", "name", "type", "typeVersion", "position", "parameters"} <= set(n)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflows_contain_no_secrets(path):
    text = path.read_text()
    assert "demo-" not in text and "sk-" not in text and "Bearer " not in text
    tokens = json.loads((N8N.parent / "data" / "tokens.json").read_text())["tokens"]
    assert not any(t in text for t in tokens)


def test_agent_workflow_wires_mcp_tool_and_model_to_the_agent():
    wf = json.loads((N8N / "03-chat-agent-with-mcp.json").read_text())
    c = wf["connections"]
    assert c["Secure Research Agent (MCP)"]["ai_tool"][0][0]["node"] == "AI Agent"
    assert c["OpenAI Chat Model"]["ai_languageModel"][0][0]["node"] == "AI Agent"
    mcp = next(n for n in wf["nodes"] if n["name"] == "Secure Research Agent (MCP)")
    assert mcp["parameters"]["endpointUrl"].endswith("/mcp")
    assert mcp["parameters"]["authentication"] == "bearerAuth"
    assert mcp["parameters"]["includeTools"] == ["research", "list_documents"]


def test_approval_workflow_waits_for_a_human():
    wf = json.loads((N8N / "04-approval-before-external-share.json").read_text())
    slack = next(n for n in wf["nodes"] if n["type"] == "n8n-nodes-base.slack")
    assert slack["parameters"]["operation"] == "sendAndWait"
    assert len(wf["connections"]["Approved?"]["main"]) == 2  # approved / rejected branches
