"""MCP server (in-process and over real HTTP), REST API and tracing."""

import json
import socket
import threading
import time

import anyio
import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from mcp import Client

from research_agent.mcp_server import build_http_app, build_mcp_server

MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
               "MCP-Protocol-Version": "2025-06-18"}


def _text(result):
    return json.loads(result.content[0].text)


# ------------------------------------------------------------ MCP in-process

def test_mcp_tools_are_listed_as_read_only(agent, directory):
    srv = build_mcp_server(agent, identity=lambda: directory.get_user("bilal"))

    async def main():
        async with Client(srv) as c:
            return (await c.list_tools()).tools

    tools = anyio.run(main)
    assert {t.name for t in tools} == {"research", "search_documents", "list_documents", "read_section",
                                       "compare_documents"}
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools)
    for t in tools:  # identity never comes from arguments
        assert not {"user", "user_id", "groups"} & set(t.input_schema.get("properties", {}))


def test_mcp_research_tool(agent, directory):
    srv = build_mcp_server(agent, identity=lambda: directory.get_user("bilal"))

    async def main():
        async with Client(srv) as c:
            return await c.call_tool("research", {"question": "What changed between the 2023 and 2025 travel policy?"})

    report = _text(anyio.run(main))
    assert report["status"] == "answered" and "PKR 4,500" in report["summary"]


def test_mcp_without_identity_is_an_error(agent):
    srv = build_mcp_server(agent, identity=lambda: None)

    async def main():
        async with Client(srv) as c:
            return await c.call_tool("list_documents", {})

    result = anyio.run(main)
    assert result.is_error


# ------------------------------------------------------------ MCP over HTTP

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def mcp_url(agent, tokens):
    port = _free_port()
    app = build_http_app(agent, tokens)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)


def _rpc(url, token, method, params=None):
    headers = dict(MCP_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.post(url, headers=headers, timeout=10,
                      json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})


def test_http_mcp_rejects_missing_and_invalid_tokens(mcp_url):
    assert _rpc(mcp_url, None, "tools/list").status_code == 401
    r = _rpc(mcp_url, "not-a-real-token", "tools/list")
    assert r.status_code == 401 and "Bearer" in r.headers.get("www-authenticate", "")


def test_http_mcp_identity_comes_from_the_token(mcp_url, token_for):
    def docs(user):
        r = _rpc(mcp_url, token_for[user], "tools/call", {"name": "list_documents", "arguments": {}})
        assert r.status_code == 200
        return {d["doc_id"] for d in json.loads(r.json()["result"]["content"][0]["text"])["documents"]}

    assert "board-2025-10" in docs("omar")
    assert "board-2025-10" not in docs("zain")
    assert docs("zain") == {"it-security", "contractors"}


# ------------------------------------------------------------ REST API

@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    from research_agent import api as api_module

    api_module.get_agent.cache_clear()
    api_module.get_tokens.cache_clear()
    return TestClient(api_module.app)


def test_rest_research(api, token_for):
    r = api.post("/research", json={"question": "Who approves international travel and what is the mileage rate?"},
                 headers={"Authorization": f"Bearer {token_for['omar']}"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered" and body["ai_generated"] is True
    assert "department head" in body["summary"] and "PKR 65" in body["summary"]


def test_rest_requires_a_valid_token(api):
    assert api.post("/research", json={"question": "annual leave?"}).status_code == 401
    assert api.post("/research", json={"question": "annual leave?"},
                    headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert api.get("/health").json()["status"] == "ok"


# ------------------------------------------------------------ tracing

def test_otel_spans_follow_genai_conventions(agent, directory):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from research_agent import tracing

    exporter = InMemorySpanExporter()
    tracing.use_exporter(exporter)
    question = "What is the mileage reimbursement per km?"
    agent.run(question, directory.get_user("sara"))
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "invoke_agent secure-research-agent" in names and "execute_tool search" in names
    root = next(s for s in spans if s.name.startswith("invoke_agent"))
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["research.status"] == "answered"
    tool = next(s for s in spans if s.name == "execute_tool search")
    assert tool.attributes["gen_ai.tool.name"] == "search" and tool.parent.span_id == root.context.span_id
    assert all(question not in str(dict(s.attributes)) for s in spans)  # no question text in traces
