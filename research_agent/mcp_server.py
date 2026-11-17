"""MCP server: the research agent as tools for Claude, Copilot, n8n or any MCP client.

Transport: Streamable HTTP at /mcp (what n8n's MCP Client Tool node uses).
Auth: Bearer token on every request; the user is resolved from the token,
never from tool arguments. All tools are read-only.

    python -m research_agent serve-mcp --port 8001
"""

from __future__ import annotations

import os
from typing import Callable

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from research_agent.agent import ResearchAgent
from research_agent.identity import SCOPE, DemoTokenVerifier, TokenStore
from research_agent.tools import ResearchTools
from securerag.models import User

INSTRUCTIONS = (
    "Secure research over company documents. Every result is limited to documents the "
    "authenticated user can already open. Use `research` for questions that need several "
    "steps (comparisons, multi-part questions); it returns verified claims with citations. "
    "Use the other tools for direct lookups."
)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)


class Unauthenticated(PermissionError):
    pass


def build_mcp_server(agent: ResearchAgent, tokens: TokenStore | None = None,
                     identity: Callable[[], User | None] | None = None,
                     public_url: str | None = None) -> MCPServer:
    """`identity` overrides token auth (used by in-process tests)."""
    auth = None
    verifier = None
    if tokens is not None and identity is None:
        public_url = public_url or os.environ.get("RESEARCH_PUBLIC_URL", "http://localhost:8001")
        verifier = DemoTokenVerifier(tokens)
        auth = AuthSettings(issuer_url=os.environ.get("RESEARCH_ISSUER_URL", "https://login.microsoftonline.com/demo"),
                            resource_server_url=f"{public_url.rstrip('/')}/mcp", required_scopes=[SCOPE],
                            validate_token_resource=False)

    server = MCPServer(name="secure-research-agent", title="Secure Research Agent",
                       instructions=INSTRUCTIONS, version="1.0.0", token_verifier=verifier, auth=auth)

    def current_user() -> User:
        if identity is not None:
            user = identity()
        else:
            token = get_access_token()
            user = tokens.directory.get_user(token.subject) if token and token.subject else None
        if user is None:
            raise Unauthenticated("Authentication required.")
        return user

    def toolset() -> ResearchTools:
        return ResearchTools(agent.backend, agent.embedder, current_user())

    @server.tool(annotations=READ_ONLY)
    def research(question: str) -> dict:
        """Answer a question by researching the user's documents in several steps.
        Returns verified claims, each with citations (document, section, page, date, link)."""
        return agent.run(question, current_user()).to_dict()

    @server.tool(annotations=READ_ONLY)
    def search_documents(query: str, top_k: int = 5) -> dict:
        """Hybrid keyword + vector search over the documents the user can read."""
        return toolset().search(query, top_k)

    @server.tool(annotations=READ_ONLY)
    def list_documents() -> dict:
        """List the documents the user can read, with their sections and last-modified dates."""
        return toolset().list_documents()

    @server.tool(annotations=READ_ONLY)
    def read_section(doc_id: str, section: str) -> dict:
        """Read the full text of one section of a document."""
        return toolset().read_section(doc_id, section)

    @server.tool(annotations=READ_ONLY)
    def compare_documents(doc_a: str, doc_b: str) -> dict:
        """Compare two versions of a document and list what changed, section by section."""
        return toolset().compare_documents(doc_a, doc_b)

    return server


def build_http_app(agent: ResearchAgent, tokens: TokenStore, allowed_hosts: list[str] | None = None):
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = allowed_hosts or os.environ.get("RESEARCH_ALLOWED_HOSTS",
                                            "localhost,localhost:*,127.0.0.1,127.0.0.1:*,host.docker.internal:*").split(",")
    server = build_mcp_server(agent, tokens)
    return server.streamable_http_app(
        stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts,
                                                     allowed_origins=["http://localhost:*", "http://127.0.0.1:*"]),
    )
