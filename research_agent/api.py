"""REST API for simple integrations (n8n HTTP Request node, webhooks, scripts).

    POST /research   {"question": "..."}   Authorization: Bearer <token>
    GET  /health
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from research_agent.agent import ROOT, ResearchAgent, build_agent_from_env
from research_agent.identity import TokenStore
from securerag.directory import JsonDirectory
from securerag.models import User

app = FastAPI(title="Secure Research Agent", version="1.0.0",
              description="Multi-step, permission-trimmed research over company documents with verified citations.")


@lru_cache(maxsize=1)
def get_agent() -> ResearchAgent:
    return build_agent_from_env()


@lru_cache(maxsize=1)
def get_tokens() -> TokenStore:
    directory = JsonDirectory(Path(os.environ.get("RAG_DIRECTORY", ROOT / "data" / "directory.json")))
    return TokenStore(Path(os.environ.get("RESEARCH_TOKENS", ROOT / "data" / "tokens.json")), directory)


def current_user(authorization: str | None = Header(default=None),
                 tokens: TokenStore = Depends(get_tokens)) -> User:
    token = authorization.split(" ", 1)[1].strip() if authorization and authorization.lower().startswith("bearer ") else None
    user = tokens.user_for(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": get_agent().mode}


@app.post("/research")
def research(req: ResearchRequest, user: User = Depends(current_user),
             agent: ResearchAgent = Depends(get_agent)) -> dict:
    return agent.run(req.question, user).to_dict()
