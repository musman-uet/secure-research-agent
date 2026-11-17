"""HTTP API.

Identity: in this demo the caller is identified by the `X-User-Id` header
and resolved against data/directory.json. In production, replace
`current_user` with validation of the Entra ID access token (issuer,
audience, signature) and read the user's object ID and `groups` claim from
it. Nothing else in the pipeline changes.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from securerag.config import Settings
from securerag.directory import JsonDirectory, UnknownUserError
from securerag.factory import build_pipeline
from securerag.models import User
from securerag.pipeline import RagPipeline

app = FastAPI(title="Secure RAG Search", version="1.0.0",
              description="Permission-trimmed, cited answers over business documents.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    return build_pipeline(get_settings())


@lru_cache(maxsize=1)
def get_directory() -> JsonDirectory:
    return JsonDirectory(get_settings().directory)


def current_user(x_user_id: str | None = Header(default=None),
                 directory: JsonDirectory = Depends(get_directory)) -> User:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing identity")
    try:
        return directory.get_user(x_user_id)
    except UnknownUserError:
        raise HTTPException(status_code=401, detail="Unknown user")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest, user: User = Depends(current_user),
        pipeline: RagPipeline = Depends(get_pipeline)) -> dict:
    return pipeline.ask(req.question, user).to_dict()
