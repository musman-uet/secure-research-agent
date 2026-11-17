"""Settings from environment variables (or a .env file). No secrets in code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    backend: str  # "local" or "azure"
    manifest: Path
    directory: Path
    audit_log: Path
    search_endpoint: str | None
    search_key: str | None
    search_index: str
    use_semantic: bool
    openai_endpoint: str | None
    openai_key: str | None
    openai_api_version: str
    chat_deployment: str | None
    embedding_deployment: str | None
    embedding_dimensions: int

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        env = os.environ.get
        return cls(
            backend=env("RAG_BACKEND", "local").lower(),
            manifest=Path(env("RAG_MANIFEST", str(ROOT / "data" / "manifest.json"))),
            directory=Path(env("RAG_DIRECTORY", str(ROOT / "data" / "directory.json"))),
            audit_log=Path(env("RAG_AUDIT_LOG", str(ROOT / "logs" / "audit.jsonl"))),
            search_endpoint=env("AZURE_SEARCH_ENDPOINT"),
            search_key=env("AZURE_SEARCH_API_KEY"),
            search_index=env("AZURE_SEARCH_INDEX", "secure-rag-chunks"),
            use_semantic=env("AZURE_SEARCH_SEMANTIC", "true").lower() == "true",
            openai_endpoint=env("AZURE_OPENAI_ENDPOINT"),
            openai_key=env("AZURE_OPENAI_API_KEY"),
            openai_api_version=env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            chat_deployment=env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            embedding_deployment=env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            embedding_dimensions=int(env("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")),
        )

    def require_azure(self) -> None:
        missing = [n for n, v in {
            "AZURE_SEARCH_ENDPOINT": self.search_endpoint, "AZURE_SEARCH_API_KEY": self.search_key,
            "AZURE_OPENAI_ENDPOINT": self.openai_endpoint, "AZURE_OPENAI_API_KEY": self.openai_key,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": self.chat_deployment,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.embedding_deployment,
        }.items() if not v]
        if missing:
            raise SystemExit(f"Missing environment variables for Azure mode: {', '.join(missing)}")
