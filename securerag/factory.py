"""Wire up the pipeline for local or Azure mode."""

from __future__ import annotations

from securerag.audit import AuditLog
from securerag.config import Settings
from securerag.embeddings import AzureOpenAIEmbedder, HashingEmbedder
from securerag.generation import AzureOpenAIGenerator, ExtractiveGenerator
from securerag.ingest.loader import load_chunks
from securerag.pipeline import RagPipeline
from securerag.search.local_backend import LocalSearchBackend


def build_embedder(s: Settings):
    if s.backend == "azure":
        return AzureOpenAIEmbedder(s.openai_endpoint, s.openai_key, s.embedding_deployment,
                                   s.openai_api_version, s.embedding_dimensions)
    return HashingEmbedder()


def build_backend(s: Settings):
    if s.backend == "azure":
        from securerag.search.azure_backend import AzureSearchBackend
        return AzureSearchBackend(s.search_endpoint, s.search_key, s.search_index,
                                  s.embedding_dimensions, s.use_semantic)
    return LocalSearchBackend()


def ingest(s: Settings, backend, embedder) -> int:
    chunks = load_chunks(s.manifest)
    vectors = embedder.embed([c.content for c in chunks])
    for c, v in zip(chunks, vectors):
        c.vector = v
    if s.backend == "azure":
        backend.create_or_update_index()
    backend.index(chunks)
    return len(chunks)


def build_pipeline(s: Settings, ingest_local: bool = True) -> RagPipeline:
    if s.backend == "azure":
        s.require_azure()
    embedder = build_embedder(s)
    backend = build_backend(s)
    if s.backend == "local" and ingest_local:
        ingest(s, backend, embedder)  # in-memory index is rebuilt on start
    generator = (AzureOpenAIGenerator(s.openai_endpoint, s.openai_key, s.chat_deployment,
                                      s.openai_api_version)
                 if s.backend == "azure" else ExtractiveGenerator())
    return RagPipeline(backend, embedder, generator, AuditLog(s.audit_log))
