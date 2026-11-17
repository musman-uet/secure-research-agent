"""Azure AI Search backend: hybrid (BM25 + vector, RRF) + semantic ranker,
with document-level security trimming via an OData filter.

Security model (Azure "security filter" pattern):
  * every chunk stores `allowed_principals` (user IDs and group IDs);
  * the field is filterable but NOT retrievable, so ACLs never leave the index;
  * each query adds  allowed_principals/any(p: search.in(p, '<ids>', '|'))
    built from the caller's identity, and vectorFilterMode=preFilter so the
    vector leg is trimmed before nearest-neighbour search, not after.
"""

from __future__ import annotations

import re

from securerag.models import Chunk, SearchHit
from securerag.search.base import SearchMode

# Entra object IDs (GUIDs), UPNs and simple group names. Anything else is
# rejected so a principal can never inject OData or break the delimiter.
_SAFE_PRINCIPAL = re.compile(r"^[A-Za-z0-9@._:\-]{1,256}$")

VECTOR_FIELD = "content_vector"
SEMANTIC_CONFIG = "default-semantic"
VECTOR_PROFILE = "default-vector-profile"
HNSW_CONFIG = "default-hnsw"
SELECT_FIELDS = ["id", "doc_id", "title", "section", "content", "source_url",
                 "last_modified", "file_type", "page"]


class UnsafePrincipalError(ValueError):
    pass


def build_security_filter(principals: list[str]) -> str:
    """OData filter that keeps only documents shared with one of `principals`."""
    if not principals:
        raise UnsafePrincipalError("No principals: refusing to search without an identity")
    for p in principals:
        if not _SAFE_PRINCIPAL.match(p):
            raise UnsafePrincipalError(f"Unsafe principal identifier: {p!r}")
    joined = "|".join(principals)
    return f"allowed_principals/any(p: search.in(p, '{joined}', '|'))"


_SAFE_DOC_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")


def build_browse_filter(principals: list[str], doc_id: str | None = None) -> str:
    """Security filter, optionally narrowed to one document."""
    f = build_security_filter(principals)
    if doc_id is not None:
        if not _SAFE_DOC_ID.match(doc_id):
            raise UnsafePrincipalError(f"Unsafe document id: {doc_id!r}")
        f = f"({f}) and doc_id eq '{doc_id}'"
    return f


def build_index(index_name: str, dimensions: int):
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration, SearchableField, SearchField, SearchFieldDataType,
        SearchIndex, SemanticConfiguration, SemanticField, SemanticPrioritizedFields,
        SemanticSearch, SimpleField, VectorSearch, VectorSearchProfile,
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="section", type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="source_url", type=SearchFieldDataType.String),
        SimpleField(name="last_modified", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="file_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32),
        SimpleField(name="allowed_principals", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True, hidden=True),
        SearchField(name=VECTOR_FIELD, type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=dimensions,
                    vector_search_profile_name=VECTOR_PROFILE),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=HNSW_CONFIG)],
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=HNSW_CONFIG)],
    )
    semantic = SemanticSearch(configurations=[SemanticConfiguration(
        name=SEMANTIC_CONFIG,
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="section")],
        ),
    )])
    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic)


def build_search_kwargs(query: str, query_vector: list[float] | None, principals: list[str],
                        top_k: int, mode: SearchMode, use_semantic: bool) -> dict:
    """Pure function (unit-tested) producing the arguments for SearchClient.search."""
    from azure.search.documents.models import VectorizedQuery

    kwargs: dict = {
        "filter": build_security_filter(principals),
        "top": top_k,
        "select": SELECT_FIELDS,
    }
    kwargs["search_text"] = query if mode in ("keyword", "hybrid") else None
    if mode in ("vector", "hybrid"):
        if query_vector is None:
            raise ValueError("vector/hybrid search needs a query vector")
        kwargs["vector_queries"] = [VectorizedQuery(vector=query_vector, k_nearest_neighbors=50,
                                                    fields=VECTOR_FIELD)]
        kwargs["vector_filter_mode"] = "preFilter"
    if use_semantic and mode != "vector":
        kwargs["query_type"] = "semantic"
        kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG
    return kwargs


class AzureSearchBackend:
    def __init__(self, endpoint: str, api_key: str, index_name: str, dimensions: int,
                 use_semantic: bool = True, search_client=None, index_client=None):
        if search_client is None or index_client is None:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient

            cred = AzureKeyCredential(api_key)
            index_client = index_client or SearchIndexClient(endpoint=endpoint, credential=cred)
            search_client = search_client or SearchClient(endpoint=endpoint, index_name=index_name,
                                                          credential=cred)
        self._index_client, self._client = index_client, search_client
        self.index_name, self.dimensions, self.use_semantic = index_name, dimensions, use_semantic

    @property
    def provides_semantic_relevance(self) -> bool:
        return self.use_semantic

    def create_or_update_index(self) -> None:
        self._index_client.create_or_update_index(build_index(self.index_name, self.dimensions))

    def index(self, chunks: list[Chunk]) -> None:
        docs = []
        for c in chunks:
            d = c.to_index_doc()
            d[VECTOR_FIELD] = d.pop("vector")
            if d[VECTOR_FIELD] is None:
                raise ValueError(f"chunk {c.chunk_id} has no vector")
            docs.append(d)
        for i in range(0, len(docs), 500):
            results = self._client.merge_or_upload_documents(documents=docs[i:i + 500])
            failed = [r.key for r in results if not r.succeeded]
            if failed:
                raise RuntimeError(f"Failed to index {len(failed)} chunks, e.g. {failed[:3]}")

    def browse(self, principals: list[str], doc_id: str | None = None, limit: int = 1000) -> list[Chunk]:
        """Read chunks the principals may access, without ranking (search_text='*')."""
        results = self._client.search(search_text="*", filter=build_browse_filter(principals, doc_id),
                                      top=limit, select=SELECT_FIELDS)
        return [Chunk(chunk_id=r["id"], doc_id=r["doc_id"], title=r["title"], section=r["section"],
                      content=r["content"], source_url=r["source_url"], last_modified=r["last_modified"],
                      file_type=r["file_type"], page=r.get("page"), allowed_principals=[])
                for r in results]

    def search(self, query: str, query_vector: list[float] | None, principals: list[str],
               top_k: int = 5, mode: SearchMode = "hybrid") -> list[SearchHit]:
        kwargs = build_search_kwargs(query, query_vector, principals, top_k, mode, self.use_semantic)
        hits = []
        for r in self._client.search(**kwargs):
            reranker = r.get("@search.reranker_score")
            chunk = Chunk(
                chunk_id=r["id"], doc_id=r["doc_id"], title=r["title"], section=r["section"],
                content=r["content"], source_url=r["source_url"], last_modified=r["last_modified"],
                file_type=r["file_type"], page=r.get("page"),
                allowed_principals=[],  # never returned by the service (hidden field)
            )
            hits.append(SearchHit(chunk=chunk, score=float(r["@search.score"]),
                                  relevance=(reranker / 4.0) if reranker is not None else None))
        return hits
