"""Core data types shared by ingestion, search, generation and the API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


@dataclass
class Section:
    """A logical part of a source document (a heading block, a PDF page, a sheet)."""

    heading: str
    text: str
    page: int | None = None


@dataclass
class Document:
    """A parsed business document plus the metadata we index alongside it."""

    doc_id: str
    title: str
    source_url: str
    last_modified: str  # ISO date, e.g. "2025-03-01"
    file_type: str  # docx | pdf | xlsx | md | txt
    sections: list[Section]
    allowed_principals: list[str]  # users and groups that may read this document


@dataclass
class Chunk:
    """One retrievable unit. This is exactly what goes into the search index."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    content: str
    source_url: str
    last_modified: str
    file_type: str
    allowed_principals: list[str]
    page: int | None = None
    vector: list[float] | None = None

    def to_index_doc(self) -> dict:
        d = asdict(self)
        d["id"] = d.pop("chunk_id")
        if d["page"] is None:
            d.pop("page")
        return d


@dataclass
class User:
    """The person asking. In production this comes from the Entra ID token."""

    user_id: str
    groups: list[str] = field(default_factory=list)

    @property
    def principals(self) -> list[str]:
        """Every identity a document ACL can grant access to."""
        return [self.user_id, *self.groups]


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # fused ranking score (only meaningful within one result list)
    relevance: float | None = None  # 0..1 confidence (semantic reranker / cosine), if known


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


@dataclass
class Citation:
    ref: int
    title: str
    section: str
    last_modified: str
    source_url: str
    doc_id: str
    page: int | None = None


@dataclass
class Answer:
    question: str
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    notes: list[str] = field(default_factory=list)
    ai_generated: bool = True  # every response is labelled as AI output

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d
