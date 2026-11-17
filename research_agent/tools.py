"""Research tools, bound to ONE user's identity.

The identity is fixed when the toolset is created (from the verified token),
and no tool takes a user or group argument. Whatever an LLM, a prompt
injection or an MCP caller passes in, every tool can only see documents the
user can already open. Unknown and forbidden documents get the same answer,
so the existence of restricted content is never revealed.

Every chunk a tool returns is registered as numbered evidence (E1, E2, ...)
so the final report can cite exactly what the agent read.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from research_agent import tracing
from securerag.evidence import assess
from securerag.models import Chunk, User
from securerag.search.local_backend import tokenize

NOT_FOUND = "Not found in the documents you have access to."

_QTY = re.compile(
    r"(?:(?P<cur>pkr|usd|eur|rs\.?)\s*)?(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|days?|hours?|weeks?|months?|minutes?|nights?|km|per\s+km|characters?)?",
    re.IGNORECASE,
)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def quantities(text: str) -> dict[str, list[str]]:
    """unit -> sorted values, e.g. {'pkr': ['4500'], 'km': ['65']}. Bare numbers are ignored."""
    out: dict[str, set[str]] = {}
    for m in _QTY.finditer(text):
        unit = (m.group("cur") or m.group("unit") or "").lower()
        if not unit:
            continue
        unit = re.sub(r"\s+", " ", unit).rstrip(".")
        unit = {"rs": "pkr", "percent": "%", "per km": "km"}.get(unit, unit)
        unit = unit[:-1] if unit.endswith("s") and unit not in ("pkr",) else unit
        out.setdefault(unit, set()).add(m.group("num").replace(",", ""))
    return {u: sorted(v) for u, v in out.items()}


def _norm_heading(h: str) -> str:
    return " ".join(tokenize(h))


def _body(chunk: Chunk) -> str:
    return chunk.content.split("\n", 1)[-1]


@dataclass
class ToolCall:
    tool: str
    args: dict
    summary: str
    ms: float


@dataclass
class ResearchTools:
    backend: object
    embedder: object
    user: User
    max_snippet: int = 700
    evidence: "OrderedDict[str, Chunk]" = field(default_factory=OrderedDict)
    calls: list[ToolCall] = field(default_factory=list)
    _ids: dict[str, str] = field(default_factory=dict)

    # ----------------------------------------------------------- evidence
    def _eid(self, chunk: Chunk) -> str:
        if chunk.chunk_id not in self._ids:
            eid = f"E{len(self._ids) + 1}"
            self._ids[chunk.chunk_id] = eid
            self.evidence[eid] = chunk
        return self._ids[chunk.chunk_id]

    def _ref(self, chunk: Chunk, full: bool = False) -> dict:
        body = _body(chunk)
        return {
            "evidence_id": self._eid(chunk),
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "section": chunk.section,
            "page": chunk.page,
            "last_modified": chunk.last_modified,
            "text": body if full else body[: self.max_snippet],
        }

    def _log(self, tool: str, args: dict, summary: str, started: float) -> None:
        self.calls.append(ToolCall(tool, args, summary, round((time.perf_counter() - started) * 1000, 1)))

    def _browse(self, doc_id: str | None = None) -> list[Chunk]:
        chunks = self.backend.browse(self.user.principals, doc_id)
        return sorted(chunks, key=lambda c: (c.doc_id, c.page or 0)) if doc_id is None else chunks

    # -------------------------------------------------------------- tools
    def list_documents(self) -> dict:
        """Documents this user can read, with their sections and dates."""
        started = time.perf_counter()
        with tracing.tool_span("list_documents") as span:
            docs: "OrderedDict[str, dict]" = OrderedDict()
            for c in self._browse():
                d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title,
                                               "last_modified": c.last_modified, "file_type": c.file_type,
                                               "sections": []})
                if c.section not in d["sections"]:
                    d["sections"].append(c.section)
            span.set_attribute("research.documents", len(docs))
        result = {"documents": list(docs.values())}
        self._log("list_documents", {}, f"{len(docs)} documents", started)
        return result

    def search(self, query: str, top_k: int = 5) -> dict:
        """Hybrid search over the user's documents, with an evidence-quality check."""
        started = time.perf_counter()
        top_k = max(1, min(int(top_k), 10))
        with tracing.tool_span("search") as span:
            vector = self.embedder.embed([query])[0]
            hits = self.backend.search(query, vector, self.user.principals, top_k, "hybrid")
            semantic = bool(getattr(self.backend, "provides_semantic_relevance", False))
            report = assess(query, hits, semantic=semantic)
            span.set_attribute("research.hits", len(hits))
            span.set_attribute("research.evidence_sufficient", report.sufficient)
        result = {
            "results": [self._ref(h.chunk) for h in hits],
            "evidence_sufficient": report.sufficient,
            "coverage": round(report.coverage, 2),
            "conflicts": report.conflicts,
        }
        self._log("search", {"query": query, "top_k": top_k},
                  f"{len(hits)} hits, sufficient={report.sufficient}", started)
        return result

    def read_section(self, doc_id: str, section: str) -> dict:
        """Full text of one section of a document, if the user can read it."""
        started = time.perf_counter()
        with tracing.tool_span("read_section") as span:
            try:
                chunks = self._browse(doc_id)
            except ValueError:
                chunks = []
            want = _norm_heading(section)
            match = [c for c in chunks if _norm_heading(c.section) == want] or \
                    [c for c in chunks if want and want in _norm_heading(c.section)]
            span.set_attribute("research.found", bool(match))
        if not match:
            self._log("read_section", {"doc_id": doc_id, "section": section}, "not found", started)
            return {"found": False, "message": NOT_FOUND}
        self._log("read_section", {"doc_id": doc_id, "section": section}, f"{len(match)} chunks", started)
        return {"found": True, "chunks": [self._ref(c, full=True) for c in match]}

    def compare_documents(self, doc_a: str, doc_b: str) -> dict:
        """Section-by-section differences between two documents (older first)."""
        started = time.perf_counter()
        with tracing.tool_span("compare_documents") as span:
            try:
                a, b = self._browse(doc_a), self._browse(doc_b)
            except ValueError:
                a, b = [], []
            if not a or not b:
                span.set_attribute("research.found", False)
                self._log("compare_documents", {"doc_a": doc_a, "doc_b": doc_b}, "not found", started)
                return {"found": False, "message": NOT_FOUND}
            if a[0].last_modified > b[0].last_modified:
                a, b = b, a

            def sections(chunks: list[Chunk]) -> "OrderedDict[str, list[Chunk]]":
                out: "OrderedDict[str, list[Chunk]]" = OrderedDict()
                for c in chunks:
                    out.setdefault(_norm_heading(c.section), []).append(c)
                return out

            sa, sb = sections(a), sections(b)
            changes, unchanged = [], []
            for key in list(sa.keys()) + [k for k in sb if k not in sa]:
                ca, cb = sa.get(key), sb.get(key)
                name = (ca or cb)[0].section
                ta = " ".join(_body(c) for c in ca) if ca else ""
                tb = " ".join(_body(c) for c in cb) if cb else ""
                ev = [self._eid(c) for c in (ca or []) + (cb or [])]
                if ca and not cb:
                    changes.append({"section": name, "type": "removed", "before": ta, "after": None, "evidence": ev})
                elif cb and not ca:
                    changes.append({"section": name, "type": "added", "before": None, "after": tb, "evidence": ev})
                else:
                    qa, qb = quantities(ta), quantities(tb)
                    if qa != qb:
                        changes.append({"section": name, "type": "changed", "before": ta, "after": tb,
                                        "before_values": qa, "after_values": qb, "evidence": ev})
                    elif set(tokenize(ta)) != set(tokenize(tb)):
                        changes.append({"section": name, "type": "reworded", "before": ta, "after": tb,
                                        "evidence": ev})
                    else:
                        unchanged.append(name)
            span.set_attribute("research.changes", len(changes))
        meta = lambda c: {"doc_id": c.doc_id, "title": c.title, "last_modified": c.last_modified}  # noqa: E731
        result = {"found": True, "older": meta(a[0]), "newer": meta(b[0]), "changes": changes,
                  "unchanged_sections": unchanged}
        self._log("compare_documents", {"doc_a": doc_a, "doc_b": doc_b}, f"{len(changes)} changes", started)
        return result


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "List the documents the current user can read, with section names and last-modified dates.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "search",
        "description": "Hybrid keyword + vector search over the user's documents. Returns numbered evidence.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A focused search query."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "read_section",
        "description": "Read the full text of one section of a document.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}, "section": {"type": "string"}},
            "required": ["doc_id", "section"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "compare_documents",
        "description": "Compare two versions of a document section by section and list what changed.",
        "parameters": {"type": "object", "properties": {
            "doc_a": {"type": "string"}, "doc_b": {"type": "string"}},
            "required": ["doc_a", "doc_b"], "additionalProperties": False}}},
]
