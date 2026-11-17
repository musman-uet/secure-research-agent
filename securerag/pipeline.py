"""The question-answering pipeline: retrieve (security-trimmed) -> check
evidence -> generate with citations -> audit."""

from __future__ import annotations

import re
import time

from securerag.audit import AuditLog
from securerag.embeddings import Embedder
from securerag.evidence import assess
from securerag.generation import Generator
from securerag.models import Answer, AnswerStatus, Citation, SearchHit, User
from securerag.search.base import SearchBackend, SearchMode

# Identical wording whether a document does not exist or exists but the user
# cannot open it - the answer must never reveal that restricted content exists.
NOT_FOUND_MESSAGE = ("I could not find an answer to this in the documents you have access to. "
                     "Please check with the document owner or rephrase the question.")


def _citation(n: int, hit: SearchHit) -> Citation:
    c = hit.chunk
    return Citation(ref=n, title=c.title, section=c.section, last_modified=c.last_modified,
                    source_url=c.source_url, doc_id=c.doc_id, page=c.page)


def _renumber(text: str, citations: list[Citation]) -> tuple[str, list[Citation]]:
    """Renumber cited sources 1..n in the order they are listed."""
    mapping = {c.ref: i for i, c in enumerate(citations, start=1)}
    text = re.sub(r"\[(\d+)\]", lambda m: f"[{mapping[int(m.group(1))]}]"
                  if int(m.group(1)) in mapping else "", text)
    for c in citations:
        c.ref = mapping[c.ref]
    return text, citations


class RagPipeline:
    def __init__(self, backend: SearchBackend, embedder: Embedder, generator: Generator,
                 audit: AuditLog | None = None, top_k: int = 5, mode: SearchMode = "hybrid"):
        self.backend, self.embedder, self.generator = backend, embedder, generator
        self.audit = audit or AuditLog(None)
        self.top_k, self.mode = top_k, mode

    def retrieve(self, question: str, user: User, top_k: int | None = None,
                 mode: SearchMode | None = None) -> list[SearchHit]:
        mode = mode or self.mode
        vector = self.embedder.embed([question])[0] if mode != "keyword" else None
        return self.backend.search(question, vector, user.principals, top_k or self.top_k, mode)

    def ask(self, question: str, user: User) -> Answer:
        started = time.perf_counter()
        hits = self.retrieve(question, user)
        semantic = bool(getattr(self.backend, "provides_semantic_relevance", False))
        report = assess(question, hits, semantic=semantic)

        if not report.sufficient:
            answer = Answer(question, AnswerStatus.INSUFFICIENT_EVIDENCE, NOT_FOUND_MESSAGE, [], [])
        else:
            gen = self.generator.generate(question, hits)
            if gen.insufficient or not gen.refs:
                # An answer with no valid citation is never shown.
                answer = Answer(question, AnswerStatus.INSUFFICIENT_EVIDENCE, NOT_FOUND_MESSAGE, [])
            else:
                citations = [_citation(n, hits[n - 1]) for n in gen.refs]
                notes = list(report.conflicts)
                conflict = gen.conflict or bool(report.conflicts)
                if conflict:
                    # Make sure every document involved in the conflict is cited.
                    cited = {c.doc_id for c in citations}
                    for n, h in enumerate(hits, start=1):
                        if h.chunk.doc_id not in cited and any(h.chunk.title in x for x in notes):
                            citations.append(_citation(n, h))
                            cited.add(h.chunk.doc_id)
                    notes.append("Sources disagree. Confirm which document is current before acting.")
                status = AnswerStatus.CONFLICTING_EVIDENCE if conflict else AnswerStatus.ANSWERED
                text, citations = _renumber(gen.text, citations)
                answer = Answer(question, status, text, citations, notes)

        self.audit.record(
            event="ask",
            user_id=user.user_id,
            question=question,
            status=answer.status.value,
            retrieved_doc_ids=[h.chunk.doc_id for h in hits],
            cited_doc_ids=[c.doc_id for c in answer.citations],
            evidence_coverage=round(report.coverage, 3),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return answer
