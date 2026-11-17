"""The research agent: plan -> use tools -> write claims -> verify -> report -> audit."""

from __future__ import annotations

import os
import time
from pathlib import Path

from research_agent import tracing
from research_agent.offline import run_offline
from research_agent.report import ResearchReport, citations_for, verify_claim
from research_agent.tools import ResearchTools
from securerag.audit import AuditLog
from securerag.embeddings import HashingEmbedder
from securerag.ingest.loader import load_chunks
from securerag.models import User
from securerag.search.local_backend import LocalSearchBackend

ROOT = Path(__file__).resolve().parent.parent


class ResearchAgent:
    def __init__(self, backend, embedder, llm=None, audit: AuditLog | None = None,
                 max_tool_calls: int = 12):
        self.backend, self.embedder, self.llm = backend, embedder, llm
        self.audit = audit or AuditLog(None)
        self.max_tool_calls = max_tool_calls

    @property
    def mode(self) -> str:
        return "llm" if self.llm is not None else "offline"

    def run(self, question: str, user: User) -> ResearchReport:
        started = time.perf_counter()
        tools = ResearchTools(self.backend, self.embedder, user)
        with tracing.agent_span(question, user.user_id, self.mode) as span:
            conflict = False
            plan_kind = "llm"
            try:
                if self.llm is not None:
                    from research_agent.llm import run_llm
                    claims, notes, conflict = run_llm(tools, question, self.llm, self.max_tool_calls)
                else:
                    claims, notes, plan_kind = run_offline(tools, question, self.max_tool_calls)
            except Exception as e:  # never leak a stack trace to the caller
                claims, notes = [], [f"Research failed: {type(e).__name__}."]

            kept, removed = [], []
            for c in claims:
                ok, reason = verify_claim(c, tools.evidence)
                if ok:
                    kept.append(c)
                else:
                    removed.append(f"{c.text} ({reason})")
            conflict = conflict or any(n.startswith("'") and " says " in n for n in notes)
            unanswered = [n for n in notes if n.startswith(("No supporting", "Not answered"))]

            if not kept:
                status = "insufficient_evidence"
            elif conflict and plan_kind != "compare":
                status = "conflicting_evidence"
            elif unanswered or removed:
                status = "partial"
            else:
                status = "answered"

            claim_dicts = [{"text": c.text, "citations": citations_for(c.evidence, tools.evidence)} for c in kept]
            summary = "\n".join(f"- {c.text} " + "".join(f"[{e}]" for e in c.evidence) for c in kept) or \
                "I could not find an answer in the documents you have access to."
            if removed:
                notes.append(f"{len(removed)} claim(s) removed because the evidence did not support them.")
            if conflict and plan_kind != "compare":
                notes.append("Sources disagree. Confirm which document is current before acting.")

            latency = round((time.perf_counter() - started) * 1000, 1)
            stats = {"tool_calls": len(tools.calls), "evidence_read": len(tools.evidence),
                     "evidence_doc_ids": sorted({c.doc_id for c in tools.evidence.values()}),
                     "claims": len(kept), "claims_removed": len(removed), "latency_ms": latency,
                     "plan": plan_kind}
            span.set_attribute("research.status", status)
            span.set_attribute("research.tool_calls", len(tools.calls))
            span.set_attribute("research.claims", len(kept))
            span.set_attribute("research.claims_removed", len(removed))

        report = ResearchReport(
            question=question, status=status, summary=summary, claims=claim_dicts, notes=notes,
            steps=[{"tool": c.tool, "args": c.args, "result": c.summary, "ms": c.ms} for c in tools.calls],
            stats=stats, mode=self.mode, removed_claims=removed,
        )
        self.audit.record(event="research", user_id=user.user_id, question=question, status=status,
                          mode=self.mode, tool_calls=[c.tool for c in tools.calls],
                          evidence_doc_ids=sorted({c.doc_id for c in tools.evidence.values()}),
                          cited_doc_ids=sorted({cit["title"] for cd in claim_dicts for cit in cd["citations"]}),
                          latency_ms=latency)
        return report


def build_local_agent(manifest: Path | None = None, audit_path: Path | None = None, llm=None) -> ResearchAgent:
    """Offline backend over the sample library (or another manifest)."""
    embedder = HashingEmbedder()
    chunks = load_chunks(manifest or ROOT / "data" / "manifest.json")
    for c, v in zip(chunks, embedder.embed([c.content for c in chunks])):
        c.vector = v
    backend = LocalSearchBackend()
    backend.index(chunks)
    return ResearchAgent(backend, embedder, llm=llm, audit=AuditLog(audit_path))


def build_agent_from_env() -> ResearchAgent:
    audit = Path(os.environ.get("RESEARCH_AUDIT_LOG", str(ROOT / "logs" / "research_audit.jsonl")))
    llm = None
    if os.environ.get("RESEARCH_LLM", "off").lower() == "on":
        from research_agent.llm import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM.from_env()
    if os.environ.get("RAG_BACKEND", "local").lower() == "azure":
        from securerag.config import Settings
        from securerag.factory import build_backend, build_embedder
        s = Settings.from_env()
        s.require_azure()
        return ResearchAgent(build_backend(s), build_embedder(s), llm=llm, audit=AuditLog(audit))
    return build_local_agent(audit_path=audit, llm=llm)
