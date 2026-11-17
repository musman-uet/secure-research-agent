"""Decide whether retrieved evidence is good enough to answer from.

Two checks run before (and independently of) the LLM:

1. Insufficient evidence - nothing retrieved, or the best results do not
   cover the question's key terms / have low semantic relevance.
2. Conflicting evidence - two *different* documents both look relevant but
   state different figures for the same thing (e.g. a 2023 and a 2025
   policy with different allowances). We surface both with dates instead of
   silently picking one.

These checks are deliberately simple and transparent. The LLM generator can
also report a conflict; the pipeline flags it if either says so.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from securerag.models import SearchHit
from securerag.search.local_backend import tokenize

# A number optionally followed by a unit word, or preceded by a currency.
_QUANTITY = re.compile(
    r"(?:(?P<cur>pkr|rs\.?|usd|\$|eur|€)\s*)?(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|percent|days?|"
    r"hours?|weeks?|months?|years?|pkr|usd|nights?|km|kilometres?|kilometers?)?",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class EvidenceReport:
    sufficient: bool
    coverage: float
    conflicts: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _coverage(question_terms: set[str], hits: list[SearchHit]) -> float:
    """Best share of the question's key terms found together in ONE chunk.

    Per-chunk (not union) coverage stops unrelated chunks that each match
    one generic word from adding up to a false "sufficient".
    """
    if not question_terms:
        return 0.0
    return max(len(question_terms & set(tokenize(h.chunk.content))) / len(question_terms) for h in hits)


def _fmt(value: str, unit: str) -> str:
    try:
        num = float(value)
        value = f"{num:,.0f}" if num.is_integer() else f"{num:,}"
    except ValueError:
        pass
    return f"{unit.upper()} {value}" if unit in ("pkr", "usd", "eur") else f"{value} {unit}"


def _quantities(text: str, question_terms: set[str], min_overlap: int) -> set[tuple[str, str]]:
    """(unit, value) pairs from sentences that talk about the question."""
    found = set()
    for sentence in _SENTENCES.split(text):
        if len(question_terms & set(tokenize(sentence))) < min_overlap:
            continue
        for m in _QUANTITY.finditer(sentence):
            unit = (m.group("cur") or m.group("unit") or "").lower().rstrip(".s")
            if not unit:
                continue  # bare numbers (years, IDs) are too noisy
            unit = {"r": "pkr", "rs": "pkr", "$": "usd", "€": "eur", "percent": "%",
                    "kilometre": "km", "kilometer": "km"}.get(unit, unit)
            found.add((unit, m.group("num").replace(",", "")))
    return found


def assess(question: str, hits: list[SearchHit], min_coverage: float = 0.5,
           min_relevance: float = 0.15, top_n: int = 3, semantic: bool = False,
           min_semantic: float = 0.5) -> EvidenceReport:
    """`semantic=True` means hit.relevance comes from a semantic reranker
    (Azure: rerankerScore / 4). That score understands paraphrases, so it
    replaces the keyword-coverage gate. Default 0.5 = rerankerScore 2.0."""
    if not hits:
        return EvidenceReport(False, 0.0, reasons=["No accessible document matched the question."])

    terms = set(tokenize(question))
    top = hits[:top_n]
    coverage = _coverage(terms, top)
    reasons = []
    best_rel = max((h.relevance for h in top if h.relevance is not None), default=None)
    if semantic and best_rel is not None:
        if best_rel < min_semantic:
            reasons.append(f"Best semantic reranker relevance {best_rel:.2f} is below {min_semantic}.")
    else:
        if coverage < min_coverage:
            reasons.append(f"Top results cover only {coverage:.0%} of the question's key terms.")
        if best_rel is not None and best_rel < min_relevance:
            reasons.append(f"Best vector similarity {best_rel:.2f} is below threshold {min_relevance}.")
    if reasons:
        return EvidenceReport(False, coverage, reasons=reasons)

    # Conflict check across distinct documents. Only sentences that share
    # most of the question's terms count, so a hotel rate on the same page
    # is not compared with a meal allowance.
    min_overlap = max(2, math.ceil(0.6 * len(terms)))
    by_doc: dict[str, tuple[SearchHit, set]] = {}
    for h in top:
        q = _quantities(h.chunk.content, terms, min_overlap)
        if q and h.chunk.doc_id not in by_doc:
            by_doc[h.chunk.doc_id] = (h, q)
    conflicts = []
    docs = list(by_doc.values())
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            (ha, qa), (hb, qb) = docs[i], docs[j]
            for unit in {u for u, _ in qa} & {u for u, _ in qb}:
                va = sorted(v for u, v in qa if u == unit)
                vb = sorted(v for u, v in qb if u == unit)
                if va != vb:
                    sa = ", ".join(_fmt(v, unit) for v in va)
                    sb = ", ".join(_fmt(v, unit) for v in vb)
                    conflicts.append(
                        f"'{ha.chunk.title}' (updated {ha.chunk.last_modified}) says {sa}; "
                        f"'{hb.chunk.title}' (updated {hb.chunk.last_modified}) says {sb}."
                    )
    return EvidenceReport(True, coverage, conflicts=conflicts)
