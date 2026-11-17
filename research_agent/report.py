"""Report types and the claim verifier."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from securerag.models import Chunk
from securerag.search.local_backend import tokenize

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_EREF = re.compile(r"\[(E\d+)\]")


@dataclass
class Claim:
    text: str
    evidence: list[str]


@dataclass
class Citation:
    evidence_id: str
    title: str
    section: str
    page: int | None
    last_modified: str
    source_url: str


@dataclass
class ResearchReport:
    question: str
    status: str  # answered | partial | conflicting_evidence | insufficient_evidence
    summary: str
    claims: list[dict]
    notes: list[str]
    steps: list[dict]
    stats: dict
    mode: str
    ai_generated: bool = True
    removed_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def verify_claim(claim: Claim, evidence: dict[str, Chunk], min_overlap: float = 0.6) -> tuple[bool, str]:
    """A claim is kept only if its cited evidence actually supports it:
    every number in the claim appears in the cited text, and most of the
    claim's content words do too. Cheap, deterministic and explainable."""
    cited = [evidence[e] for e in claim.evidence if e in evidence]
    if not cited:
        return False, "no valid evidence cited"
    text = " ".join(c.content for c in cited)
    missing = _numbers(_EREF.sub("", claim.text)) - _numbers(text)
    if missing:
        return False, f"numbers not found in cited evidence: {', '.join(sorted(missing))}"
    terms = set(tokenize(_EREF.sub("", claim.text)))
    if terms:
        overlap = len(terms & set(tokenize(text))) / len(terms)
        if overlap < min_overlap:
            return False, f"only {overlap:.0%} of the claim's words appear in the cited evidence"
    return True, "supported"


def citations_for(ids: list[str], evidence: dict[str, Chunk]) -> list[dict]:
    out = []
    for e in ids:
        c = evidence.get(e)
        if c is not None:
            out.append(asdict(Citation(e, c.title, c.section, c.page, c.last_modified, c.source_url)))
    return out
