"""Deterministic research strategy (no LLM).

Used for local runs, CI and as a fallback when the LLM is unavailable. It
follows the same loop an LLM agent would, with explicit rules:

1. Look at the catalog of documents the user can read.
2. Plan: a version comparison ("what changed between the 2023 and 2025
   travel policy?"), or one search per sub-question for multi-part questions.
3. Execute: search; if the evidence is weak, fall back to reading the most
   relevant section directly (search -> browse -> read).
4. Write claims that quote the evidence and cite it as [E#].
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from research_agent.report import Claim
from research_agent.tools import ResearchTools
from securerag.search.local_backend import tokenize

_COMPARE = re.compile(r"\b(what\s+changed|changes?|differen\w*|compare|comparison|vs\.?|versus|updated?)\b", re.I)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_SPLIT = re.compile(r"\?\s*|,?\s+and\s+(?=(?:what|who|whom|how|when|where|which|is|are|do|does|can|should|must)\b)",
                    re.I)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Plan:
    kind: str  # "compare" | "search"
    sub_questions: list[str]
    compare: tuple[str, str] | None = None


def _title_key(title: str) -> str:
    return " ".join(t for t in tokenize(title) if not _YEAR.fullmatch(t))


def plan(question: str, catalog: list[dict]) -> Plan:
    q_terms = set(tokenize(question)) - {m.group(0) for m in _YEAR.finditer(question)}
    years = {m.group(0) for m in _YEAR.finditer(question)}
    if _COMPARE.search(question) or len(years) >= 2:
        groups: dict[str, list[dict]] = {}
        for d in catalog:
            groups.setdefault(_title_key(d["title"]), []).append(d)
        best, best_score = None, 0
        for key, docs in groups.items():
            if len(docs) < 2:
                continue
            score = len(q_terms & set(key.split()))
            if score > best_score:
                best, best_score = docs, score
        if best:
            if years:
                picked = [d for d in best if any(y in d["title"] for y in years)]
                best = picked if len(picked) >= 2 else best
            best = sorted(best, key=lambda d: d["last_modified"])[-2:]
            return Plan("compare", [question], (best[0]["doc_id"], best[1]["doc_id"]))

    parts = [p.strip(" ,.") for p in _SPLIT.split(question) if p and p.strip(" ,.")]
    parts = [p for p in parts if len(tokenize(p)) >= 2] or [question]
    return Plan("search", parts)


_QUESTION_WORDS = {"many", "much", "get", "need", "needed"}
_IRREGULAR = {"spent": "spend", "paid": "pay", "made": "make", "bought": "buy", "sent": "send",
              "written": "write", "wrote": "write", "held": "hold", "left": "leave", "given": "give"}


def _stem(t: str) -> str:
    """Very light suffix stripping: receipts->receipt, hiring/hired->hir, actually->actual."""
    if t in _IRREGULAR:
        return _IRREGULAR[t]
    for suffix in ("ing", "ed", "ly", "s"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 3 and not t.endswith("ss"):
            return t[: -len(suffix)]
    return t


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if t not in _QUESTION_WORDS}


def _best_sentences(sub_q: str, refs: list[dict], limit: int = 2, full_q: str = "") -> list[Claim]:
    q = _stems(sub_q)
    # The full question breaks ties: in "...cloud infrastructure and how much was
    # actually spent?" the second part is about the cloud row, not any row.
    full = _stems(full_q) - q
    scored = []
    for rank, r in enumerate(refs):
        heading = _stems(r.get("section", ""))
        for idx, s in enumerate(_SENT.split(r["text"])):
            s = s.strip()
            if len(s.split()) < 4:
                continue
            # The section heading is context for every sentence under it
            # ("Mileage" -> "reimbursed PKR 65 per km").
            terms = _stems(s) | heading
            overlap = len(q & terms)
            # One matching word is only enough when it is the section's topic
            # ("Mileage"); a stray word in the body ("engineer") is not.
            if overlap >= 2 or (overlap and (q & heading or len(q) == 1)):
                scored.append((overlap, len(full & terms), -rank, -idx, s, r["evidence_id"], r["doc_id"]))
    if not scored:
        return []
    scored.sort(reverse=True)
    best = scored[0][0]
    strength = best / max(1, len(q))
    if strength < 0.5:  # at least half of the sub-question's key terms
        return []
    # Weak matches give one sentence (best search rank). Only strong matches
    # may quote several documents, e.g. both sides of a conflict.
    limit = limit if strength >= 0.75 else 1
    claims, docs = [], set()
    for overlap, _, _, _, s, eid, doc in scored:
        if overlap < best or len(claims) >= limit:
            break
        if doc in docs and claims:
            continue
        docs.add(doc)
        claims.append(Claim(s.rstrip(".") + ".", [eid]))
    return claims


def _section_fallback(tools: ResearchTools, sub_q: str, catalog: list[dict]) -> list[dict]:
    """Search was weak: pick the section whose heading best matches, and read it."""
    q = set(tokenize(sub_q))
    best, best_score = None, 0
    for d in catalog:
        title_terms = set(tokenize(d["title"]))
        for s in d["sections"]:
            score = 2 * len(q & set(tokenize(s))) + len(q & title_terms)
            if score > best_score:
                best, best_score = (d["doc_id"], s), score
    if not best or best_score < 2:
        return []
    res = tools.read_section(*best)
    return res.get("chunks", []) if res.get("found") else []


def _fmt_values(values: dict) -> str:
    parts = []
    for unit, vals in values.items():
        nums = ", ".join(f"{int(float(v)):,}" if float(v).is_integer() else v for v in vals)
        parts.append(f"{unit.upper()} {nums}" if unit in ("pkr", "usd", "eur") else f"{nums} {unit}")
    return "; ".join(parts) or "no figures"


def _first_sentence(text: str) -> str:
    return _SENT.split(text.strip())[0].rstrip(".") + "."


def _novel_sentence(before: str, after: str) -> str:
    """The sentence in the new version that differs most from the old version."""
    old = set(tokenize(before))
    sents = [s.strip() for s in _SENT.split(after.strip()) if len(s.split()) >= 4]
    if not sents:
        return _first_sentence(after)
    best = min(sents, key=lambda s: len(set(tokenize(s)) & old) / max(1, len(set(tokenize(s)))))
    return best.rstrip(".") + "."


def run_offline(tools: ResearchTools, question: str, max_tool_calls: int = 12) -> tuple[list[Claim], list[str], str]:
    """Returns (claims, notes, plan_kind)."""
    catalog = tools.list_documents()["documents"]
    p = plan(question, catalog)
    notes: list[str] = []

    if p.kind == "compare":
        result = tools.compare_documents(*p.compare)
        if not result.get("found"):
            return [], ["The documents to compare are not available to you."], p.kind
        old, new = result["older"]["title"], result["newer"]["title"]
        claims = []
        for ch in result["changes"]:
            ev = ch["evidence"]
            if ch["type"] == "changed":
                claims.append(Claim(f"{ch['section']}: {_fmt_values(ch['before_values'])} in {old}, "
                                    f"{_fmt_values(ch['after_values'])} in {new}.", ev))
            elif ch["type"] == "added":
                claims.append(Claim(f"New in {new}, {ch['section']}: {_first_sentence(ch['after'])}", ev))
            elif ch["type"] == "removed":
                claims.append(Claim(f"Removed in {new}, {ch['section']}: {_first_sentence(ch['before'])}", ev))
            else:
                claims.append(Claim(f"{ch['section']} changed. {new} now says: "
                                    f"{_novel_sentence(ch['before'], ch['after'])}", ev))
        if result["unchanged_sections"]:
            notes.append("Unchanged sections: " + ", ".join(result["unchanged_sections"]) + ".")
        return claims, notes, p.kind

    claims: list[Claim] = []
    for sub_q in p.sub_questions:
        if len(tools.calls) >= max_tool_calls:
            notes.append(f"Stopped early: tool-call budget of {max_tool_calls} reached.")
            break
        res = tools.search(sub_q)
        # The agent applies its own sentence-level check (with stemming) to the
        # results, so a weak search score alone does not discard good evidence.
        found = _best_sentences(sub_q, res["results"], full_q=question)
        if not found:
            refs = _section_fallback(tools, sub_q, catalog)
            found = _best_sentences(sub_q, refs, full_q=question)
        if found:
            # Several parts may be answered by the same sentence; keep it once.
            claims.extend(c for c in found if c.text not in {x.text for x in claims})
            notes.extend(res.get("conflicts", []))
        else:
            notes.append(f"No supporting evidence found for: \"{sub_q}\".")
    return claims, notes, p.kind
