"""Answer generation with mandatory citations.

Both generators return GeneratedAnswer(text, refs, conflict). `refs` are the
1-based source numbers the answer cites. The pipeline rejects any answer
that cites nothing or cites a source it was not given.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from securerag.models import SearchHit
from securerag.search.local_backend import tokenize

_REF = re.compile(r"\[(\d+)\]")
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class GeneratedAnswer:
    text: str
    refs: list[int]
    conflict: bool = False
    insufficient: bool = False


class Generator(Protocol):
    def generate(self, question: str, hits: list[SearchHit]) -> GeneratedAnswer: ...


def format_sources(hits: list[SearchHit]) -> str:
    blocks = []
    for n, h in enumerate(hits, start=1):
        c = h.chunk
        where = f"{c.section}" + (f", page {c.page}" if c.page else "")
        blocks.append(f"[{n}] {c.title} | {where} | last modified {c.last_modified}\n{c.content}")
    return "\n\n".join(blocks)


def extract_refs(text: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in _REF.finditer(text):
        r = int(m.group(1))
        if 1 <= r <= n_sources and r not in seen:
            seen.append(r)
    return seen


class ExtractiveGenerator:
    """Offline generator: quotes the most relevant sentences, each with its [n].

    No LLM, no hallucination - useful for local demos, tests and as a
    fallback when Azure OpenAI is unavailable.
    """

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[SearchHit]) -> GeneratedAnswer:
        q = set(tokenize(question))
        candidates = []
        for n, h in enumerate(hits, start=1):
            body = h.chunk.content.split("\n", 1)[-1]  # drop "title - section" prefix
            for s in _SENTENCES.split(body):
                s = s.strip()
                if len(s.split()) < 4:
                    continue
                overlap = len(q & set(tokenize(s)))
                if overlap:
                    # Prefer higher-ranked sources on ties.
                    candidates.append((overlap, -n, s, n))
        candidates.sort(reverse=True)
        if not candidates or candidates[0][0] < min(2, len(q)):
            return GeneratedAnswer("", [], insufficient=True)
        best = candidates[0][0]
        # Keep only the sentences that match the question best (ties), plus
        # the best sentence from every other document that ties, so both
        # sides of a conflict are quoted.
        top = [c for c in candidates if c[0] == best]
        picked, refs = [], []
        for _, _, s, n in top[: self.max_sentences]:
            picked.append(f"{s} [{n}]")
            if n not in refs:
                refs.append(n)
        if not picked:
            return GeneratedAnswer("", [], insufficient=True)
        return GeneratedAnswer(" ".join(picked), refs)


SYSTEM_PROMPT = """You answer employee questions using ONLY the numbered sources provided.
Rules:
- Every sentence in your answer must end with at least one citation like [1] or [2][3].
- Never use outside knowledge. If the sources do not contain the answer, set "status" to "insufficient_evidence" and leave "answer" empty.
- If sources disagree (for example different figures in an older and a newer document), set "status" to "conflicting_evidence" and describe each version with its citation and date. Do not pick one silently.
- Be concise and factual.
Respond with JSON: {"status": "answered" | "insufficient_evidence" | "conflicting_evidence", "answer": "..."}"""


class AzureOpenAIGenerator:
    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str = "2024-10-21",
                 client=None):
        if client is None:
            from openai import AzureOpenAI
            client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self._client = client
        self._deployment = deployment

    def generate(self, question: str, hits: list[SearchHit]) -> GeneratedAnswer:
        user = f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"
        resp = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        )
        return parse_llm_json(resp.choices[0].message.content or "", len(hits))


def parse_llm_json(raw: str, n_sources: int) -> GeneratedAnswer:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return GeneratedAnswer("", [], insufficient=True)
    status = str(data.get("status", "")).lower()
    text = str(data.get("answer", "")).strip()
    refs = extract_refs(text, n_sources)
    return GeneratedAnswer(
        text=text,
        refs=refs,
        conflict=status == "conflicting_evidence",
        insufficient=status == "insufficient_evidence" or not text,
    )
