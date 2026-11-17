"""Section-aware chunking.

Chunks never cross a section boundary, so a citation like
"Travel Policy 2025 > Meal allowance" is always accurate. Long sections are
split on paragraph/sentence boundaries with a small word overlap.
"""

from __future__ import annotations

import hashlib
import re

from securerag.models import Chunk, Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_long(text: str, max_words: int, overlap_words: int) -> list[str]:
    units = [u.strip() for u in _SENTENCE_SPLIT.split(text) if u.strip()]
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        n = len(unit.split())
        if current and count + n > max_words:
            pieces.append("\n".join(current))
            tail = " ".join(current).split()[-overlap_words:] if overlap_words else []
            current, count = ([" ".join(tail)] if tail else []), len(tail)
        current.append(unit)
        count += n
    if current:
        # One sentence / table row per line keeps rows intact for citation.
        pieces.append("\n".join(current))
    return pieces


def chunk_document(doc: Document, max_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s_index, section in enumerate(doc.sections):
        for p_index, piece in enumerate(_split_long(section.text, max_words, overlap_words)):
            raw_id = f"{doc.doc_id}|{s_index}|{p_index}"
            # Azure AI Search keys allow letters, digits, '_', '-', '='.
            chunk_id = hashlib.sha1(raw_id.encode()).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=section.heading,
                    # Title + section prefix improves keyword and vector recall.
                    content=f"{doc.title} - {section.heading}\n{piece}",
                    source_url=doc.source_url,
                    last_modified=doc.last_modified,
                    file_type=doc.file_type,
                    allowed_principals=list(doc.allowed_principals),
                    page=section.page,
                )
            )
    return chunks
