"""Load a document library described by a manifest.

The manifest mirrors what you get from a SharePoint document library:
file path, web URL, last-modified date and who has read access. In a
production tenant these values come from Microsoft Graph
(driveItem + /permissions); here they are kept in a JSON file so the whole
pipeline can be run and tested offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from securerag.ingest.chunker import chunk_document
from securerag.ingest.parsers import parse_file
from securerag.models import Chunk, Document


def load_documents(manifest_path: Path) -> list[Document]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    docs = []
    for entry in manifest["documents"]:
        path = base / entry["path"]
        principals = entry.get("allowed_principals") or []
        if not principals:
            # Fail closed: a document without an ACL is never indexed.
            raise ValueError(f"{entry['doc_id']} has no allowed_principals; refusing to index")
        docs.append(
            Document(
                doc_id=entry["doc_id"],
                title=entry["title"],
                source_url=entry["source_url"],
                last_modified=entry["last_modified"],
                file_type=path.suffix.lstrip(".").lower(),
                sections=parse_file(path),
                allowed_principals=principals,
            )
        )
    return docs


def load_chunks(manifest_path: Path, max_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(manifest_path):
        chunks.extend(chunk_document(doc, max_words=max_words, overlap_words=overlap_words))
    return chunks
