"""Turn Word, PDF, Excel and text files into structured sections.

Sections keep their heading (Word), page number (PDF) or sheet name (Excel),
so every answer can cite *where* in a document it came from.
"""

from __future__ import annotations

from pathlib import Path

from securerag.models import Section


def parse_docx(path: Path) -> list[Section]:
    import docx

    document = docx.Document(str(path))
    sections: list[Section] = []
    heading, buf = "Introduction", []

    def flush() -> None:
        text = "\n".join(t for t in buf if t.strip())
        if text:
            sections.append(Section(heading=heading, text=text))

    for para in document.paragraphs:
        style = (para.style.name or "").lower() if para.style is not None else ""
        if style.startswith("heading") or style == "title":
            flush()
            heading, buf = para.text.strip() or heading, []
        else:
            buf.append(para.text)
    flush()

    # Word tables are flattened row by row so their values stay searchable.
    for t_index, table in enumerate(document.tables, start=1):
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
        if rows:
            sections.append(Section(heading=f"Table {t_index}", text="\n".join(rows)))
    return sections


def _looks_like_heading(line: str) -> bool:
    words = line.split()
    return 0 < len(words) <= 6 and line[-1] not in ".,;:!?" and not line[0].isdigit()


def parse_pdf(path: Path) -> list[Section]:
    """One or more sections per page. Short unpunctuated lines are treated as
    headings; wrapped body lines are re-joined into sentences."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections: list[Section] = []
    for i, page in enumerate(reader.pages, start=1):
        lines = [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
        if not lines:
            # Image-only (scanned) page. Kept visible so it can be routed to OCR
            # (e.g. Azure AI Document Intelligence) instead of silently dropped.
            sections.append(Section(f"Page {i}", "[No extractable text on this page - needs OCR]", page=i))
            continue
        heading, body = f"Page {i}", []
        for ln in lines:
            if _looks_like_heading(ln):
                if body:
                    sections.append(Section(heading, " ".join(body), page=i))
                heading, body = ln, []
            else:
                body.append(ln)
        if body:
            sections.append(Section(heading, " ".join(body), page=i))
    return sections


def parse_xlsx(path: Path) -> list[Section]:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    sections = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        header = [str(h) if h is not None else "" for h in rows[0]]
        lines = []
        for row in rows[1:]:
            if all(v is None for v in row):
                continue
            # "Header: value" pairs keep each cell's meaning after chunking.
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if v is not None]
            lines.append("; ".join(pairs))
        if lines:
            sections.append(Section(heading=f"Sheet {ws.title}", text="\n".join(lines)))
    wb.close()
    return sections


def parse_text(path: Path) -> list[Section]:
    sections: list[Section] = []
    heading, buf = "Introduction", []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if "".join(buf).strip():
                sections.append(Section(heading=heading, text="\n".join(buf).strip()))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append(Section(heading=heading, text="\n".join(buf).strip()))
    return sections


PARSERS = {
    ".docx": parse_docx,
    ".pdf": parse_pdf,
    ".xlsx": parse_xlsx,
    ".md": parse_text,
    ".txt": parse_text,
}


def parse_file(path: Path) -> list[Section]:
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return parser(path)
