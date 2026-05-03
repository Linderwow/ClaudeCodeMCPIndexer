"""Document chunker for PDF/DOCX/MD.

Stub in Phase 4 — a real implementation lands in Phase 6. The interface is
stable so the MCP ingest_doc tool wires against it now.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from code_rag.logging import get
from code_rag.models import Chunk, ChunkKind
from code_rag.util.hashing import chunk_id

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")


@dataclass
class _Section:
    trail: str          # "Parent > Child" heading breadcrumb
    body: str           # section content including its heading line
    start_line: int
    end_line: int


def _split_by_heading(text: str) -> list[_Section]:
    """Split markdown by ATX headings. The returned sections cover the whole
    document (pre-heading preamble goes under symbol "")."""
    lines = text.splitlines()
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []       # (level, heading text)
    cur_body: list[str] = []
    cur_start = 1

    def flush(end_line: int) -> None:
        body = "\n".join(cur_body).strip("\n")
        if body:
            trail = " > ".join(s for _, s in stack)
            sections.append(_Section(trail=trail, body=body,
                                     start_line=cur_start, end_line=end_line))

    for idx, ln in enumerate(lines, start=1):
        m = _HEADING_RE.match(ln)
        if m:
            flush(idx - 1)
            cur_body = [ln]
            cur_start = idx
            level = len(m.group(1))
            heading = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            cur_body.append(ln)
    flush(len(lines))
    return sections

log = get(__name__)


class DocChunker:
    def __init__(self, min_chars: int, max_chars: int) -> None:
        self._min = min_chars
        self._max = max_chars

    def chunk_file(self, repo: str, abs_path: Path, rel_path: str) -> list[Chunk]:
        suffix = abs_path.suffix.lower()
        if suffix == ".md":
            return self._chunk_markdown(repo, abs_path, rel_path)
        if suffix == ".pdf":
            return self._chunk_pdf(repo, abs_path, rel_path)
        if suffix == ".docx":
            return self._chunk_docx(repo, abs_path, rel_path)
        if suffix in (".html", ".scss", ".css"):
            # Plain text pack — no AST for these. The language id ends up in
            # the chunk so users can filter `lang=html` if needed.
            return self._chunk_plaintext(repo, abs_path, rel_path, language=suffix.lstrip("."))
        log.debug("docs.unsupported", path=rel_path, suffix=suffix)
        return []

    def _chunk_plaintext(
        self, repo: str, abs_path: Path, rel_path: str, language: str,
    ) -> list[Chunk]:
        try:
            text = abs_path.read_text("utf-8", errors="replace")
        except OSError:
            return []
        return self._windowize(repo, rel_path, text, language=language)

    # ---- format-specific ---------------------------------------------------

    def _chunk_markdown(self, repo: str, abs_path: Path, rel_path: str) -> list[Chunk]:
        try:
            text = abs_path.read_text("utf-8", errors="replace")
        except OSError:
            return []
        # Heading-aware split: each "# Heading" starts a new section. Within a
        # section, we pack into max_chars windows. Symbol is the heading trail.
        sections = _split_by_heading(text)
        out: list[Chunk] = []
        for sec in sections:
            for idx, w in enumerate(self._split_text_windows(sec.body), start=1):
                sym = sec.trail + (f"#{idx}" if idx > 1 else "")
                out.append(Chunk(
                    id=chunk_id(repo, rel_path, sym or f"line:{sec.start_line}", w,
                                sec.start_line),
                    repo=repo, path=rel_path, language="markdown",
                    symbol=sym or None, kind=ChunkKind.DOC,
                    start_line=sec.start_line, end_line=sec.end_line, text=w,
                ))
        return out

    def _chunk_pdf(self, repo: str, abs_path: Path, rel_path: str) -> list[Chunk]:
        try:
            import pypdf
        except ImportError:  # pragma: no cover
            log.warning("docs.pypdf_missing")
            return []
        try:
            reader = pypdf.PdfReader(str(abs_path))
        except Exception as e:
            log.warning("docs.pdf_parse_fail", path=rel_path, err=str(e))
            return []
        # Each page becomes its own "section" for coarse locality; Phase 6 adds
        # header-aware splitting.
        parts: list[tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                parts.append((i, page.extract_text() or ""))
            except Exception:
                continue
        out: list[Chunk] = []
        for page_no, text in parts:
            if len(text) < self._min:
                continue
            windows = self._split_text_windows(text)
            for idx, w in enumerate(windows, start=1):
                sym = f"page:{page_no}#{idx}"
                out.append(Chunk(
                    id=chunk_id(repo, rel_path, sym, w, page_no),
                    repo=repo, path=rel_path, language="pdf",
                    symbol=sym, kind=ChunkKind.DOC,
                    start_line=page_no, end_line=page_no, text=w,
                ))

        # Phase 37-D: extract OCR'd text from embedded images. Architecture
        # diagrams, mermaid renders, screenshots — text that ONLY exists
        # inside images and is invisible to page.extract_text(). Best-effort:
        # silently no-ops if Tesseract isn't installed (the regular text
        # chunks above are still emitted).
        from code_rag.chunking.images import extract_pdf_image_text
        try:
            ocr_pages = extract_pdf_image_text(abs_path)
        except Exception as e:
            log.debug("docs.pdf_ocr_fail", path=rel_path, err=str(e))
            ocr_pages = []
        for page_no, ocr_text in ocr_pages:
            for idx, w in enumerate(self._split_text_windows(ocr_text), start=1):
                sym = f"page:{page_no}/ocr#{idx}"
                # Tag chunk text so it's visible to BOTH bm25 (token match)
                # and the searcher's match_reason hint that this came from OCR.
                marked = f"[OCR page {page_no}]\n{w}"
                out.append(Chunk(
                    id=chunk_id(repo, rel_path, sym, marked, page_no),
                    repo=repo, path=rel_path, language="pdf",
                    symbol=sym, kind=ChunkKind.DOC,
                    start_line=page_no, end_line=page_no, text=marked,
                ))
        return out

    def _chunk_docx(self, repo: str, abs_path: Path, rel_path: str) -> list[Chunk]:
        try:
            import docx  # python-docx
        except ImportError:  # pragma: no cover
            log.warning("docs.pydocx_missing")
            return []
        try:
            d = docx.Document(str(abs_path))
        except Exception as e:
            log.warning("docs.docx_parse_fail", path=rel_path, err=str(e))
            return []
        text = "\n\n".join(p.text for p in d.paragraphs if p.text)
        return self._windowize(repo, rel_path, text, language="docx")

    # ---- utils -------------------------------------------------------------

    def _windowize(
        self, repo: str, rel_path: str, text: str, language: str,
    ) -> list[Chunk]:
        if len(text) < self._min:
            return []
        out: list[Chunk] = []
        for idx, w in enumerate(self._split_text_windows(text), start=1):
            sym = f"window:{idx}"
            out.append(Chunk(
                id=chunk_id(repo, rel_path, sym, w, idx),
                repo=repo, path=rel_path, language=language,
                symbol=sym, kind=ChunkKind.DOC,
                start_line=1, end_line=1, text=w,
            ))
        return out

    def _split_text_windows(self, text: str) -> list[str]:
        if not text:
            return []
        # Paragraph-aware packing: split on blank lines, pack until max_chars.
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        out: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for p in paras:
            if buf_len + len(p) + 2 > self._max and buf:
                out.append("\n\n".join(buf))
                buf = []
                buf_len = 0
            buf.append(p)
            buf_len += len(p) + 2
        if buf and buf_len >= self._min:
            out.append("\n\n".join(buf))
        return out
