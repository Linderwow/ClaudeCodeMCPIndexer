"""Multimodal extraction: pull text out of images embedded in PDFs.

Why this lives separate from `docs.py`
--------------------------------------
The plain text extraction in `docs._chunk_pdf` covers ~95% of code-RAG
use cases. But architecture diagrams, mermaid renders, screenshots of
configuration UIs, and PDF design docs frequently contain text that
ONLY exists inside embedded images — never reachable from
`page.extract_text()`. NVIDIA's Blueprint solves this with NeMo
Retriever Page Elements + Graphic Elements + OCR NIMs (a 7B-class
VLM). We solve it with a thin wrapper around Tesseract.

Practical scope
---------------
- Optional dependency: `pytesseract` + system `tesseract.exe` on PATH.
  If either is missing, image-text extraction is silently skipped
  (the regular PDF text extraction still runs).
- One chunk per page that has at least `min_chars` of OCR'd text.
  The chunk's text is `[OCR page N] <extracted text>` so it shows
  up alongside the regular text page chunks under the same path.
- We don't extract images for embedding (no CLIP). The OCR output
  becomes a regular text chunk that the existing dense + BM25 + rerank
  pipeline scores. This is intentional: code-RAG users search for
  STRINGS the diagram contains ("MNQAlpha state diagram", "OAuth flow
  step 3"), not visual similarity.

Tradeoff
--------
Tesseract on architecture diagrams is mediocre (typical accuracy
60-80% depending on font + DPI). That's still infinitely better than
no signal at all when the PDF is the only place "InvoiceProcessor"
appears as a label inside a flowchart box.
"""
from __future__ import annotations

import io
from pathlib import Path

from code_rag.logging import get

log = get(__name__)


# Tesseract config: assume single uniform block of text per image.
# `--oem 1` forces LSTM engine (better than legacy on small text).
_TESSERACT_CONFIG = "--oem 1 --psm 6"

# Minimum chars in OCR output to keep — under this threshold the result
# is almost always noise (single ligature / stray glyph).
_MIN_OCR_CHARS = 20

# Skip images smaller than 50x50 — tiny icons / bullet graphics produce
# only noise from OCR and waste CPU.
_MIN_IMAGE_DIM = 50


# Cache the "is Tesseract usable" probe result for the process lifetime.
# pytesseract.get_tesseract_version() shells out and is slow to repeat.
_TESSERACT_AVAILABLE: bool | None = None


def tesseract_available() -> bool:
    """True iff pytesseract is importable AND the tesseract binary is
    callable. Result is cached for the process lifetime."""
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE
    try:
        import pytesseract
        try:
            pytesseract.get_tesseract_version()
            _TESSERACT_AVAILABLE = True
        except Exception as e:
            log.info("images.tesseract_binary_missing",
                     hint="install Tesseract-OCR and ensure tesseract.exe is on PATH",
                     err=f"{type(e).__name__}: {e}")
            _TESSERACT_AVAILABLE = False
    except ImportError:
        log.info("images.pytesseract_not_installed",
                 hint="pip install pytesseract to enable PDF image OCR")
        _TESSERACT_AVAILABLE = False
    return _TESSERACT_AVAILABLE


def reset_tesseract_cache_for_tests() -> None:
    """Test helper: clear the availability cache so each test can re-probe."""
    global _TESSERACT_AVAILABLE
    _TESSERACT_AVAILABLE = None


def extract_pdf_image_text(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract OCR'd text from every embedded image in a PDF.

    Returns a list of `(page_number, ocr_text)` tuples. Skips images
    smaller than 50x50 px and OCR results under 20 chars (noise).
    Returns an empty list if Tesseract isn't available — callers MUST
    treat this as the normal "no extra chunks to add" path.

    Best-effort: per-image failures are logged + swallowed so a single
    corrupt image never aborts the whole PDF's OCR pass.
    """
    if not tesseract_available():
        return []

    try:
        import pypdf
        import pytesseract
        from PIL import Image
    except ImportError as e:  # pragma: no cover — covered by tesseract_available
        log.warning("images.dep_missing", err=str(e))
        return []

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as e:
        log.warning("images.pdf_open_fail", path=str(pdf_path), err=str(e))
        return []

    out: list[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        # pypdf >= 4.0 exposes images via `page.images` returning ImageFile
        # objects with `.data` (bytes) and `.image` (PIL.Image-compatible).
        try:
            images = list(page.images)
        except Exception as e:
            # Some PDFs have malformed inline image streams. Log + skip.
            log.debug("images.page_iter_fail",
                      path=str(pdf_path), page=page_no, err=str(e))
            continue

        for img in images:
            try:
                # Either path produces a PIL Image; pypdf normalizes most
                # encodings (FlateDecode, DCTDecode/JPEG, JBIG2, etc).
                pil_img = img.image if hasattr(img, "image") and img.image else \
                    Image.open(io.BytesIO(img.data))
                if pil_img.size[0] < _MIN_IMAGE_DIM or pil_img.size[1] < _MIN_IMAGE_DIM:
                    continue
                text = pytesseract.image_to_string(pil_img, config=_TESSERACT_CONFIG)
            except Exception as e:
                log.debug("images.ocr_fail",
                          path=str(pdf_path), page=page_no,
                          err=f"{type(e).__name__}: {e}")
                continue
            text = (text or "").strip()
            if len(text) >= _MIN_OCR_CHARS:
                out.append((page_no, text))

    return out
