"""Phase 37-D: PDF image OCR.

These tests cover the pure-Python parts of `chunking/images.py`. The
actual OCR pipeline requires Tesseract installed system-wide; we mock
pytesseract via monkeypatch so the tests run cleanly on CI machines
that don't have it.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from code_rag.chunking import images
from code_rag.chunking.images import (
    extract_pdf_image_text,
    reset_tesseract_cache_for_tests,
    tesseract_available,
)


def test_tesseract_available_caches_negative_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """When pytesseract isn't importable, the probe returns False AND
    caches that result so subsequent calls don't re-shell to check."""
    reset_tesseract_cache_for_tests()
    # Pretend pytesseract is not installed.
    monkeypatch.setitem(sys.modules, "pytesseract", None)

    # First call: probes and caches.
    assert tesseract_available() is False
    # Second call: returns from cache (no probe).
    # We assert this by replacing sys.modules with something different — if
    # the cache works, the change is invisible.
    fake_pyt = types.ModuleType("pytesseract")
    fake_pyt.get_tesseract_version = lambda: "5.0.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pyt)
    # Cache hit: still False from previous probe.
    assert tesseract_available() is False


def test_tesseract_available_caches_positive_result(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_tesseract_cache_for_tests()
    fake_pyt = types.ModuleType("pytesseract")
    fake_pyt.get_tesseract_version = lambda: "5.0.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pyt)
    assert tesseract_available() is True
    # Cache hit even after we break the module.
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    assert tesseract_available() is True


def test_extract_pdf_image_text_no_tesseract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Tesseract isn't available, extract returns [] and never even
    tries to open the PDF."""
    reset_tesseract_cache_for_tests()
    monkeypatch.setattr("code_rag.chunking.images.tesseract_available",
                        lambda: False)
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not a real pdf")  # never opened
    assert extract_pdf_image_text(fake_pdf) == []


def test_extract_pdf_image_text_pdf_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt PDF logs + returns [] without raising."""
    reset_tesseract_cache_for_tests()
    monkeypatch.setattr("code_rag.chunking.images.tesseract_available",
                        lambda: True)

    # pypdf.PdfReader on garbage bytes raises. Verify the wrapper swallows.
    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"\x00\x01\x02 not a real PDF")
    assert extract_pdf_image_text(bogus) == []


def test_extract_pdf_image_text_skips_short_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pages whose OCR returns < 20 chars are dropped (noise filter)."""
    reset_tesseract_cache_for_tests()
    monkeypatch.setattr("code_rag.chunking.images.tesseract_available",
                        lambda: True)

    # Stand up the smallest possible PDF reader stub.
    fake_image = types.SimpleNamespace(
        size=(200, 200),
        # PIL Image protocol bits used by tesseract are duck-typed.
    )
    fake_image_file = types.SimpleNamespace(image=fake_image, data=b"")
    fake_page = types.SimpleNamespace(images=[fake_image_file])
    fake_reader = types.SimpleNamespace(pages=[fake_page])

    monkeypatch.setattr("pypdf.PdfReader", lambda _path: fake_reader)

    fake_pyt = types.ModuleType("pytesseract")
    # Return exactly 5 chars — under _MIN_OCR_CHARS=20.
    fake_pyt.image_to_string = lambda *_a, **_kw: "noise"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pyt)

    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil_image.open = lambda _b: fake_image  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_image  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)

    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"unused")
    assert extract_pdf_image_text(pdf) == []


def test_extract_pdf_image_text_keeps_substantial_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pages with >= 20 chars of OCR'd text are returned."""
    reset_tesseract_cache_for_tests()
    monkeypatch.setattr("code_rag.chunking.images.tesseract_available",
                        lambda: True)

    fake_image = types.SimpleNamespace(size=(400, 300))
    fake_image_file = types.SimpleNamespace(image=fake_image, data=b"")
    page1 = types.SimpleNamespace(images=[fake_image_file])
    page2 = types.SimpleNamespace(images=[fake_image_file])
    fake_reader = types.SimpleNamespace(pages=[page1, page2])

    monkeypatch.setattr("pypdf.PdfReader", lambda _path: fake_reader)

    fake_pyt = types.ModuleType("pytesseract")
    fake_pyt.image_to_string = lambda *_a, **_kw: \
        "OAuth Login Flow: user -> browser -> auth-server"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pyt)

    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil_image.open = lambda _b: fake_image  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_image  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)

    pdf = tmp_path / "diagram.pdf"
    pdf.write_bytes(b"unused")
    out = extract_pdf_image_text(pdf)
    assert len(out) == 2
    assert out[0][0] == 1   # page_no
    assert "OAuth" in out[0][1]
    assert out[1][0] == 2


def test_extract_pdf_image_text_skips_tiny_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Images smaller than 50x50 px are skipped before OCR (noise filter)."""
    reset_tesseract_cache_for_tests()
    monkeypatch.setattr("code_rag.chunking.images.tesseract_available",
                        lambda: True)

    tiny = types.SimpleNamespace(size=(20, 20))     # below threshold
    tiny_file = types.SimpleNamespace(image=tiny, data=b"")
    page = types.SimpleNamespace(images=[tiny_file])
    fake_reader = types.SimpleNamespace(pages=[page])
    monkeypatch.setattr("pypdf.PdfReader", lambda _p: fake_reader)

    # Even if tesseract WOULD return real text, we must not call it on a
    # 20x20 thumbnail. Spy by raising in image_to_string.
    fake_pyt = types.ModuleType("pytesseract")
    def _should_not_be_called(*_a, **_kw):  # pragma: no cover
        raise AssertionError("OCR should not run on tiny images")
    fake_pyt.image_to_string = _should_not_be_called  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pyt)
    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil_image.open = lambda _b: tiny  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_image  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)

    pdf = tmp_path / "tiny.pdf"
    pdf.write_bytes(b"unused")
    assert extract_pdf_image_text(pdf) == []


# Sanity: verify we don't accidentally cache state between tests.
def test_module_constants_are_sane() -> None:
    assert images._MIN_OCR_CHARS >= 10, "noise threshold should reject single glyphs"
    assert images._MIN_IMAGE_DIM >= 32, "tiny-image threshold should skip icons"
    assert "--psm" in images._TESSERACT_CONFIG
