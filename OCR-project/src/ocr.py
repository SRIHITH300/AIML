"""
OCR and text extraction module for document classification.

Supports:
- Text-based PDFs (PyMuPDF)
- Scanned PDFs / images (RapidOCR-ONNX as primary, Tesseract as fallback)
- Graceful degradation when engines are unavailable
"""

import io
import os
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF (alias for pymupdf)
from PIL import Image

_rapidocr = None
_tesseract_available = False

try:
    from rapidocr_onnxruntime import RapidOCR
    _rapidocr = RapidOCR()
except Exception:
    _rapidocr = None

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    _tesseract_available = True
except Exception:
    _tesseract_available = False


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract embedded text from a PDF using PyMuPDF."""
    text_parts = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)
        doc.close()
    except Exception:
        pass
    return "\n".join(text_parts).strip()


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes using RapidOCR (primary) or Tesseract (fallback)."""
    if _rapidocr is not None:
        try:
            result, _ = _rapidocr(image_bytes)
            if result:
                lines = [line[1] for line in result if len(line) >= 2 and line[1]]
                return "\n".join(lines).strip()
        except Exception:
            pass

    if _tesseract_available:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(img).strip()
        except Exception:
            pass

    return ""


def _ocr_pdf_scanned(pdf_bytes: bytes, dpi: int = 200) -> str:
    """Convert a scanned PDF to images and OCR each page."""
    text_parts = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            page_text = _ocr_image_bytes(img_bytes)
            if page_text:
                text_parts.append(page_text)
        doc.close()
    except Exception:
        pass
    return "\n".join(text_parts).strip()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF. First tries embedded text; if that yields
    little content, falls back to OCR on rendered page images.
    """
    embedded = _extract_pdf_text(pdf_bytes)
    if len(embedded) >= 50:
        return embedded

    ocr_text = _ocr_pdf_scanned(pdf_bytes)
    if ocr_text:
        return ocr_text

    return embedded


def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from an image using OCR."""
    return _ocr_image_bytes(image_bytes)


def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file (PDF or image) given its path."""
    path = Path(file_path)
    ext = path.suffix.lower()

    with open(path, "rb") as f:
        raw = f.read()

    if ext in PDF_EXTENSIONS:
        return extract_text_from_pdf(raw)
    elif ext in IMAGE_EXTENSIONS:
        return extract_text_from_image(raw)
    else:
        return ""


def extract_text_from_bytes(raw: bytes, filename: str = "") -> str:
    """Extract text from raw bytes, using the filename hint for format detection."""
    ext = Path(filename).suffix.lower() if filename else ""

    if ext in PDF_EXTENSIONS:
        return extract_text_from_pdf(raw)
    elif ext in IMAGE_EXTENSIONS:
        return extract_text_from_image(raw)
    else:
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
            doc.close()
            return extract_text_from_pdf(raw)
        except Exception:
            return extract_text_from_image(raw)


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, remove control chars."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_engines_available() -> dict:
    """Return a dict describing which OCR engines are available."""
    return {
        "rapidocr": _rapidocr is not None,
        "tesseract": _tesseract_available,
        "pymupdf": True,
    }
