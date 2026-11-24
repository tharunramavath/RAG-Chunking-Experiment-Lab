"""
Document Loader module for text and PDF documents.
"""

import io
from typing import Union, BinaryIO

try:
    import pypdf
except ImportError:
    pypdf = None


def load_text_from_file(file_obj: Union[BinaryIO, bytes, str], filename: str = "") -> str:
    """
    Extract raw text from a text or PDF file object or path.
    """
    if isinstance(file_obj, str):
        # File path
        if file_obj.lower().endswith(".pdf"):
            with open(file_obj, "rb") as f:
                return _extract_pdf(f)
        else:
            with open(file_obj, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    # Bytes or file-like stream
    if filename.lower().endswith(".pdf"):
        return _extract_pdf(file_obj)
    else:
        if isinstance(file_obj, bytes):
            return file_obj.decode("utf-8", errors="ignore")
        content = file_obj.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)


def _extract_pdf(stream: Union[BinaryIO, bytes]) -> str:
    """Extract all text pages from a PDF stream using pypdf."""
    if pypdf is None:
        raise ImportError(
            "The 'pypdf' library is required to extract text from PDF files. "
            "Please install it using: pip install pypdf"
        )
    if isinstance(stream, bytes):
        stream = io.BytesIO(stream)
    reader = pypdf.PdfReader(stream)
    text_parts = []
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text.strip())
    return "\n\n".join(text_parts)
