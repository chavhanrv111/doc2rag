from __future__ import annotations

from pathlib import Path
from typing import Sequence

from doc2rag.backends import backend_class, backend_names

PDF_LIKE = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
LEGACY_OFFICE = frozenset({".doc", ".ppt", ".xls"})


class UnsupportedFormat(ValueError):
    pass


def supported_extensions() -> set[str]:
    exts: set[str] = set()
    for name in backend_names():
        exts |= backend_class(name).extensions
    return exts


def select_backend(
    path: Path,
    backend: str = "auto",
    pdf_preference: Sequence[str] = ("mineru", "docling"),
    fallback: bool = True,
) -> str:
    """Pick the backend for ``path``.

    ``auto`` uses ``pdf_preference`` for PDFs/images and Docling for everything else.
    An explicit backend that cannot read the file type falls back to any capable,
    installed backend when ``fallback`` is true (e.g. ``mineru`` + ``.docx`` -> docling).
    """
    ext = path.suffix.lower()
    if ext in LEGACY_OFFICE:
        raise UnsupportedFormat(
            f"{path.name}: legacy binary Office format; save it as {ext}x (docx/pptx/xlsx) first")

    if backend == "auto":
        order = list(pdf_preference) if ext in PDF_LIKE else []
    else:
        order = [backend]
    if backend == "auto" or fallback:
        order += [n for n in backend_names() if n not in order]

    tried = []
    for name in order:
        cls = backend_class(name)
        if ext not in cls.extensions:
            continue
        if cls.is_available():
            return name
        tried.append(name)

    if tried:
        raise UnsupportedFormat(
            f"{path.name}: needs one of {tried}, none installed (pip install \"doc2rag[{tried[0]}]\")")
    raise UnsupportedFormat(f"{path.name}: no backend handles '{ext}' files")
