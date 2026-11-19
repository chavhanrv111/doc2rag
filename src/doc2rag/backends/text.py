from __future__ import annotations

from pathlib import Path
from typing import Any

from doc2rag.backends.base import Backend, register_backend
from doc2rag.models import Block


@register_backend
class PlainTextBackend(Backend):
    """Dependency-free reader for .txt files: blank-line separated paragraphs."""

    name = "text"
    extensions = frozenset({".txt"})

    @classmethod
    def is_available(cls) -> bool:
        return True

    def parse(self, path: Path, artifacts_dir: Path, **_: Any) -> list[Block]:
        raw = path.read_bytes()
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                content = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        paragraphs = [p.strip() for p in content.replace("\r\n", "\n").split("\n\n")]
        return [Block("text", p) for p in paragraphs if p]
