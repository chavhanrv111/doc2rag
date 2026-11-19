"""One-call helpers: the shortest path from files to RAG-ready data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from doc2rag.chunker import ChunkConfig
from doc2rag.models import Chunk, ParsedDocument, TextFormat
from doc2rag.pipeline import PathLike, Sources, iter_documents, run


def parse(source: Sources, **options: Any) -> Union[ParsedDocument, list[ParsedDocument]]:
    """Parse without chunking. A single file returns one document, anything else a list.

    Raises on failure (use :func:`doc2rag.run` for fault-tolerant batches).
    """
    docs = list(iter_documents(source, raise_errors=True, **options))
    single = isinstance(source, (str, Path)) and Path(source).is_file()
    return docs[0] if single else docs


def extract_text(
    source: PathLike, fmt: TextFormat = "markdown", page_breaks: bool = False, **options: Any
) -> str:
    """Full text of one file (Markdown by default) — bring your own chunker/LLM."""
    doc = parse(source, **options)
    assert isinstance(doc, ParsedDocument)
    return doc.render(fmt, page_breaks)[0]


def load(
    sources: Sources,
    chunker: Any = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
    **options: Any,
) -> list[Chunk]:
    """Parse + chunk files/folders and return :class:`Chunk` objects ready to embed.

    ``chunker`` may be ``None`` (built-in, structure-aware), a :class:`ChunkConfig`, a
    LangChain/LlamaIndex text splitter, a ``str -> list[str]`` function, or a
    ``(doc: ParsedDocument) -> list[Chunk]`` function.
    """
    if chunker is None and (chunk_size or overlap is not None):
        defaults = ChunkConfig()
        chunker = ChunkConfig(max_chars=chunk_size or defaults.max_chars,
                              overlap=defaults.overlap if overlap is None else overlap)
    report = run(sources, chunker=chunker, extra_metadata=extra_metadata, **options)
    return report.chunks


def load_records(sources: Sources, **kwargs: Any) -> list[dict[str, Any]]:
    """Like :func:`load` but returns ``{"id", "text", "metadata"}`` dicts (scalar metadata)."""
    return [c.to_record() for c in load(sources, **kwargs)]
