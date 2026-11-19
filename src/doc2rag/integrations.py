"""Adapters for popular RAG frameworks. Each framework is imported lazily, only when used."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Optional

from doc2rag.models import Chunk


def to_langchain(chunks: Iterable[Chunk]) -> list[Any]:
    """``list[langchain_core.documents.Document]``"""
    try:
        from langchain_core.documents import Document
    except ImportError as e:
        raise ImportError("pip install langchain-core") from e
    return [Document(page_content=c.text, metadata=c.metadata, id=c.id) for c in chunks]


def to_llamaindex(chunks: Iterable[Chunk]) -> list[Any]:
    """``list[llama_index.core.schema.TextNode]`` (use with ``VectorStoreIndex(nodes)``)."""
    try:
        from llama_index.core.schema import TextNode
    except ImportError as e:
        raise ImportError("pip install llama-index-core") from e
    return [TextNode(id_=c.id, text=c.text, metadata=c.metadata) for c in chunks]


def to_haystack(chunks: Iterable[Chunk]) -> list[Any]:
    """``list[haystack.Document]``"""
    try:
        from haystack import Document
    except ImportError as e:
        raise ImportError("pip install haystack-ai") from e
    return [Document(id=c.id, content=c.text, meta=c.metadata) for c in chunks]


def to_columns(chunks: Iterable[Chunk]) -> dict[str, list[Any]]:
    """Column format used by Chroma ``collection.add(**cols)`` and similar clients.

    Returns ``{"ids": [...], "documents": [...], "metadatas": [...]}``.
    """
    chunks = list(chunks)
    return {
        "ids": [c.id for c in chunks],
        "documents": [c.text for c in chunks],
        "metadatas": [c.metadata for c in chunks],
    }


def batched(items: Iterable[Any], size: int = 64) -> Iterator[list[Any]]:
    """Yield lists of ``size`` items — handy for embedding APIs with batch limits."""
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class Doc2RagLoader:
    """LangChain-style loader: ``Doc2RagLoader("docs/").load()`` -> ``list[Document]``.

    Duck-typed (``load``/``lazy_load``), so it works in LangChain pipelines without
    making langchain a dependency of doc2rag.
    """

    def __init__(self, sources: Any, chunker: Any = None, extra_metadata: Optional[dict] = None,
                 **options: Any) -> None:
        self.sources = sources
        self.chunker = chunker
        self.extra_metadata = extra_metadata
        self.options = options

    def lazy_load(self) -> Iterator[Any]:
        from doc2rag.chunker import make_chunker
        from doc2rag.pipeline import iter_documents

        chunk_fn = make_chunker(self.chunker)
        for doc in iter_documents(self.sources, **self.options):
            chunks = chunk_fn(doc)
            for c in chunks:
                c.extra.update(self.extra_metadata or {})
            yield from to_langchain(chunks)

    def load(self) -> list[Any]:
        return list(self.lazy_load())
