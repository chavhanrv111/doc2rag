"""doc2rag: turn any document into structured text and vector-store-ready chunks.

Quick start::

    import doc2rag

    text   = doc2rag.extract_text("contract.pdf")          # full Markdown, no chunking
    doc    = doc2rag.parse("contract.pdf")                 # blocks + to_markdown()/page_texts()
    chunks = doc2rag.load("contracts/")                    # list[Chunk], ready to embed
    rows   = doc2rag.load_records("contracts/")            # [{"id", "text", "metadata"}]
"""

from doc2rag.api import extract_text, load, load_records, parse
from doc2rag.backends import Backend, BackendNotAvailable, get_backend, register_backend
from doc2rag.certs import use_system_certs
from doc2rag.chunker import ChunkConfig, chunk_document, make_chunker, split_document
from doc2rag.integrations import Doc2RagLoader, batched, to_columns, to_haystack, to_langchain, to_llamaindex
from doc2rag.models import Block, Chunk, ParsedDocument
from doc2rag.pipeline import RunReport, iter_documents, parse_file, run
from doc2rag.router import UnsupportedFormat, select_backend

__version__ = "0.2.0"

__all__ = [
    "Backend",
    "BackendNotAvailable",
    "Block",
    "Chunk",
    "ChunkConfig",
    "Doc2RagLoader",
    "ParsedDocument",
    "RunReport",
    "UnsupportedFormat",
    "batched",
    "chunk_document",
    "extract_text",
    "get_backend",
    "iter_documents",
    "load",
    "load_records",
    "make_chunker",
    "parse",
    "parse_file",
    "register_backend",
    "run",
    "select_backend",
    "split_document",
    "to_columns",
    "to_haystack",
    "to_langchain",
    "to_llamaindex",
    "use_system_certs",
]
