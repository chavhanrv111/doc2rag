from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from doc2rag.models import Block, Chunk, ParsedDocument, TextFormat
from doc2rag.utils import short_hash

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")

TextSplitter = Callable[[str], list[str]]
"""Any ``str -> list[str]`` function, e.g. LangChain's ``RecursiveCharacterTextSplitter(...).split_text``."""

Chunker = Callable[[ParsedDocument], list[Chunk]]


@dataclass
class ChunkConfig:
    max_chars: int = 1500
    overlap: int = 200
    prefix_headings: bool = True
    keep_uncaptioned_images: bool = False


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split text at sentence (then word) boundaries so no piece exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    cut = text.find(" ", len(text) - overlap)
    return text[cut + 1:] if cut != -1 else text[-overlap:]


def _table_text(block: Block) -> str:
    return block.render("text")


def _split_table(text: str, max_chars: int) -> list[str]:
    """Split a Markdown table by rows, repeating the header in every part."""
    lines = text.split("\n")
    header_end = next((i for i, l in enumerate(lines) if re.match(r"^\|?\s*-{3}", l)), None)
    if len(text) <= max_chars or header_end is None:
        return [text]
    header, rows = lines[: header_end + 1], lines[header_end + 1:]
    parts, current = [], list(header)
    for row in rows:
        if len("\n".join(current + [row])) > max_chars and len(current) > len(header):
            parts.append("\n".join(current))
            current = list(header)
        current.append(row)
    parts.append("\n".join(current))
    return parts


@dataclass
class _State:
    doc: ParsedDocument
    cfg: ChunkConfig
    chunks: list[Chunk] = field(default_factory=list)
    headings: list[tuple[int, str]] = field(default_factory=list)
    buf: str = ""
    pages: set[int] = field(default_factory=set)

    @property
    def heading_path(self) -> list[str]:
        return [h for _, h in self.headings]

    def emit(self, text: str, ctype: str, pages: set[Optional[int]], img_path=None, table_html=None):
        path = self.heading_path
        if self.cfg.prefix_headings and path and not text.startswith(path[-1]):
            text = f"{' > '.join(path)}\n{text}"
        self.chunks.append(Chunk(
            id=short_hash(self.doc.doc_id, len(self.chunks)),
            doc_id=self.doc.doc_id,
            source=self.doc.source,
            chunk_index=len(self.chunks),
            content_type=ctype,
            section=path[-1] if path else "",
            heading_path=path,
            pages=sorted(p + 1 for p in pages if isinstance(p, int)),
            text=text,
            img_path=img_path,
            table_html=table_html,
        ))

    def flush(self, keep_overlap: bool = False):
        tail = _tail(self.buf, self.cfg.overlap) if keep_overlap else ""
        if self.buf.strip():
            self.emit(self.buf.strip(), "text", self.pages)
        self.buf = tail
        self.pages = {max(self.pages)} if tail and self.pages else set()

    def add_text(self, text: str, page: Optional[int]):
        for piece in _split_long(text, max(1, self.cfg.max_chars - self.cfg.overlap - 1)):
            if self.buf and len(self.buf) + 1 + len(piece) > self.cfg.max_chars:
                self.flush(keep_overlap=True)
            self.buf = f"{self.buf}\n{piece}" if self.buf else piece
            if page is not None:
                self.pages.add(page)


def chunk_document(doc: ParsedDocument, config: Optional[ChunkConfig] = None) -> list[Chunk]:
    """Turn parsed blocks into retrieval chunks.

    Text/list blocks are packed up to ``max_chars`` with ``overlap`` and never cross a
    heading. Tables, images, equations and code become standalone chunks.
    """
    cfg = config or ChunkConfig()
    st = _State(doc, cfg)
    for b in doc.blocks:
        if b.is_heading:
            st.flush()
            while st.headings and st.headings[-1][0] >= b.text_level:
                st.headings.pop()
            st.headings.append((b.text_level, b.text))
            st.add_text(b.text, b.page_idx)
        elif b.type in ("text", "list"):
            st.add_text(b.text, b.page_idx)
        elif b.type == "table":
            st.flush()
            for part in _split_table(_table_text(b), cfg.max_chars):
                if part:
                    st.emit(f"[TABLE]\n{part}", "table", {b.page_idx}, img_path=b.img_path,
                            table_html=b.table_html)
        elif b.type == "image":
            caption = " ".join(b.caption + b.footnote).strip()
            if caption or cfg.keep_uncaptioned_images:
                st.flush()
                st.emit(f"[IMAGE] {caption}".strip(), "image", {b.page_idx}, img_path=b.img_path)
        elif b.type in ("equation", "code"):
            st.flush()
            label = "EQUATION" if b.type == "equation" else "CODE"
            st.emit(f"[{label}]\n{b.latex or b.text}", b.type, {b.page_idx}, img_path=b.img_path)
    st.flush()
    return st.chunks


def split_document(
    doc: ParsedDocument,
    splitter: TextSplitter,
    fmt: TextFormat = "markdown",
    prefix_headings: bool = False,
) -> list[Chunk]:
    """Chunk the full document text with *any* external splitter, keeping page/section metadata.

    The splitter only sees plain text; each returned piece is located in the rendered
    document so its pages and heading path can be recovered from block spans.
    """
    text, spans = doc.render(fmt)
    chunks: list[Chunk] = []
    cursor = 0
    for piece in splitter(text):
        if not piece or not piece.strip():
            continue
        start = text.find(piece, cursor)
        if start == -1:
            start = text.find(piece)
        if start == -1:  # splitter rewrote whitespace: anchor on the first line instead
            probe = piece.strip().split("\n", 1)[0][:80]
            start = text.find(probe, max(0, cursor - len(piece)))
        end = start + len(piece) if start != -1 else -1
        hit = [s for s in spans if start != -1 and s.start < end and s.end > start]
        if start != -1:
            cursor = start + 1
        heading_path = hit[0].heading_path if hit else []
        types = {doc.blocks[s.block_index].type for s in hit}
        body = piece.strip()
        if prefix_headings and heading_path and not body.startswith(("#", heading_path[-1])):
            body = f"{' > '.join(heading_path)}\n{body}"
        chunks.append(Chunk(
            id=short_hash(doc.doc_id, "split", len(chunks)),
            doc_id=doc.doc_id,
            source=doc.source,
            chunk_index=len(chunks),
            content_type=types.pop() if len(types) == 1 else ("mixed" if types else "text"),
            section=heading_path[-1] if heading_path else "",
            heading_path=heading_path,
            pages=sorted({s.page_idx + 1 for s in hit if s.page_idx is not None}),
            text=body,
        ))
    return chunks


def make_chunker(
    chunker: Union[None, ChunkConfig, TextSplitter, Chunker, Any] = None,
) -> Chunker:
    """Normalise the many ways a caller can say "how to chunk" into one ``doc -> chunks`` callable.

    Accepts ``None`` (built-in structure-aware chunker), a :class:`ChunkConfig`, a
    ``doc -> list[Chunk]`` function, a ``str -> list[str]`` splitter, or any object with
    ``split_text`` (LangChain / LlamaIndex text splitters).
    """
    if chunker is None or isinstance(chunker, ChunkConfig):
        return lambda doc: chunk_document(doc, chunker)
    split_text = getattr(chunker, "split_text", None)
    if callable(split_text):
        return lambda doc: split_document(doc, split_text)
    if callable(chunker):
        if _accepts_document(chunker):
            return chunker
        return lambda doc: split_document(doc, chunker)
    raise TypeError(f"Unsupported chunker: {chunker!r}")


def _accepts_document(fn: Callable) -> bool:
    """True when ``fn``'s first parameter is annotated as ParsedDocument (a doc -> chunks function)."""
    import inspect

    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if not params:
        return False
    ann = params[0].annotation
    return ann is ParsedDocument or (isinstance(ann, str) and ann.split(".")[-1] == "ParsedDocument")
