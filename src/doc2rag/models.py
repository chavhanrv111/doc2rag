from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from doc2rag.utils import html_table_to_markdown

BlockType = Literal["text", "list", "table", "image", "equation", "code"]
TextFormat = Literal["markdown", "text"]


@dataclass
class Block:
    """One structural element of a parsed document, in reading order.

    ``page_idx`` is 0-based. ``text_level`` > 0 marks a heading (1 = top level).
    """

    type: BlockType
    text: str = ""
    page_idx: Optional[int] = None
    text_level: int = 0
    img_path: Optional[str] = None
    caption: list[str] = field(default_factory=list)
    footnote: list[str] = field(default_factory=list)
    table_html: Optional[str] = None
    table_markdown: Optional[str] = None
    latex: Optional[str] = None

    @property
    def is_heading(self) -> bool:
        return self.type == "text" and self.text_level > 0

    def table_as_markdown(self) -> str:
        if self.table_markdown:
            return self.table_markdown
        return html_table_to_markdown(self.table_html) if self.table_html else ""

    def render(self, fmt: TextFormat = "markdown", include_images: bool = True) -> str:
        """Render this block as Markdown or plain text ("" if it carries no text)."""
        md = fmt == "markdown"
        caption, notes = " ".join(self.caption), " ".join(self.footnote)
        if self.is_heading:
            return f"{'#' * min(self.text_level, 6)} {self.text}" if md else self.text
        if self.type in ("text", "list"):
            return self.text
        if self.type == "table":
            title = (f"**Table: {caption}**" if md else f"Table: {caption}") if caption else ""
            return "\n\n".join(p for p in (title, self.table_as_markdown(), notes) if p)
        if self.type == "image":
            if not include_images or not (caption or notes):
                return ""
            label = " ".join(p for p in (caption, notes) if p)
            return f"![{label}]({self.img_path or ''})" if md else f"[Image: {label}]"
        if self.type == "equation":
            body = self.latex or self.text
            return f"$$\n{body}\n$$" if md else body
        if self.type == "code":
            return f"```\n{self.text}\n```" if md else self.text
        return self.text

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if k in ("type", "page_idx") or v not in (None, "", [], 0)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Block":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Span:
    """Character range of one rendered block inside the full document text."""

    start: int
    end: int
    block_index: int
    page_idx: Optional[int]
    heading_path: list[str]


@dataclass
class ParsedDocument:
    source: str
    doc_id: str
    backend: str
    blocks: list[Block]
    artifacts_dir: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(
        self, fmt: TextFormat = "markdown", page_breaks: bool = False, include_images: bool = True
    ) -> tuple[str, list[Span]]:
        """Full document text plus the span of every block (used to map text back to pages)."""
        parts: list[str] = []
        spans: list[Span] = []
        headings: list[tuple[int, str]] = []
        pos, last_page = 0, None
        for i, b in enumerate(self.blocks):
            if b.is_heading:
                while headings and headings[-1][0] >= b.text_level:
                    headings.pop()
                headings.append((b.text_level, b.text))
            rendered = b.render(fmt, include_images)
            if not rendered:
                continue
            if page_breaks and b.page_idx is not None and b.page_idx != last_page:
                marker = f"<!-- page {b.page_idx + 1} -->" if fmt == "markdown" else f"--- page {b.page_idx + 1} ---"
                if parts:
                    pos += 2
                parts.append(marker)
                pos += len(marker)
                last_page = b.page_idx
            if parts:
                pos += 2
            spans.append(Span(pos, pos + len(rendered), i, b.page_idx, [h for _, h in headings]))
            parts.append(rendered)
            pos += len(rendered)
        return "\n\n".join(parts), spans

    def to_markdown(self, page_breaks: bool = False, include_images: bool = True) -> str:
        """Whole document as Markdown (headings, tables, equations), before any chunking."""
        return self.render("markdown", page_breaks, include_images)[0]

    def to_text(self, page_breaks: bool = False) -> str:
        """Whole document as plain text (tables are still Markdown to keep rows/columns)."""
        return self.render("text", page_breaks)[0]

    @property
    def text(self) -> str:
        return self.to_markdown()

    def page_texts(self, fmt: TextFormat = "markdown") -> dict[int, str]:
        """Text per 1-based page number (blocks without a page are attached to page 1)."""
        pages: dict[int, list[str]] = {}
        for b in self.blocks:
            rendered = b.render(fmt)
            if rendered:
                pages.setdefault((b.page_idx or 0) + 1, []).append(rendered)
        return {p: "\n\n".join(parts) for p, parts in sorted(pages.items())}

    def to_dict(self, include_text: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "doc_id": self.doc_id,
            "backend": self.backend,
            "artifacts_dir": self.artifacts_dir,
            "metadata": self.metadata,
        }
        if include_text:
            data["text"] = self.to_markdown()
        data["blocks"] = [b.to_dict() for b in self.blocks]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedDocument":
        return cls(
            source=data["source"],
            doc_id=data["doc_id"],
            backend=data["backend"],
            artifacts_dir=data.get("artifacts_dir"),
            metadata=data.get("metadata", {}),
            blocks=[Block.from_dict(b) for b in data["blocks"]],
        )


@dataclass
class Chunk:
    """Vector-store-ready unit: ``text`` is what you embed, the rest is metadata."""

    id: str
    doc_id: str
    source: str
    chunk_index: int
    content_type: str
    section: str
    heading_path: list[str]
    pages: list[int]
    text: str
    img_path: Optional[str] = None
    table_html: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def metadata(self) -> dict[str, Any]:
        """Scalar-only metadata, accepted by every vector store (Chroma, Pinecone, Qdrant, ...)."""
        meta: dict[str, Any] = {
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "content_type": self.content_type,
            "section": self.section,
            "heading_path": " > ".join(self.heading_path),
            "pages": ",".join(map(str, self.pages)),
            "page_start": self.pages[0] if self.pages else -1,
            "page_end": self.pages[-1] if self.pages else -1,
        }
        if self.img_path:
            meta["img_path"] = self.img_path
        for k, v in self.extra.items():
            meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        return meta

    def to_record(self) -> dict[str, Any]:
        """``{"id", "text", "metadata"}``: the shape most vector-store clients expect."""
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {})}
