from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

from doc2rag.backends import get_backend
from doc2rag.chunker import make_chunker
from doc2rag.models import Chunk, ParsedDocument
from doc2rag.router import select_backend, supported_extensions
from doc2rag.utils import file_sha256, short_hash

log = logging.getLogger("doc2rag")

PathLike = Union[str, Path]
Sources = Union[PathLike, Iterable[PathLike]]

# Options that change parser output; a change invalidates the incremental cache.
_RESULT_OPTIONS = ("method", "lang", "mineru_backend", "formula", "table", "images_scale", "include_layout")


def default_output_dir() -> Path:
    return Path(os.environ.get("DOC2RAG_OUTPUT_DIR", "doc2rag_output"))


def _as_list(sources: Sources) -> list[PathLike]:
    return [sources] if isinstance(sources, (str, Path)) else list(sources)


def parse_file(
    path: PathLike,
    output_dir: Optional[PathLike] = None,
    backend: str = "auto",
    pdf_preference: Sequence[str] = ("mineru", "docling"),
    **options: Any,
) -> ParsedDocument:
    """Parse one file into a :class:`ParsedDocument` (blocks in reading order, no chunking)."""
    path = Path(path).resolve()
    name = select_backend(path, backend, pdf_preference)
    sha = file_sha256(path)
    root = Path(output_dir or default_output_dir()).resolve()
    artifacts = root / "documents" / f"{path.stem}_{short_hash(path, length=8)}"
    artifacts.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    blocks = get_backend(name).parse(path, artifacts, **options)
    doc = ParsedDocument(
        source=path.name,
        doc_id=sha[:16],
        backend=name,
        blocks=blocks,
        artifacts_dir=str(artifacts),
        metadata={
            "path": str(path),
            "sha256": sha,
            "options": {k: options[k] for k in _RESULT_OPTIONS if k in options},
            "parse_seconds": round(time.perf_counter() - started, 2),
        },
    )
    (artifacts / "blocks.json").write_text(json.dumps(doc.to_dict(), ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (artifacts / "full.md").write_text(doc.to_markdown(page_breaks=True), encoding="utf-8")
    return doc


def iter_input_files(inputs: Sources, recursive: bool = True) -> list[Path]:
    exts = supported_extensions() | {".doc", ".ppt", ".xls"}
    files: list[Path] = []
    for item in _as_list(inputs):
        p = Path(item)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            files += sorted(f for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in exts)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def iter_documents(
    inputs: Sources,
    output_dir: Optional[PathLike] = None,
    backend: str = "auto",
    incremental: bool = True,
    recursive: bool = True,
    pdf_preference: Sequence[str] = ("mineru", "docling"),
    errors: Optional[dict[str, str]] = None,
    raise_errors: bool = False,
    **options: Any,
) -> Iterator[ParsedDocument]:
    """Yield parsed documents one by one; unchanged files are served from the cache.

    Failures are logged and recorded in ``errors`` (file name -> message) unless
    ``raise_errors`` is true.
    """
    out = Path(output_dir or default_output_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    opts_key = json.dumps({"backend": backend, **{k: options.get(k) for k in _RESULT_OPTIONS}}, sort_keys=True)

    files = iter_input_files(inputs, recursive)
    for i, f in enumerate(files, 1):
        key = str(f.resolve())
        try:
            entry = manifest.get(key)
            blocks_file = Path(entry["blocks"]) if entry else None
            if (incremental and entry and entry["options"] == opts_key and blocks_file.exists()
                    and entry["sha256"] == file_sha256(f)):
                doc = ParsedDocument.from_dict(json.loads(blocks_file.read_text(encoding="utf-8")))
                doc.metadata["cached"] = True
                log.info("[%d/%d] cached  %s", i, len(files), f.name)
            else:
                log.info("[%d/%d] parsing %s", i, len(files), f.name)
                doc = parse_file(f, out, backend, pdf_preference, **options)
                manifest[key] = {
                    "sha256": doc.metadata["sha256"],
                    "options": opts_key,
                    "backend": doc.backend,
                    "blocks": str(Path(doc.artifacts_dir) / "blocks.json"),
                }
                manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
                log.info("        %s: %d blocks in %.1fs", doc.backend, len(doc.blocks),
                         doc.metadata["parse_seconds"])
        except Exception as e:  # one bad file must not stop a batch
            if raise_errors:
                raise
            if errors is not None:
                errors[f.name] = f"{type(e).__name__}: {e}"
            log.error("[%d/%d] FAILED  %s: %s", i, len(files), f.name, e)
            continue
        yield doc


@dataclass
class RunReport:
    documents: list[ParsedDocument] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    output_dir: Optional[Path] = None

    @property
    def skipped(self) -> list[str]:
        return [d.source for d in self.documents if d.metadata.get("cached")]

    @property
    def chunks_file(self) -> Optional[Path]:
        return self.output_dir / "chunks.jsonl" if self.output_dir and self.chunks else None

    def summary(self) -> str:
        cached = len(self.skipped)
        return (f"{len(self.documents)} documents ({len(self.documents) - cached} parsed, {cached} cached), "
                f"{len(self.chunks)} chunks, {len(self.errors)} errors -> {self.output_dir}")


def run(
    inputs: Sources,
    output_dir: Optional[PathLike] = None,
    backend: str = "auto",
    chunker: Any = None,
    chunk: bool = True,
    extra_metadata: Optional[dict[str, Any]] = None,
    incremental: bool = True,
    recursive: bool = True,
    pdf_preference: Sequence[str] = ("mineru", "docling"),
    write: bool = True,
    **options: Any,
) -> RunReport:
    """Parse, (optionally) chunk, and write ``documents.jsonl`` + ``chunks.jsonl`` to ``output_dir``.

    ``chunker`` is anything accepted by :func:`doc2rag.chunker.make_chunker`.
    """
    out = Path(output_dir or default_output_dir()).resolve()
    report = RunReport(output_dir=out)
    chunk_fn = make_chunker(chunker) if chunk else None
    for doc in iter_documents(inputs, out, backend, incremental, recursive, pdf_preference,
                              errors=report.errors, **options):
        report.documents.append(doc)
        if chunk_fn:
            chunks = chunk_fn(doc)
            if extra_metadata:
                for c in chunks:
                    c.extra.update(extra_metadata)
            report.chunks += chunks

    if write:
        with (out / "documents.jsonl").open("w", encoding="utf-8") as fh:
            for d in report.documents:
                fh.write(json.dumps({
                    "doc_id": d.doc_id, "source": d.source, "backend": d.backend,
                    "pages": len(d.page_texts()), "text": d.to_markdown(), "metadata": d.metadata,
                }, ensure_ascii=False) + "\n")
        if chunk_fn:
            with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
                for c in report.chunks:
                    fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (out / "errors.json").write_text(json.dumps(report.errors, indent=1), encoding="utf-8")
    return report
