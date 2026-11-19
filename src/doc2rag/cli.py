from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from doc2rag.backends import backend_class, backend_names
from doc2rag.chunker import ChunkConfig
from doc2rag.pipeline import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="doc2rag",
        description="Parse documents into full text and vector-store-ready chunks.",
        epilog="Outputs: <output>/documents.jsonl (full text per document), <output>/chunks.jsonl, "
               "<output>/documents/<name>_<hash>/{full.md,blocks.json,images}",
    )
    p.add_argument("inputs", nargs="*", help="Files and/or directories")
    p.add_argument("-o", "--output", default=None, help="Output directory (default: $DOC2RAG_OUTPUT_DIR or ./doc2rag_output)")
    p.add_argument("-b", "--backend", default="auto", choices=["auto", *backend_names()])
    p.add_argument("--pdf-preference", default="mineru,docling",
                   help="Backend order for PDFs/images when --backend auto")
    p.add_argument("-m", "--method", default="auto", choices=["auto", "txt", "ocr"],
                   help="auto: detect; txt: text layer only; ocr: force OCR (scans)")
    p.add_argument("-l", "--lang", default=None, help="MinerU OCR language, e.g. en, latin, ch")
    p.add_argument("--mineru-backend", default="pipeline",
                   help="MinerU -b value (pipeline is the CPU-friendly default)")
    p.add_argument("--device", default=None, help="MinerU device, e.g. cpu, cuda")
    p.add_argument("--timeout", type=int, default=None, help="Per-file MinerU timeout (seconds)")

    g = p.add_argument_group("text / chunking")
    g.add_argument("--text-only", action="store_true",
                   help="Skip chunking; write full text only (documents.jsonl + full.md)")
    g.add_argument("--print", dest="print_text", choices=["markdown", "text"],
                   help="Print the full text of the parsed documents to stdout")
    g.add_argument("--chunk-size", type=int, default=1500, help="Max characters per text chunk")
    g.add_argument("--overlap", type=int, default=200, help="Characters of overlap between text chunks")
    g.add_argument("--no-heading-prefix", action="store_true", help="Do not prefix chunks with heading path")
    g.add_argument("--keep-uncaptioned-images", action="store_true")
    g.add_argument("--include-layout", action="store_true", help="Keep page headers/footers")

    p.add_argument("--no-incremental", action="store_true", help="Re-parse even unchanged files")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--system-certs", action="store_true",
                   help="Trust the OS certificate store for model downloads (corporate proxies; needs doc2rag[certs])")
    p.add_argument("--list-backends", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s",
                        stream=sys.stderr)

    if args.list_backends:
        for name in backend_names():
            cls = backend_class(name)
            status = "installed" if cls.is_available() else "missing"
            print(f"{name:8} {status:9} {' '.join(sorted(cls.extensions))}")
        return 0
    if not args.inputs:
        p.error("at least one input is required")
    if args.system_certs:
        os.environ["DOC2RAG_SYSTEM_CERTS"] = "1"

    report = run(
        args.inputs,
        output_dir=args.output,
        backend=args.backend,
        pdf_preference=[s.strip() for s in args.pdf_preference.split(",") if s.strip()],
        chunker=ChunkConfig(args.chunk_size, args.overlap, not args.no_heading_prefix,
                            args.keep_uncaptioned_images),
        chunk=not args.text_only,
        incremental=not args.no_incremental,
        recursive=not args.no_recursive,
        method=args.method,
        lang=args.lang,
        mineru_backend=args.mineru_backend,
        device=args.device,
        timeout=args.timeout,
        include_layout=args.include_layout,
    )
    if args.print_text:
        for doc in report.documents:
            print(f"\n===== {doc.source} =====\n")
            print(doc.render(args.print_text, page_breaks=True)[0])
    print(report.summary(), file=sys.stderr)
    for name, err in report.errors.items():
        print(f"  ERROR {name}: {err}", file=sys.stderr)
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
