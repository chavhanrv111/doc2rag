import json
from pathlib import Path

import pytest

import doc2rag
from doc2rag.backends.mineru import content_list_to_blocks
from doc2rag.chunker import make_chunker, split_document
from doc2rag.cli import main
from doc2rag.models import ParsedDocument

FIXTURE = Path(__file__).parent / "fixtures" / "sample_content_list.json"


@pytest.fixture
def doc():
    items = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return ParsedDocument("a.pdf", "d1", "mineru", content_list_to_blocks(items, Path("C:/out")))


@pytest.fixture
def corpus(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "terms.txt").write_text("Payment is due in 30 days.\n\nLate fees apply at 1% per month.",
                                   encoding="utf-8")
    (src / "nda.txt").write_text("Confidential information must not be disclosed.", encoding="utf-8")
    return src


def test_full_markdown_before_chunking(doc):
    md = doc.to_markdown()
    assert md.startswith("# MASTER LICENSE AGREEMENT")
    assert "## 1. Definitions" in md and "**Table: Exhibit A - Fees**" in md
    assert "| License | 100,000 |" in md and "$$\nF = 1.05 \\times B\n$$" in md
    assert "![Company logo](" in md
    assert "sig.jpg" not in md  # uncaptioned image carries no text


def test_plain_text_and_page_breaks(doc):
    text = doc.to_text(page_breaks=True)
    assert "#" not in text.split("\n")[1]
    assert text.index("--- page 1 ---") < text.index("--- page 2 ---") < text.index("--- page 4 ---")
    assert "[Image: Company logo]" in text


def test_page_texts(doc):
    pages = doc.page_texts()
    assert sorted(pages) == [1, 2, 3, 4]
    assert "Exhibit A - Fees" in pages[2] and "three years" in pages[4]


def test_spans_point_into_text(doc):
    text, spans = doc.render()
    for s in spans:
        assert text[s.start:s.end] == doc.blocks[s.block_index].render()


def test_external_splitter_keeps_pages_and_sections(doc):
    def by_paragraph(text):
        return text.split("\n\n")

    chunks = split_document(doc, by_paragraph)
    term = next(c for c in chunks if "three years" in c.text)
    assert term.pages == [4] and term.heading_path == ["MASTER LICENSE AGREEMENT", "2. Term"]
    table = next(c for c in chunks if "| License |" in c.text)
    assert table.content_type == "table" and table.pages == [2]


def test_make_chunker_accepts_all_styles(doc):
    class FakeLangChainSplitter:
        def split_text(self, text):
            return [text[:200], text[200:]]

    def doc_level(d: ParsedDocument):
        return doc2rag.chunk_document(d)[:1]

    assert len(make_chunker(None)(doc)) == 6
    assert len(make_chunker(doc2rag.ChunkConfig(max_chars=60, overlap=0))(doc)) > 6
    assert len(make_chunker(FakeLangChainSplitter())(doc)) == 2
    assert len(make_chunker(lambda t: t.split("\n\n"))(doc)) > 6
    assert len(make_chunker(doc_level)(doc)) == 1
    with pytest.raises(TypeError):
        make_chunker(42)


def test_records_and_columns_are_scalar(doc):
    chunks = doc2rag.chunk_document(doc)
    rec = chunks[1].to_record()
    assert set(rec) == {"id", "text", "metadata"}
    assert rec["metadata"]["pages"] == "1,2" and rec["metadata"]["page_start"] == 1
    assert all(isinstance(v, (str, int, float, bool)) for c in chunks for v in c.metadata.values())
    cols = doc2rag.to_columns(chunks)
    assert len(cols["ids"]) == len(cols["documents"]) == len(cols["metadatas"]) == len(chunks)
    assert [len(b) for b in doc2rag.batched(range(10), 4)] == [4, 4, 2]


def test_one_call_api(corpus, tmp_path):
    out = tmp_path / "out"
    text = doc2rag.extract_text(corpus / "terms.txt", output_dir=out)
    assert text == "Payment is due in 30 days.\n\nLate fees apply at 1% per month."

    single = doc2rag.parse(corpus / "nda.txt", output_dir=out)
    assert isinstance(single, ParsedDocument) and single.backend == "text"
    assert len(doc2rag.parse(corpus, output_dir=out)) == 2

    records = doc2rag.load_records(corpus, output_dir=out, extra_metadata={"tenant": "pega"})
    assert {r["metadata"]["source"] for r in records} == {"terms.txt", "nda.txt"}
    assert all(r["metadata"]["tenant"] == "pega" for r in records)
    assert (out / "documents.jsonl").exists() and (out / "chunks.jsonl").exists()

    small = doc2rag.load(corpus, output_dir=out, chunk_size=30, overlap=0)
    assert len(small) > len(records)


def test_parse_raises_on_bad_file(tmp_path):
    bad = tmp_path / "old.doc"
    bad.write_bytes(b"x")
    with pytest.raises(doc2rag.UnsupportedFormat):
        doc2rag.parse(bad, output_dir=tmp_path / "out")


def test_cli_text_only_and_print(corpus, tmp_path, capsys):
    out = tmp_path / "out"
    assert main([str(corpus), "-o", str(out), "--text-only", "--print", "text"]) == 0
    printed = capsys.readouterr().out
    assert "===== nda.txt =====" in printed and "Late fees apply" in printed
    assert not (out / "chunks.jsonl").exists()
    docs = [json.loads(l) for l in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {d["source"] for d in docs} == {"terms.txt", "nda.txt"}
    md_files = list((out / "documents").glob("*/full.md"))
    assert len(md_files) == 2


def test_cli_incremental_and_errors(corpus, tmp_path, capsys):
    out = tmp_path / "out"
    (corpus / "legacy.xls").write_bytes(b"x")
    assert main([str(corpus), "-o", str(out)]) == 1
    errors = json.loads((out / "errors.json").read_text(encoding="utf-8"))
    assert list(errors) == ["legacy.xls"]
    main([str(corpus), "-o", str(out)])
    assert "0 parsed, 2 cached" in capsys.readouterr().err
