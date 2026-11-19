import json
from pathlib import Path

from doc2rag.backends.mineru import content_list_to_blocks
from doc2rag.chunker import ChunkConfig, _split_long, chunk_document
from doc2rag.models import Block, ParsedDocument

FIXTURE = Path(__file__).parent / "fixtures" / "sample_content_list.json"


def make_doc(blocks):
    return ParsedDocument(source="a.pdf", doc_id="d1", backend="test", blocks=blocks)


def fixture_doc():
    items = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return make_doc(content_list_to_blocks(items, Path("C:/out")))


def test_sections_tables_images_equations():
    chunks = chunk_document(fixture_doc())
    types = [c.content_type for c in chunks]
    assert types == ["text", "text", "table", "image", "equation", "text"]

    definitions, table = chunks[1], chunks[2]
    assert definitions.heading_path == ["MASTER LICENSE AGREEMENT", "1. Definitions"]
    assert definitions.pages == [1, 2]
    assert "| License | 100,000 |" in table.text and "Exhibit A - Fees" in table.text
    assert table.text.startswith("MASTER LICENSE AGREEMENT > 1. Definitions\n[TABLE]")
    assert table.table_html and table.pages == [2]
    assert chunks[3].text.endswith("[IMAGE] Company logo")
    assert chunks[5].section == "2. Term" and chunks[5].pages == [4]


def test_uncaptioned_images_optional():
    cfg = ChunkConfig(keep_uncaptioned_images=True)
    assert sum(c.content_type == "image" for c in chunk_document(fixture_doc(), cfg)) == 2


def test_ids_deterministic_and_unique():
    a, b = chunk_document(fixture_doc()), chunk_document(fixture_doc())
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_long_text_respects_size_and_overlap():
    sentences = [f"Clause {i} states an obligation of the licensee." for i in range(200)]
    doc = make_doc([Block("text", "Terms", 0, text_level=1), Block("text", " ".join(sentences), 0)])
    chunks = chunk_document(doc, ChunkConfig(max_chars=500, overlap=100, prefix_headings=False))
    assert len(chunks) > 5
    assert all(len(c.text) <= 500 for c in chunks)
    tail = chunks[1].text[-60:]
    assert tail in chunks[2].text


def test_split_long_handles_unbroken_text():
    pieces = _split_long("x" * 1000, 300)
    assert all(len(p) <= 300 for p in pieces) and "".join(pieces) == "x" * 1000


def test_heading_stack_pops_siblings():
    doc = make_doc([
        Block("text", "A", 0, text_level=1), Block("text", "A1", 0, text_level=2),
        Block("text", "B", 0, text_level=1), Block("text", "body", 0),
    ])
    assert chunk_document(doc)[-1].heading_path == ["B"]


def test_large_table_split_repeats_header():
    rows = "".join(f"<tr><td>row {i}</td><td>{i * 10}</td></tr>" for i in range(100))
    doc = make_doc([Block("table", page_idx=0, table_html=f"<table><tr><th>Name</th><th>Val</th></tr>{rows}</table>")])
    chunks = chunk_document(doc, ChunkConfig(max_chars=400))
    assert len(chunks) > 1
    assert all("| Name | Val |" in c.text for c in chunks)
