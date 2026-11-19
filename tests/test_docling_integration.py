import json

import pytest

from doc2rag import run
from doc2rag.backends import backend_class

pytestmark = pytest.mark.skipif(not backend_class("docling").is_available(), reason="docling not installed")


def test_docx_end_to_end(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Services Agreement", level=1)
    d.add_paragraph("The vendor shall provide support services.")
    d.add_heading("Fees", level=2)
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Item", "Amount"
    table.cell(1, 0).text, table.cell(1, 1).text = "Support", "5,000"
    d.add_paragraph("First bullet", style="List Bullet")
    src = tmp_path / "in" / "agreement.docx"
    src.parent.mkdir()
    d.save(src)
    (tmp_path / "in" / "notes.txt").write_text("Para one.\n\nPara two.", encoding="utf-8")

    out = tmp_path / "out"
    report = run([src.parent], out)
    assert not report.errors, report.errors
    by_source = {doc.source: doc for doc in report.documents}
    assert by_source["agreement.docx"].backend == "docling"
    assert by_source["notes.txt"].backend == "text"

    types = [b.type for b in by_source["agreement.docx"].blocks]
    assert "table" in types and "list" in types
    headings = [b.text for b in by_source["agreement.docx"].blocks if b.is_heading]
    assert headings == ["Services Agreement", "Fees"]

    lines = [json.loads(l) for l in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    table_chunk = next(c for c in lines if c["content_type"] == "table")
    assert "| Support | 5,000 |" in table_chunk["text"]
    assert table_chunk["heading_path"] == ["Services Agreement", "Fees"]

    again = run([src.parent], out)
    assert sorted(again.skipped) == ["agreement.docx", "notes.txt"]
    assert [c.id for c in again.chunks] == [c.id for c in report.chunks]
