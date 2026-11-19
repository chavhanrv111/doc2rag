import json
from pathlib import Path

from doc2rag.backends.mineru import content_list_to_blocks, find_content_list

FIXTURE = Path(__file__).parent / "fixtures" / "sample_content_list.json"


def load_blocks(include_layout=False):
    items = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return content_list_to_blocks(items, Path("C:/out"), include_layout=include_layout)


def test_maps_all_block_types_and_drops_layout():
    blocks = load_blocks()
    assert [b.type for b in blocks] == [
        "text", "text", "text", "list", "table", "image", "image", "equation", "text", "text"]
    assert blocks[0].text_level == 1 and blocks[2].text_level == 2
    assert blocks[3].text.splitlines() == ["- Software means the licensed programs.",
                                           "- Fees means the amounts in Exhibit A."]


def test_table_image_equation_fields():
    blocks = load_blocks()
    table = blocks[4]
    assert table.caption == ["Exhibit A - Fees"] and table.footnote == ["All amounts in USD"]
    assert table.table_html.startswith("<table>") and table.page_idx == 1
    assert Path(table.img_path).is_absolute()
    assert blocks[6].caption == ["Company logo"]
    assert blocks[7].latex == "F = 1.05 \\times B"


def test_include_layout_keeps_header_and_page_number():
    texts = [b.text for b in load_blocks(include_layout=True)]
    assert "Confidential" in texts and "3" in texts


def test_find_content_list_prefers_exact_legacy(tmp_path):
    d = tmp_path / "doc" / "auto"
    d.mkdir(parents=True)
    (d / "doc_content_list_v2.json").write_text("[]")
    (d / "doc_content_list.json").write_text("[]")
    (d / "other_content_list.json").write_text("[]")
    assert find_content_list(tmp_path, "doc").name == "doc_content_list.json"
