from pathlib import Path

import pytest

from doc2rag.backends import backend_class
from doc2rag.router import UnsupportedFormat, select_backend
from doc2rag.utils import html_table_to_markdown


@pytest.fixture
def availability(monkeypatch):
    def set_(mineru: bool, docling: bool):
        monkeypatch.setattr(backend_class("mineru"), "is_available", classmethod(lambda cls: mineru))
        monkeypatch.setattr(backend_class("docling"), "is_available", classmethod(lambda cls: docling))
    return set_


def test_auto_prefers_mineru_for_pdf(availability):
    availability(True, True)
    assert select_backend(Path("a.pdf")) == "mineru"
    assert select_backend(Path("a.pdf"), pdf_preference=["docling", "mineru"]) == "docling"


def test_office_goes_to_docling_even_if_mineru_requested(availability):
    availability(True, True)
    assert select_backend(Path("a.docx"), backend="mineru") == "docling"
    with pytest.raises(UnsupportedFormat):
        select_backend(Path("a.docx"), backend="mineru", fallback=False)


def test_falls_back_when_preferred_missing(availability):
    availability(False, True)
    assert select_backend(Path("scan.png")) == "docling"


def test_txt_and_legacy_and_unknown(availability):
    availability(False, False)
    assert select_backend(Path("notes.txt")) == "text"
    with pytest.raises(UnsupportedFormat, match="docx"):
        select_backend(Path("old.doc"))
    with pytest.raises(UnsupportedFormat, match="none installed"):
        select_backend(Path("a.pdf"))
    with pytest.raises(UnsupportedFormat, match="no backend"):
        select_backend(Path("a.xyz"))


def test_html_table_to_markdown_colspan_and_pipes():
    md = html_table_to_markdown(
        "<table><tr><th colspan='2'>Head</th></tr><tr><td>a|b</td><td>c<br>d</td></tr></table>")
    assert md.splitlines() == ["| Head |  |", "|---|---|", "| a\\|b | c d |"]
