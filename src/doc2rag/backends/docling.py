from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

from doc2rag.backends.base import Backend, register_backend
from doc2rag.models import Block


@register_backend
class DoclingBackend(Backend):
    """Parses PDF, Office (docx/pptx/xlsx), HTML, Markdown and images in-process via Docling."""

    name = "docling"
    extensions = frozenset({
        ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".xhtml", ".md", ".csv",
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
    })

    def __init__(self) -> None:
        self._converters: dict[str, Any] = {}

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("docling") is not None

    def _converter(self, method: str, images_scale: float):
        key = f"{method}:{images_scale}"
        if key not in self._converters:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

            opts = PdfPipelineOptions()
            opts.do_table_structure = True
            opts.generate_picture_images = True
            opts.images_scale = images_scale
            opts.do_ocr = method != "txt"
            if method == "ocr":
                opts.ocr_options.force_full_page_ocr = True
            self._converters[key] = DocumentConverter(format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
            })
        return self._converters[key]

    def parse(
        self,
        path: Path,
        artifacts_dir: Path,
        method: str = "auto",
        images_scale: float = 2.0,
        include_layout: bool = False,
        timeout: Optional[int] = None,
        **_: Any,
    ) -> list[Block]:
        out_dir = artifacts_dir / "docling"
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        result = self._converter(method, images_scale).convert(str(path))
        doc = result.document
        (out_dir / f"{path.stem}.md").write_text(doc.export_to_markdown(), encoding="utf-8")
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(doc.export_to_dict(), ensure_ascii=False), encoding="utf-8")
        return docling_to_blocks(doc, img_dir, include_layout=include_layout)


def _page_idx(item: Any) -> Optional[int]:
    for prov in getattr(item, "prov", None) or []:
        if getattr(prov, "page_no", 0) > 0:
            return prov.page_no - 1
    return None


def _ref_texts(doc: Any, refs: Any) -> list[str]:
    texts = []
    for ref in refs or []:
        try:
            text = ref.resolve(doc).text.strip()
        except Exception:
            continue
        if text:
            texts.append(text)
    return texts


def docling_to_blocks(doc: Any, img_dir: Path, include_layout: bool = False) -> list[Block]:
    from docling_core.types.doc import DocItemLabel, PictureItem, TableItem, TextItem

    attached = {
        ref.cref
        for item in [*doc.tables, *doc.pictures]
        for ref in [*(item.captions or []), *(getattr(item, "footnotes", None) or [])]
    }
    layout_labels = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

    blocks: list[Block] = []
    pic_no = 0
    for item, _level in doc.iterate_items():
        page = _page_idx(item)
        label = getattr(item, "label", None)

        if isinstance(item, TableItem):
            blocks.append(Block(
                "table", page_idx=page,
                caption=_ref_texts(doc, item.captions),
                footnote=_ref_texts(doc, getattr(item, "footnotes", None)),
                table_html=item.export_to_html(doc=doc) or None,
                table_markdown=item.export_to_markdown(doc=doc) or None,
            ))
        elif isinstance(item, PictureItem):
            img_path = None
            image = item.get_image(doc)
            if image is not None:
                pic_no += 1
                target = img_dir / f"picture_{pic_no:04d}.png"
                image.save(target, "PNG")
                img_path = str(target.resolve())
            blocks.append(Block(
                "image", page_idx=page, img_path=img_path,
                caption=_ref_texts(doc, item.captions),
                footnote=_ref_texts(doc, getattr(item, "footnotes", None)),
            ))
        elif isinstance(item, TextItem):
            text = (item.text or "").strip()
            if not text or item.self_ref in attached:
                continue
            if label in layout_labels:
                if include_layout:
                    blocks.append(Block("text", text, page))
            elif label == DocItemLabel.TITLE:
                blocks.append(Block("text", text, page, text_level=1))
            elif label == DocItemLabel.SECTION_HEADER:
                blocks.append(Block("text", text, page, text_level=int(getattr(item, "level", 1) or 1) + 1))
            elif label == DocItemLabel.LIST_ITEM:
                blocks.append(Block("list", f"- {text}", page))
            elif label == DocItemLabel.FORMULA:
                blocks.append(Block("equation", text, page, latex=text))
            elif label == DocItemLabel.CODE:
                blocks.append(Block("code", text, page))
            else:
                blocks.append(Block("text", text, page))
    return blocks
