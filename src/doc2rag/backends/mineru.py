from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from doc2rag.backends.base import Backend, register_backend
from doc2rag.models import Block

LAYOUT_TYPES = frozenset({"header", "footer", "page_number", "aside_text", "page_footnote"})
_LATEX_DELIMS = re.compile(r"^\s*(\$\$|\$|\\\[|\\\()\s*|\s*(\$\$|\$|\\\]|\\\))\s*$")


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(v).strip() for v in value if str(v).strip()]


def _abs_img(rel: Optional[str], base_dir: Path) -> Optional[str]:
    if not rel:
        return None
    p = Path(rel)
    return str(p if p.is_absolute() else (base_dir / p).resolve())


def content_list_to_blocks(
    items: list[dict[str, Any]], base_dir: Path, include_layout: bool = False
) -> list[Block]:
    """Map MinerU's legacy ``*_content_list.json`` items to :class:`Block` objects."""
    blocks: list[Block] = []
    for it in items:
        t = it.get("type")
        page = it.get("page_idx")
        if t in LAYOUT_TYPES:
            if include_layout and (it.get("text") or "").strip():
                blocks.append(Block("text", it["text"].strip(), page))
            continue
        if t == "text":
            text = (it.get("text") or "").strip()
            if text:
                blocks.append(Block("text", text, page, text_level=int(it.get("text_level") or 0)))
        elif t == "list":
            lines = [f"- {x.strip()}" for x in it.get("list_items", []) if str(x).strip()]
            text = "\n".join(lines) or (it.get("text") or "").strip()
            if text:
                blocks.append(Block("list", text, page))
        elif t == "table":
            blocks.append(Block(
                "table",
                page_idx=page,
                img_path=_abs_img(it.get("img_path"), base_dir),
                caption=_as_list(it.get("table_caption")),
                footnote=_as_list(it.get("table_footnote")),
                table_html=(it.get("table_body") or "").strip() or None,
            ))
        elif t == "image":
            blocks.append(Block(
                "image",
                page_idx=page,
                img_path=_abs_img(it.get("img_path"), base_dir),
                caption=_as_list(it.get("image_caption") or it.get("img_caption")),
                footnote=_as_list(it.get("image_footnote") or it.get("img_footnote")),
            ))
        elif t == "equation":
            raw = (it.get("text") or it.get("latex") or "").strip()
            if raw:
                blocks.append(Block(
                    "equation", raw, page,
                    latex=_LATEX_DELIMS.sub("", raw),
                    img_path=_abs_img(it.get("img_path"), base_dir),
                ))
        elif t == "code":
            text = (it.get("code_body") or it.get("text") or "").strip()
            if text:
                blocks.append(Block("code", text, page, caption=_as_list(it.get("code_caption"))))
        else:
            text = str(it.get("text") or it.get("content") or "").strip()
            if text:
                blocks.append(Block("text", text, page))
    return blocks


def find_content_list(out_dir: Path, stem: str) -> Path:
    candidates = [p for p in out_dir.rglob("*content_list.json") if "_v2" not in p.name]
    exact = [p for p in candidates if p.name == f"{stem}_content_list.json"]
    found = exact or candidates
    if not found:
        raise FileNotFoundError(f"MinerU produced no content_list.json under {out_dir}")
    return max(found, key=lambda p: p.stat().st_mtime)


def _mineru_executable() -> Optional[str]:
    local = Path(sys.executable).parent / ("mineru.exe" if sys.platform == "win32" else "mineru")
    return str(local) if local.exists() else shutil.which("mineru")


@register_backend
class MineruBackend(Backend):
    """Runs the MinerU 3.x CLI (``mineru -p ... -o ...``) and reads its content list."""

    name = "mineru"
    extensions = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})

    @classmethod
    def is_available(cls) -> bool:
        return _mineru_executable() is not None

    def parse(
        self,
        path: Path,
        artifacts_dir: Path,
        method: str = "auto",
        lang: Optional[str] = None,
        mineru_backend: str = "pipeline",
        device: Optional[str] = None,
        formula: bool = True,
        table: bool = True,
        timeout: Optional[int] = None,
        include_layout: bool = False,
        **_: Any,
    ) -> list[Block]:
        out_dir = artifacts_dir / "mineru"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [_mineru_executable() or "mineru", "-p", str(path), "-o", str(out_dir), "-m", method,
               "-b", mineru_backend]
        if lang:
            cmd += ["-l", lang]
        if device:
            cmd += ["-d", device]
        if not formula:
            cmd += ["-f", "false"]
        if not table:
            cmd += ["-t", "false"]

        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
            raise RuntimeError(f"MinerU failed (exit {proc.returncode}) on {path.name}:\n{tail}")

        content_file = find_content_list(out_dir, path.stem)
        items = json.loads(content_file.read_text(encoding="utf-8"))
        return content_list_to_blocks(items, content_file.parent, include_layout=include_layout)
