from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


PMID_RE = re.compile(r"\bPMID\s*:\s*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LoadedDoc:
    title: str
    content: str
    pmid: str | None


def _extract_title_from_text(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        return s[:200] if s else "Untitled"
    return "Untitled"


def _extract_pmid(text: str) -> str | None:
    m = PMID_RE.search(text)
    return m.group(1) if m else None


def _text_from_section_list(sections: list[dict] | None) -> str:
    """Extract and join 'text' fields from a list of section dicts (e.g. abstract, body_text)."""
    if not sections:
        return ""
    parts: list[str] = []
    for item in sections:
        if isinstance(item, dict) and "text" in item:
            t = item.get("text")
            if t is not None and str(t).strip():
                parts.append(str(t).strip())
    return "\n\n".join(parts)


def _load_json(path: Path) -> list[LoadedDoc]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    # Support nested metadata.title (e.g. CORD-19 / paper_id style) or top-level title
    metadata = obj.get("metadata") or {}
    title = str(metadata.get("title") or obj.get("title") or "Untitled")
    logger.info("Loaded title: %s", title)
    # Top-level content string, or build from abstract + body_text
    content = str(obj.get("content") or "").strip()
    if not content:
        abstract_text = _text_from_section_list(obj.get("abstract"))
        body_text = _text_from_section_list(obj.get("body_text"))
        content = "\n\n".join(filter(None, [abstract_text, body_text]))
    pmid = obj.get("pmid") or metadata.get("pmid")
    pmid = str(pmid) if pmid is not None and str(pmid).strip() else None
    if not content.strip():
        return []
    return [LoadedDoc(title=title, content=content, pmid=pmid)]


def _load_jsonl(path: Path) -> list[LoadedDoc]:
    docs: list[LoadedDoc] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        title = str(obj.get("title") or "Untitled")
        logger.info("Loaded title: %s", title)
        content = str(obj.get("content") or "")
        pmid = obj.get("pmid")
        pmid = str(pmid) if pmid is not None and str(pmid).strip() else None
        if content.strip():
            docs.append(LoadedDoc(title=title, content=content, pmid=pmid))
    return docs


def _load_textlike(path: Path) -> list[LoadedDoc]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = _extract_title_from_text(text)
    pmid = _extract_pmid(text)
    content = text.strip()
    if not content:
        return []
    return [LoadedDoc(title=title, content=content, pmid=pmid)]


def _load_pdf(path: Path) -> list[LoadedDoc]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    doc = fitz.open(str(path))
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    text = "\n".join(parts).strip()
    title = path.stem
    pmid = _extract_pmid(text)
    if not text:
        return []
    return [LoadedDoc(title=title, content=text, pmid=pmid)]


def load_documents(file_path: str) -> list[LoadedDoc]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix in (".txt", ".md"):
        return _load_textlike(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    # Unknown file type: ignore
    return []

