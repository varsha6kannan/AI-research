from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingest.chunker import semantic_chunk
from app.ingest.loader import load_documents
from app.models.medcpt import get_medcpt_article_encoder
from app.vectorstore.chroma_store import (
    delete_by_ids,
    delete_by_source_path,
    get_chunk_ids_and_hashes_by_source_path,
    upsert_chunks,
)
from app.watcher.webhook import FileEventType, WebhookPayload


_STATE_PATH = Path("runtime") / "file_state.json"


def _load_state() -> dict[str, Any]:
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict[str, Any]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_chunk_text(text: str) -> str:
    return " ".join(text.split()).strip()


def handle_payload(payload: WebhookPayload) -> None:
    """
    Entrypoint for watcher-triggered ingestion. Synchronous by design
    (called via `asyncio.to_thread`).
    """
    source_path = payload.file_path
    if payload.event_type == FileEventType.deleted:
        delete_by_source_path(source_path)
        state = _load_state()
        if source_path in state:
            state.pop(source_path, None)
            _save_state(state)
        return

    # created / modified
    docs = load_documents(source_path)
    if not docs:
        state = _load_state()
        state[source_path] = {"last_modified_ts": payload.last_modified_ts, "file_type": payload.file_type}
        _save_state(state)
        return

    file_path_hash = _sha256_hex(source_path)
    # Build new chunk records: (cid, ctext, chash, meta). Position-based cid for stable diff.
    new_chunks: list[tuple[str, str, str, dict[str, Any]]] = []
    for doc_i, doc in enumerate(docs):
        chunks = semantic_chunk(doc.content)
        if not chunks:
            continue
        chunk_texts = [_normalize_chunk_text(c.text) for c in chunks if _normalize_chunk_text(c.text)]
        if not chunk_texts:
            continue
        for j, ctext in enumerate(chunk_texts):
            chash = _sha256_hex(ctext)
            cid = f"{file_path_hash}:{doc_i}:{j}"
            meta = {
                "pmid": doc.pmid or "",
                "title": doc.title,
                "source_path": source_path,
                "source_file_type": payload.file_type,
                "source_doc_index": doc_i,
                "chunk_hash": chash,
                "chunk_index": j,
                "last_modified_ts": payload.last_modified_ts,
            }
            new_chunks.append((cid, ctext, chash, meta))

    if not new_chunks:
        state = _load_state()
        state[source_path] = {"last_modified_ts": payload.last_modified_ts, "file_type": payload.file_type}
        _save_state(state)
        return

    if payload.event_type == FileEventType.created:
        # Full ingest: no existing chunks; embed and upsert all.
        to_upsert = new_chunks
        ids_to_delete = []
    else:
        # Modified: load existing, diff. Treat missing/empty chunk_hash as changed (re-embed).
        existing_chunks = get_chunk_ids_and_hashes_by_source_path(source_path)
        existing_ids = set(existing_chunks)
        to_upsert = [
            ch for ch in new_chunks
            if ch[0] not in existing_ids or (existing_chunks.get(ch[0]) or "") != ch[2]
        ]
        new_ids = {ch[0] for ch in new_chunks}
        ids_to_delete = list(existing_ids - new_ids)

    if to_upsert:
        article_encoder = get_medcpt_article_encoder()
        texts = [ch[1] for ch in to_upsert]
        embeddings = article_encoder.encode_texts(texts, batch_size=settings.embed_batch_size)
        ids = [ch[0] for ch in to_upsert]
        documents = [ch[1] for ch in to_upsert]
        metadatas = [ch[3] for ch in to_upsert]
        upsert_chunks(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    if ids_to_delete:
        delete_by_ids(ids_to_delete)

    state = _load_state()
    state[source_path] = {"last_modified_ts": payload.last_modified_ts, "file_type": payload.file_type}
    _save_state(state)

