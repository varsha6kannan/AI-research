from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingest.chunker import semantic_chunk
from app.ingest.loader import load_documents
from app.models.medcpt import get_medcpt_article_encoder
from app.vectorstore.chroma_store import delete_by_source_path, upsert_chunks
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
        # Still track that we saw the file.
        state = _load_state()
        state[source_path] = {"last_modified_ts": payload.last_modified_ts, "file_type": payload.file_type}
        _save_state(state)
        return

    all_ids: list[str] = []
    all_emb: list[list[float]] = []
    all_docs: list[str] = []
    all_meta: list[dict[str, Any]] = []

    article_encoder = get_medcpt_article_encoder()
    file_path_hash = _sha256_hex(source_path)

    for doc_i, doc in enumerate(docs):
        chunks = semantic_chunk(doc.content)
        if not chunks:
            continue
        chunk_texts = [_normalize_chunk_text(c.text) for c in chunks if _normalize_chunk_text(c.text)]
        if not chunk_texts:
            continue

        embeddings = article_encoder.encode_texts(chunk_texts, batch_size=settings.embed_batch_size)

        for j, ctext in enumerate(chunk_texts):
            chash = _sha256_hex(ctext)
            # Deterministic id: stable across re-ingests, avoids collision across multi-doc files.
            cid = f"{file_path_hash}:{doc_i}:{chash}"
            all_ids.append(cid)
            all_emb.append(embeddings[j])
            all_docs.append(ctext)
            all_meta.append(
                {
                    "pmid": doc.pmid or "",
                    "title": doc.title,
                    "source_path": source_path,
                    "source_file_type": payload.file_type,
                    "source_doc_index": doc_i,
                    "chunk_hash": chash,
                    "chunk_index": j,
                    "last_modified_ts": payload.last_modified_ts,
                }
            )

    if all_ids:
        # Replace vectors for this file by deleting then upserting.
        # This makes delete events and modifications consistent.
        delete_by_source_path(source_path)
        upsert_chunks(ids=all_ids, embeddings=all_emb, documents=all_docs, metadatas=all_meta)

    state = _load_state()
    state[source_path] = {"last_modified_ts": payload.last_modified_ts, "file_type": payload.file_type}
    _save_state(state)

