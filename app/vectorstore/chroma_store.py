from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import settings


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


@lru_cache(maxsize=1)
def get_collection() -> Collection:
    client = get_chroma_client()
    # cosine is appropriate for normalized dense embeddings
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    *,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    if not ids:
        return
    col = get_collection()
    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


# Page size when fetching existing chunks by source_path (Chroma may cap results).
_CHUNK_GET_PAGE_SIZE = 1000


def get_chunk_ids_and_hashes_by_source_path(source_path: str) -> dict[str, str]:
    """
    Load existing chunk ids and their chunk_hash metadata for a file.
    Uses pagination so all chunks are retrieved even if they exceed default page size.
    Rows missing chunk_hash (legacy) get "" so the pipeline treats them as changed (re-embed).
    """
    col = get_collection()
    result: dict[str, str] = {}
    offset = 0
    while True:
        page = col.get(
            where={"source_path": source_path},
            include=["metadatas"],
            limit=_CHUNK_GET_PAGE_SIZE,
            offset=offset,
        )
        ids = page.get("ids") or []
        metadatas = page.get("metadatas") or []
        for id_, meta in zip(ids, metadatas, strict=False):
            meta = meta or {}
            # Missing or empty chunk_hash -> treat as legacy (re-embed)
            result[id_] = (meta.get("chunk_hash") or "") or ""
        if len(ids) < _CHUNK_GET_PAGE_SIZE:
            break
        offset += len(ids)
    return result


def delete_by_ids(ids: list[str]) -> None:
    if not ids:
        return
    col = get_collection()
    col.delete(ids=ids)


def delete_by_source_path(source_path: str) -> None:
    col = get_collection()
    # Chroma supports server-side delete by metadata filter.
    col.delete(where={"source_path": source_path})

