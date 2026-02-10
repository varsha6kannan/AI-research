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


def get_all_unique_source_paths() -> list[str]:
    """
    Collect all unique source_path values from the chunk collection.
    Paginates through the collection to support large corpora.
    """
    col = get_collection()
    seen: set[str] = set()
    offset = 0
    while True:
        page = col.get(
            limit=_CHUNK_GET_PAGE_SIZE,
            offset=offset,
            include=["metadatas"],
        )
        ids = page.get("ids") or []
        metadatas = page.get("metadatas") or []
        for meta in metadatas:
            if isinstance(meta, dict) and "source_path" in meta:
                seen.add(meta["source_path"])
        if len(ids) < _CHUNK_GET_PAGE_SIZE:
            break
        offset += len(ids)
    return sorted(seen)


def get_document_view(source_path: str, content_max_chars: int = 2000) -> tuple[str, str] | None:
    """
    For a given source_path, return (title, content_snippet) from its chunks.
    Chunks are ordered by chunk_index; title from first chunk; content_snippet is
    concatenated chunk texts truncated to content_max_chars.
    Returns None if no chunks found.
    """
    col = get_collection()
    page = col.get(
        where={"source_path": source_path},
        include=["documents", "metadatas"],
        limit=_CHUNK_GET_PAGE_SIZE,
    )
    ids = page.get("ids") or []
    documents = page.get("documents") or []
    metadatas = page.get("metadatas") or []
    if not ids:
        return None
    # Sort by chunk_index
    indexed = []
    for i, meta in enumerate(metadatas):
        meta = meta or {}
        idx = meta.get("chunk_index", 0)
        if isinstance(idx, (int, float)):
            indexed.append((int(idx), meta.get("title", ""), documents[i] if i < len(documents) else ""))
        else:
            indexed.append((0, meta.get("title", ""), documents[i] if i < len(documents) else ""))
    indexed.sort(key=lambda x: x[0])
    title = indexed[0][1] if indexed else ""
    parts = [t for (_, _, t) in indexed if t]
    content_snippet = (" ".join(parts))[:content_max_chars] if parts else ""
    return (title, content_snippet)

