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


def delete_by_source_path(source_path: str) -> None:
    col = get_collection()
    # Chroma supports server-side delete by metadata filter.
    col.delete(where={"source_path": source_path})

