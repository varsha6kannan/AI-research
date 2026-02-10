"""
Intent index for semantic query disambiguation.
Builds document-level view from Chroma, embeds with MedCPT article encoder,
clusters with k-means, and persists centroid embeddings + descriptions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from app.config import PROJECT_ROOT, settings
from app.models.medcpt import get_medcpt_article_encoder
from app.vectorstore.chroma_store import get_document_view, get_all_unique_source_paths


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _get_document_views() -> list[tuple[str, str, str]]:
    """Return list of (source_path, title, content_snippet) for each document in Chroma."""
    source_paths = get_all_unique_source_paths()
    out: list[tuple[str, str, str]] = []
    for sp in source_paths:
        view = get_document_view(sp, content_max_chars=2000)
        if view is None:
            continue
        title, content_snippet = view
        if not title and not content_snippet:
            continue
        out.append((sp, title or "Untitled", content_snippet))
    return out


def build_intent_index() -> list[dict[str, Any]]:
    """
    Build intent index from Chroma: document-level embeddings, k-means clustering,
    centroid + description per intent. Returns list of { id, embedding, description }.
    """
    views = _get_document_views()
    if not views:
        return []
    if len(views) < 2:
        # Single document: one intent with short description
        enc = get_medcpt_article_encoder()
        text = views[0][1] + " " + views[0][2]
        emb = enc.encode_texts([text], batch_size=1)[0]
        title = (views[0][1] or "").strip()
        desc = (title[:80] + ("..." if len(title) > 80 else "")) if title else "Single document"
        return [
            {
                "id": "i0",
                "embedding": emb,
                "description": desc,
            }
        ]

    texts = [title + " " + content for (_, title, content) in views]
    encoder = get_medcpt_article_encoder()
    embeddings = encoder.encode_texts(texts, batch_size=settings.embed_batch_size)
    X = np.asarray(embeddings, dtype=np.float32)

    k = min(settings.intent_cluster_k, len(views))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Centroids: mean then L2-normalize for cosine
    centroids: list[np.ndarray] = []
    for i in range(k):
        mask = labels == i
        cen = X[mask].mean(axis=0)
        norm = np.linalg.norm(cen)
        if norm > 1e-12:
            cen = cen / norm
        centroids.append(cen)

    # Description: intent-level short phrase (semantic category), not full paper title
    _MAX_INTENT_DESC_CHARS = 80

    def _intent_level_description(title: str, fallback: str) -> str:
        if not (title or "").strip():
            return fallback
        t = title.strip()
        if len(t) <= _MAX_INTENT_DESC_CHARS:
            return t
        return t[:_MAX_INTENT_DESC_CHARS].strip() + "..."

    intents = []
    for i in range(k):
        cen = centroids[i]
        mask = labels == i
        indices = np.where(mask)[0]
        best_idx = indices[
            int(np.argmax([_cosine_sim(X[j], cen) for j in indices]))
        ]
        title = views[best_idx][1]
        description = _intent_level_description(title, f"Topic cluster {i}")
        intents.append({
            "id": f"i{i}",
            "embedding": centroids[i].tolist(),
            "description": description,
        })
    return intents


def get_intent_index_path() -> Path:
    return PROJECT_ROOT / settings.intent_index_path


def load_intent_index() -> list[dict[str, Any]]:
    """Load intent index from JSON. Returns empty list if missing or invalid."""
    path = get_intent_index_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []


def save_intent_index(intents: list[dict[str, Any]]) -> None:
    path = get_intent_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(intents, ensure_ascii=False, indent=2), encoding="utf-8")


def get_or_build_intent_index() -> list[dict[str, Any]]:
    """
    Load intent index from disk; if missing or empty, build from Chroma and persist.
    """
    intents = load_intent_index()
    if intents:
        return intents
    intents = build_intent_index()
    if intents:
        save_intent_index(intents)
    return intents


def get_domains_for_titles(titles: list[str]) -> list[str]:
    """
    Map each document title to its nearest intent description (semantic domain).
    Used for Gate 4: conceptual underspecification (domains from top-K docs).
    Returns list of intent-level descriptions, one per title; empty if no intents.
    """
    intents = get_or_build_intent_index()
    if not intents or not titles:
        return []
    encoder = get_medcpt_article_encoder()
    title_embeddings = encoder.encode_texts(titles, batch_size=min(len(titles), 32))
    intent_embeddings = [np.asarray(i["embedding"], dtype=np.float32) for i in intents]
    out: list[str] = []
    for te in title_embeddings:
        e = np.asarray(te, dtype=np.float32)
        best_idx = int(np.argmax([float(np.dot(e, ie)) for ie in intent_embeddings]))
        out.append(intents[best_idx]["description"])
    return out
