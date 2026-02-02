from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.models.medcpt import get_medcpt_article_encoder


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    # Very lightweight sentence splitter (no extra deps).
    raw = _SENT_SPLIT_RE.split(text.strip())
    out: list[str] = []
    for s in raw:
        s2 = " ".join(s.split())
        if s2:
            out.append(s2)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


def semantic_chunk(text: str) -> list[Chunk]:
    """
    Sentence-split then boundary-detect using embedding similarity.
    Also enforces max/min chunk character lengths.
    """
    sents = _sentences(text)
    if not sents:
        return []

    # If it's already small, just return one chunk.
    if len(text) <= settings.chunk_max_chars:
        return [Chunk(text=text.strip(), index=0)]

    enc = get_medcpt_article_encoder()
    sent_emb = enc.encode_texts(sents, batch_size=settings.sentence_embed_batch_size)
    emb = np.asarray(sent_emb, dtype=np.float32)

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    idx = 0

    def flush() -> None:
        nonlocal buf, buf_len, idx
        if not buf:
            return
        chunk_text = " ".join(buf).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, index=idx))
            idx += 1
        buf = []
        buf_len = 0

    for i, sent in enumerate(sents):
        sent_len = len(sent)
        if buf_len + sent_len + 1 > settings.chunk_max_chars and buf_len >= settings.chunk_min_chars:
            flush()

        if buf:
            sim = _cosine(emb[i - 1], emb[i])
            if sim < settings.chunk_similarity_threshold and buf_len >= settings.chunk_min_chars:
                flush()

        buf.append(sent)
        buf_len += sent_len + 1

    flush()
    return chunks

