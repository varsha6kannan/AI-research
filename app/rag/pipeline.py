from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models.medcpt import get_medcpt_cross_encoder, get_medcpt_query_encoder
from app.rag.prompts import CONTEXT_PROMPT_PREFIX, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.vectorstore.chroma_store import get_collection


def _parse_llm_json(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    if "response" not in obj or "used_pmids" not in obj:
        raise ValueError("LLM output missing required keys: response, used_pmids")
    if not isinstance(obj["used_pmids"], list):
        raise ValueError("used_pmids must be a list")
    return obj


def _build_context_docs(
    *,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    for i, c in enumerate(chunks, start=1):
        ctx[f"doc{i}"] = {
            "title": c.get("title", ""),
            "content": c.get("content", ""),
            "pmid": c.get("pmid", ""),
            "relevance_score": str(c.get("relevance_score", "")),
        }
    return ctx


def answer_question(user_question: str, *, top_k: int) -> dict[str, Any]:
    # 1) Embed query
    qenc = get_medcpt_query_encoder()
    q_emb = qenc.encode_texts([user_question], batch_size=1)[0]

    # 2) Retrieve top-K from Chroma
    col = get_collection()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        # Some Chroma versions only support these keys in `include`.
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    if not docs:
        # Still call LLM with empty context; it should respond grounded (likely cannot answer).
        context = {}
        return _call_llm(user_question=user_question, context=context)

    # 3) Re-rank top-3 with cross encoder
    top_n = min(settings.rerank_top_n, len(docs))
    candidates = []
    for i in range(top_n):
        meta = metas[i] or {}
        candidates.append(
            {
                "content": docs[i],
                "pmid": meta.get("pmid", ""),
                "title": meta.get("title", ""),
                "distance": float(dists[i]) if dists else 0.0,
            }
        )

    xenc = get_medcpt_cross_encoder()
    scores = xenc.score_pairs([(user_question, c["content"]) for c in candidates], batch_size=8)
    for c, s in zip(candidates, scores, strict=False):
        c["relevance_score"] = float(s)

    candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    # 4) Build context JSON
    context = _build_context_docs(chunks=candidates)

    # 5) LLM call (JSON-only)
    return _call_llm(user_question=user_question, context=context)


def _call_llm(*, user_question: str, context: dict[str, Any]) -> dict[str, Any]:
    client = OpenAI(
        api_key="ollama",  # dummy value
        base_url="http://localhost:11434/v1",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(user_question=user_question),
        },
        {
            "role": "user",
            "content": CONTEXT_PROMPT_PREFIX
            + "\n"
            + json.dumps(context, ensure_ascii=False, indent=2),
        },
    ]

    resp = client.chat.completions.create(
        model="phi3:latest",  # ✅ NO prefix
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )

    text = resp.choices[0].message.content or "{}"
    return _parse_llm_json(text)


