from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from app.config import settings
from app.models.medcpt import get_medcpt_cross_encoder, get_medcpt_query_encoder
from app.rag.prompts import (
    CITATION_TITLES_INSTRUCTION,
    CONTEXT_PROMPT_PREFIX,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from app.vectorstore.chroma_store import get_collection


def _parse_llm_json(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    if "response" not in obj or "used_citations" not in obj:
        raise ValueError("LLM output missing required keys: response, used_citations")
    used = obj["used_citations"]
    if not isinstance(used, list):
        raise ValueError("used_citations must be a list")
    for i, item in enumerate(used):
        if not isinstance(item, str):
            raise ValueError(f"used_citations[{i}] must be a string (document title)")
    return obj


def _fix_used_citations(
    obj: dict[str, Any],
    citation_titles: list[str],
) -> dict[str, Any]:
    """Replace invalid citation strings with actual document titles by position."""
    if not citation_titles:
        return obj
    allowed = set(citation_titles)
    used = obj.get("used_citations") or []
    cleaned: list[str] = []
    for i, cit in enumerate(used):
        if cit in allowed:
            cleaned.append(cit)
        elif i < len(citation_titles):
            cleaned.append(citation_titles[i])
    obj = {**obj, "used_citations": cleaned}
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
        return _call_llm(user_question=user_question, context=context, citation_titles=[])

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

    # 4) Build context JSON and extract titles for citation placeholder
    context = _build_context_docs(chunks=candidates)
    citation_titles = list(dict.fromkeys(c["title"] for c in candidates if c.get("title")))

    # 5) LLM call (JSON-only), then fix any invalid citation strings
    result = _call_llm(user_question=user_question, context=context, citation_titles=citation_titles)
    return _fix_used_citations(result, citation_titles)


def _call_llm(
    *,
    user_question: str,
    context: dict[str, Any],
    citation_titles: list[str] | None = None,
) -> dict[str, Any]:
    client = OpenAI(
        api_key="ollama",  # dummy value
        base_url="http://localhost:11434/v1",
    )

    citation_titles = citation_titles or []
    citation_instruction = (
        CITATION_TITLES_INSTRUCTION.format(
            citation_titles=json.dumps(citation_titles, ensure_ascii=False),
        )
        if citation_titles
        else ""
    )
    context_content = (
        CONTEXT_PROMPT_PREFIX
        + ("\n" + citation_instruction + "\n\n" if citation_instruction else "")
        + json.dumps(context, ensure_ascii=False, indent=2)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(user_question=user_question),
        },
        {"role": "user", "content": context_content},
    ]

    resp = client.chat.completions.create(
        model="phi3:latest",  # ✅ NO prefix
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )

    text = resp.choices[0].message.content or "{}"
    return _parse_llm_json(text)


