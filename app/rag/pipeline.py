from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
from openai import OpenAI

from app.config import settings
from app.guardrails.guards import get_input_guard, get_output_guard
from app.models.medcpt import get_medcpt_cross_encoder, get_medcpt_query_encoder
from app.rag.intent_index import get_domains_for_titles, get_or_build_intent_index
from app.rag.prompts import (
    CITATION_TITLES_INSTRUCTION,
    CONTEXT_PROMPT_PREFIX,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from app.vectorstore.chroma_store import get_collection

logger = logging.getLogger(__name__)

# Safe fallback when output guardrails fail
_GUARDRAILS_OUTPUT_FALLBACK = "The provided documents do not contain information about this topic."
# Message when input guardrails reject the question
_GUARDRAILS_INPUT_REJECT = (
    "Your question could not be processed. Please rephrase or avoid including sensitive or off-topic content."
)


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


# Phrases indicating the response says "no info in provided documents" -> used_citations must be empty
_NO_DOC_INFO_PHRASES = (
    "do not contain",
    "does not contain",
    "don't contain",
    "no information",
    "not found in",
    "cannot find",
    "could not find",
    "not in the provided",
    "not in these documents",
    "not in the documents",
    "not contain information",
    "does not have information",
    "no relevant",
    "no applicable",
    "not mentioned in",
    "not present in",
    "not defined in the provided",
    "not defined in the documents",
    "not defined in",
    "does not appear to be",
    "does not appear in",
    "not appear in the",
    "is not defined in the provided",
    "is not in the provided",
    "not a term related",  # e.g. "does not appear to be a term related to..."
)


def _response_indicates_no_document_info(response: str) -> bool:
    """True if the response states that the provided documents do not contain the answer."""
    if not (response or "").strip():
        return False
    r = response.strip().lower()
    return any(p in r for p in _NO_DOC_INFO_PHRASES)


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


def _rewrite_query(q: str, intent_description: str) -> str:
    return q + " in the context of " + intent_description


# Definition/factual question patterns: skip disambiguation and proceed to RAG.
_DEFINITION_PREFIXES = (
    "what is ",
    "what are ",
    "define ",
    "explain ",
    "overview of ",
    "definition of ",
    "meaning of ",
    "describe ",
)


def _is_definition_query(query: str) -> bool:
    """If True, query is definition-style (may still trigger Gate 4 underspecification)."""
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(q.startswith(p) or (" " + p in " " + q) for p in _DEFINITION_PREFIXES)


def _extract_head_concept(query: str) -> str:
    """
    Extract the key term from a definition-style query (Gate 4).
    E.g. 'What is an amplicon?' -> 'amplicon'.
    """
    q = (query or "").strip()
    if not q:
        return ""
    q_lower = q.lower()
    for p in _DEFINITION_PREFIXES:
        if q_lower.startswith(p):
            q = q[len(p) :].strip()
            break
        if " " + p in " " + q_lower:
            i = q_lower.index(" " + p) + len(" " + p)
            q = q[i:].strip()
            break
    # Strip leading article
    for article in ("a ", "an ", "the "):
        if q.lower().startswith(article):
            q = q[len(article) :].strip()
            break
    q = q.rstrip("?").strip()
    return q or "it"


def _short_content_snippet(content: str, max_chars: int = 120, fallback_title: str = "") -> str:
    """Build a short user-friendly snippet from chunk content; fallback to truncated title."""
    try:
        max_chars = int(max_chars) if max_chars is not None else 120
    except (TypeError, ValueError):
        max_chars = 120
    max_chars = max(1, min(max_chars, 2000))
    text = (content or "").strip() if isinstance(content, str) else ""
    if text:
        if len(text) <= max_chars:
            return text
        cut = text[: max_chars + 1].rsplit(maxsplit=1)
        snippet = (cut[0] if cut else text[:max_chars]).strip()
        return snippet + "..." if len(snippet) < len(text) else snippet
    title = (fallback_title or "").strip() if isinstance(fallback_title, str) else ""
    if not title:
        return "No description"
    return (title[: max_chars] + "...") if len(title) > max_chars else title


def _short_label(content: str, fallback_title: str = "", max_words: int = 5) -> str:
    """First max_words words from content or title, for easy display (e.g. 4-5 words)."""
    try:
        max_words = int(max_words) if max_words is not None else 5
    except (TypeError, ValueError):
        max_words = 5
    max_words = max(1, min(max_words, 20))
    text = (content or "").strip() if isinstance(content, str) else ""
    text = text or ((fallback_title or "").strip() if isinstance(fallback_title, str) else "")
    if not text:
        return "No description"
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def _summarize_chunk_for_label(content: str, title: str = "", max_words: int = 5) -> str:
    """
    Ask the LLM for a 4-5 word summary of the chunk for clarification display.
    Falls back to _short_label on any failure or empty response.
    """
    try:
        max_words = int(max_words) if max_words is not None else 5
    except (TypeError, ValueError):
        max_words = 5
    max_words = max(1, min(max_words, 20))
    text = (content or "").strip() if isinstance(content, str) else ""
    title_str = (title or "").strip() if isinstance(title, str) else ""
    if not text and title_str:
        return _short_label("", title_str, max_words)
    if not text:
        return "No description"
    # Limit input to avoid long prompts and latency
    max_input_chars = 400
    if len(text) > max_input_chars:
        text = text[: max_input_chars].rsplit(maxsplit=1)[0] + "..."
    title_part = f"Title: {title_str}\n\n" if title_str else ""
    prompt = (
        "Summarize the following in exactly 4 to 5 words, for a user choosing a search context. "
        "Output only that phrase, nothing else. No quotes or punctuation.\n\n"
        f"{title_part}Content:\n{text}"
    )
    try:
        api_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key or "ollama"
        base_url = settings.openai_base_url or "http://localhost:11434/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)
        model = getattr(settings, "openai_model", None) or os.environ.get("OPENAI_MODEL", "phi3:latest")
        if model.startswith("ollama/"):
            model = model.replace("ollama/", "", 1)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return _short_label(content, title_str, max_words)
        words = raw.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return " ".join(words) if words else _short_label(content, title_str, max_words)
    except Exception:
        return _short_label(content, title_str, max_words)


def _run_rag(query: str, top_k: int) -> dict[str, Any]:
    """
    Run retrieval, rerank, LLM, and citation fix. Returns { response, used_citations }.
    Used both for direct RAG and after query rewrite in disambiguation.
    """
    qenc = get_medcpt_query_encoder()
    q_emb = qenc.encode_texts([query], batch_size=1)[0]

    col = get_collection()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    if not docs:
        context = {}
        return _call_llm(user_question=query, context=context, citation_titles=[])

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
    scores = xenc.score_pairs([(query, c["content"]) for c in candidates], batch_size=8)
    for c, s in zip(candidates, scores, strict=False):
        c["relevance_score"] = float(s)

    candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    context = _build_context_docs(chunks=candidates)
    citation_titles = list(dict.fromkeys(c["title"] for c in candidates if c.get("title")))

    result = _call_llm(user_question=query, context=context, citation_titles=citation_titles)
    result = _fix_used_citations(result, citation_titles)
    if _response_indicates_no_document_info(result.get("response", "")):
        result["used_citations"] = None  # null in JSON when response says no info in documents
    # Fallback: empty response + empty citations -> use generic message (never echo query to avoid PII leak)
    if not (result.get("response") or "").strip() and not result.get("used_citations"):
        result["response"] = _GUARDRAILS_OUTPUT_FALLBACK
        result["used_citations"] = None

    # --- Output guardrails ---
    output_guard = get_output_guard()
    if output_guard is not None:
        response_text = (result.get("response") or "").strip()
        try:
            parsed = output_guard.parse(llm_output=response_text, num_reasks=0)
            if not getattr(parsed, "validation_passed", True):
                logger.info("Guardrails output validation failed; returning fallback.")
                result["response"] = _GUARDRAILS_OUTPUT_FALLBACK
                result["used_citations"] = []
        except Exception as e:
            logger.warning("Guardrails output validation error: %s", e)
            result["response"] = _GUARDRAILS_OUTPUT_FALLBACK
            result["used_citations"] = []

    return result


def _retrieve_top_k_chunks(query: str, top_k: int) -> list[dict[str, Any]]:
    """Retrieve top-K chunks with title/metadata for Gate 4 domain inspection."""
    qenc = get_medcpt_query_encoder()
    q_emb = qenc.encode_texts([query], batch_size=1)[0]
    col = get_collection()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    chunks = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        meta = meta or {}
        chunks.append({
            "content": docs[i],
            "title": meta.get("title", ""),
            "pmid": meta.get("pmid", ""),
            "distance": float(dists[i]) if i < len(dists) else 0.0,
        })
    return chunks


def answer_question(user_question: str, *, top_k: int) -> dict[str, Any]:
    # --- Input guardrails (optional) ---
    input_guard = get_input_guard()
    if input_guard is not None:
        try:
            parsed = input_guard.parse(llm_output=user_question, num_reasks=0)
            if not getattr(parsed, "validation_passed", True):
                return {
                    "response": _GUARDRAILS_INPUT_REJECT,
                    "used_citations": [],
                }
        except Exception as e:
            logger.warning("Guardrails input validation error: %s", e)
            return {
                "response": _GUARDRAILS_INPUT_REJECT,
                "used_citations": [],
            }

    # --- Retrieval ---
    qenc = get_medcpt_query_encoder()
    q_emb = qenc.encode_texts([user_question], batch_size=1)[0]
    col = get_collection()

    res = col.query(
        query_embeddings=[q_emb],
        n_results=max(2, top_k),
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    if not docs or not dists:
        return {
            "response": "",
            "used_citations": [],
        }

    # Run RAG (no disambiguation or confidence gating)
    result = _run_rag(user_question, top_k)
    return result



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

    stream = client.chat.completions.create(
        model="phi3:latest",  # ✅ NO prefix
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
        stream=True,
    )

    text_parts: list[str] = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            text_parts.append(chunk.choices[0].delta.content)
    text = "".join(text_parts) or "{}"
    return _parse_llm_json(text)


