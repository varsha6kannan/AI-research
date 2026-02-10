"""
Gradio UI for DynamicMedRAG. Calls the FastAPI backend at POST /query.
Run the backend first: uvicorn app.api.main:app --reload
Then run this script: python run_gradio.py
"""
from __future__ import annotations

import os
from typing import Any

import gradio as gr
import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
QUERY_URL = f"{BACKEND_URL.rstrip('/')}/query"


def query_backend(question: str) -> dict[str, Any] | None:
    """POST /query with user_question. Returns JSON or None on error."""
    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                QUERY_URL,
                json={"user_question": question, "top_k": None},
            )
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        return {"_error": str(e)}
    except Exception as e:
        return {"_error": str(e)}


def format_citations(citations: list[str] | None) -> str:
    if not citations:
        return "No citations"
    return "\n".join(f"- {c}" for c in citations)


_NO_INFO_SENTINEL = "The provided documents do not contain information about this topic."
_NEUTRAL_NO_ANSWER = "No answer could be generated from the provided documents."


def _is_no_answer_response(response: str) -> bool:
    r = (response or "").strip()
    if not r:
        return True
    if r == _NO_INFO_SENTINEL:
        return True
    # Backend may return variants like: "The provided documents do not contain information about 'X'."
    if r.lower().startswith("the provided documents do not contain information"):
        return True
    return False


def on_ask(question: str) -> tuple[str, str]:
    """Handle Ask button: call API and treat response as final (no clarification flow)."""
    if not (question or "").strip():
        return ("Please enter a question.", "No citations")

    q = question.strip()
    data = query_backend(q)
    if not data:
        return (f"Could not reach backend at {QUERY_URL}. Is it running?", "No citations")
    if data.get("_error"):
        return (f"Error: {data['_error']}", "No citations")

    response = (data.get("response") or "").strip()
    citations = data.get("used_citations")
    citations_list = citations if isinstance(citations, list) else []
    citations_text = format_citations(citations_list)

    if _is_no_answer_response(response):
        return (_NEUTRAL_NO_ANSWER, citations_text)

    return (response, citations_text)


# Light medical theme: white/light pastel, calm colors, good contrast
MEDICAL_CSS = """
body, #root, .gradio-container { background: #dbeafe !important; }
.gradio-container { max-width: 100%; padding: 1rem 1rem; }
.main-heading { text-align: center; font-family: system-ui, -apple-system, sans-serif; font-size: 1.75rem; font-weight: 600; color: #1a365d; margin-bottom: 1.5rem; }
.citations-heading, .citations-heading p, .citations-heading strong { color: #1a365d !important; }
.contained { max-width: 800px; margin: 0 auto; }
.primary-btn .primary { background: #4a90a4 !important; color: #fff !important; border-radius: 6px; }
.input-box textarea, .answer-wrap textarea, .citations-wrap textarea { border-radius: 8px; background: #ffffff !important; color: #1e293b !important; }
.answer-wrap, .citations-wrap { padding: 2px !important; }
.answer-wrap .container, .citations-wrap .container { padding: 4px !important; min-height: 0 !important; }
"""


def main() -> None:
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue), css=MEDICAL_CSS, title="Medical RAG") as app:
        gr.HTML('<h1 class="main-heading">Medical RAG 🩺</h1>')
        with gr.Row(elem_classes=["contained"]):
            with gr.Column(scale=1, min_width=320):
                question_box = gr.Textbox(
                    label="Ask your query here",
                    placeholder="Enter your medical question",
                    lines=3,
                    max_lines=6,
                    show_label=False,
                    container=False,
                    elem_classes=["input-box"],
                )
                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary", size="sm", elem_classes=["primary-btn"])
                    refresh_btn = gr.Button("Refresh", size="sm")
                gr.Markdown("**Answer**", elem_classes=["citations-heading"])
                answer_out = gr.Textbox(
                    label="",
                    value="",
                    lines=10,
                    max_lines=24,
                    interactive=False,
                    show_label=False,
                    elem_classes=["answer-wrap"],
                )
                gr.Markdown("**Citations**", elem_classes=["citations-heading"])
                citations_out = gr.Textbox(
                    label="",
                    value="No citations",
                    lines=4,
                    max_lines=8,
                    interactive=False,
                    show_label=False,
                    elem_classes=["citations-wrap"],
                )
        def on_refresh():
            return ("", "", "No citations")

        ask_btn.click(
            fn=on_ask,
            inputs=[question_box],
            outputs=[answer_out, citations_out],
        )
        refresh_btn.click(
            fn=on_refresh,
            inputs=[],
            outputs=[question_box, answer_out, citations_out],
        )

    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
