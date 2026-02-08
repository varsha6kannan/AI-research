from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import PROJECT_ROOT, settings
from app.watcher.file_watcher import FileWatcher, WatcherConfig
from app.watcher.webhook import WebhookPayload, post_webhook_payload


class QueryRequest(BaseModel):
    user_question: str = Field(min_length=1)
    top_k: int | None = None


class IngestRequest(BaseModel):
    # If omitted, ingest all supported files in datasets_dir
    file_path: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    ingest_queue: asyncio.Queue[WebhookPayload] = asyncio.Queue()

    def on_payload(payload: WebhookPayload) -> None:
        ingest_queue.put_nowait(payload)

    # Use project-root-relative path so we watch the correct directory regardless of process cwd
    watch_dir = (PROJECT_ROOT / settings.datasets_dir).resolve()
    watcher = FileWatcher(WatcherConfig(watch_dir=watch_dir), loop)
    watcher.start(on_payload=on_payload)

    app.state.ingest_queue = ingest_queue
    app.state.watcher = watcher
    app.state.ingest_task = None

    async def ingest_worker():
        # Lazy imports to avoid slow startup import time in tooling.
        from app.ingest.pipeline import handle_payload

        while True:
            payload = await ingest_queue.get()
            try:
                if settings.webhook_url:
                    await post_webhook_payload(settings.webhook_url, payload)
                await asyncio.to_thread(handle_payload, payload)
            except Exception:
                # Best-effort worker; keep running.
                pass
            finally:
                ingest_queue.task_done()

    app.state.ingest_task = asyncio.create_task(ingest_worker())

    try:
        yield
    finally:
        if app.state.ingest_task:
            app.state.ingest_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await app.state.ingest_task
        watcher.stop()


app = FastAPI(title="DynamicMedRAG", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "datasets_dir": str(Path(settings.datasets_dir).resolve()),
        "chroma_persist_dir": str(Path(settings.chroma_persist_dir).resolve()),
        "collection": settings.chroma_collection,
    }


@app.post("/query")
async def query(req: QueryRequest) -> Any:
    from app.rag.pipeline import answer_question

    top_k = req.top_k or settings.top_k
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    try:
        return await asyncio.to_thread(answer_question, req.user_question, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, Any]:
    """
    Trigger manual ingestion for a single file or for the whole datasets directory.
    This is helpful when starting with an existing dataset folder.
    """
    from app.ingest.pipeline import handle_payload
    from app.watcher.webhook import FileEventType, build_payload

    if req.file_path:
        payload = build_payload(req.file_path, FileEventType.modified)
        await asyncio.to_thread(handle_payload, payload)
        return {"status": "ok", "ingested": [payload.file_path]}

    datasets_dir = Path(settings.datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    ingested: list[str] = []
    for p in datasets_dir.rglob("*"):
        if not p.is_file():
            continue
        payload = build_payload(str(p), FileEventType.modified)
        # Ignore unknown file types in pipeline; it is safe to call.
        await asyncio.to_thread(handle_payload, payload)
        ingested.append(str(p.resolve()))
    return {"status": "ok", "ingested": ingested}

