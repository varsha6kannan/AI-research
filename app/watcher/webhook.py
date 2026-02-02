from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field


class FileEventType(str, Enum):
    created = "created"
    modified = "modified"
    deleted = "deleted"


class WebhookPayload(BaseModel):
    file_name: str
    file_path: str
    last_modified_ts: str = Field(
        description="ISO-8601 UTC timestamp, or empty string for delete when unknown"
    )
    file_type: str = Field(description="Lowercase file extension without dot, or empty")
    event_type: FileEventType


def build_payload(path: str | Path, event_type: FileEventType) -> WebhookPayload:
    p = Path(path)
    file_type = p.suffix.lower().lstrip(".")
    file_name = p.name
    file_path = str(p.resolve())

    if event_type == FileEventType.deleted:
        ts = ""
    else:
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            ts = mtime.isoformat()
        except FileNotFoundError:
            ts = ""

    return WebhookPayload(
        file_name=file_name,
        file_path=file_path,
        last_modified_ts=ts,
        file_type=file_type,
        event_type=event_type,
    )


async def post_webhook_payload(webhook_url: str, payload: WebhookPayload) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(webhook_url, json=payload.model_dump())

