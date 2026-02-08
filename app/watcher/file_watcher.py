from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from watchdog.events import FileSystemEventHandler

from app.watcher.webhook import FileEventType, WebhookPayload, build_payload
from watchdog.observers.polling import PollingObserver


@dataclass(frozen=True)
class WatcherConfig:
    watch_dir: Path
    ignore_extensions: tuple[str, ...] = (".tmp", ".swp", ".part")
    debounce_ms: int = 400


class _DebouncedHandler(FileSystemEventHandler):
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        cfg: WatcherConfig,
        on_payload: Callable[[WebhookPayload], None],
    ) -> None:
        self._loop = loop
        self._cfg = cfg
        self._on_payload = on_payload
        self._pending: dict[tuple[str, FileEventType], None] = {}
        self._drain_handle: asyncio.TimerHandle | None = None

    def _should_ignore(self, path: str) -> bool:
        p = Path(path)
        if p.is_dir():
            return True
        if p.suffix.lower() in self._cfg.ignore_extensions:
            return True
        return False

    def _schedule(self, path: str, event_type: FileEventType) -> None:
        if self._should_ignore(path):
            return
        key = (path, event_type)
        self._pending[key] = None
        self._loop.call_soon_threadsafe(self._schedule_drain)

    def _schedule_drain(self) -> None:
        if self._drain_handle is not None:
            self._drain_handle.cancel()
        self._drain_handle = self._loop.call_later(
            self._cfg.debounce_ms / 1000.0,
            self._drain,
        )

    def _drain(self) -> None:
        self._drain_handle = None
        if not self._pending:
            return
        ready = list(self._pending.items())
        self._pending.clear()
        for (path, et), _ in ready:
            payload = build_payload(path, et)
            self._on_payload(payload)

    def on_created(self, event):  # type: ignore[override]
        if not event.is_directory:
            self._schedule(event.src_path, FileEventType.created)

    def on_modified(self, event):  # type: ignore[override]
        if not event.is_directory:
            self._schedule(event.src_path, FileEventType.modified)

    def on_deleted(self, event):  # type: ignore[override]
        if not event.is_directory:
            self._schedule(event.src_path, FileEventType.deleted)

    def on_moved(self, event): 
        if not event.is_directory:
            # Treat move/rename as a modification of the destination file
            self._schedule(event.dest_path, FileEventType.modified)


class FileWatcher:
    def __init__(self, cfg: WatcherConfig, loop: asyncio.AbstractEventLoop) -> None:
        self._cfg = cfg
        self._loop = loop
        self._observer = PollingObserver(timeout=1.0)
        self._started = False

    def start(self, on_payload: Callable[[WebhookPayload], None]) -> None:
        assert self._loop.is_running()
        self._cfg.watch_dir.mkdir(parents=True, exist_ok=True)
        handler = _DebouncedHandler(loop=self._loop, cfg=self._cfg, on_payload=on_payload)
        self._observer.schedule(handler, str(self._cfg.watch_dir), recursive=True)
        self._observer.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._observer.stop()
        self._observer.join(timeout=5.0)
        self._started = False

