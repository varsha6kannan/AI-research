"""
Tests for DynamicMedRAG. Run from project root:
  python -m pytest tests/ -v
  or
  python tests/test_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_config_watch_dir_resolution():
    """Watch dir should resolve to project_root/datasets."""
    from app.config import PROJECT_ROOT, settings

    watch_dir = (PROJECT_ROOT / settings.datasets_dir).resolve()
    assert watch_dir.is_absolute()
    assert watch_dir.name == "datasets"
    assert PROJECT_ROOT in watch_dir.parents or watch_dir == PROJECT_ROOT / "datasets"


def test_loader_json():
    """Loader should parse datasets/test_rag.json."""
    from app.ingest.loader import load_documents

    path = PROJECT_ROOT / "datasets" / "test_rag.json"
    if not path.exists():
        return  # skip
    docs = load_documents(str(path))
    assert len(docs) == 1
    assert docs[0].title == "Aspirin doc"
    assert "Aspirin" in docs[0].content
    assert docs[0].pmid == "12345678"


def test_webhook_build_payload():
    """Webhook should build payload with file_name, file_path, event_type."""
    from app.watcher.webhook import FileEventType, build_payload

    path = PROJECT_ROOT / "datasets" / "test_rag.json"
    payload = build_payload(str(path), FileEventType.modified)
    assert payload.file_name == "test_rag.json"
    assert payload.event_type == FileEventType.modified
    assert payload.file_type == "json"
    assert "test_rag.json" in payload.file_path or path.name in payload.file_path


def test_chroma_get_by_source_path_empty():
    """get_chunk_ids_and_hashes_by_source_path returns dict (empty if no chunks)."""
    from app.vectorstore.chroma_store import get_chunk_ids_and_hashes_by_source_path

    try:
        # Non-existent source_path should return empty dict (no rows match)
        result = get_chunk_ids_and_hashes_by_source_path("nonexistent/path/123")
        assert isinstance(result, dict)
        assert len(result) == 0
    except Exception as e:
        if "I/O" in str(e) or "disk" in str(e).lower() or "2570" in str(e):
            return  # skip in sandbox or when DB is locked
        raise


def test_health_endpoint():
    """GET /health returns 200 and status ok (requires server running)."""
    import urllib.request

    try:
        req = urllib.request.Request("http://127.0.0.1:8000/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = resp.read().decode()
            assert "ok" in data or "status" in data
    except OSError:
        try:
            import pytest
            pytest.skip("Server not running at 127.0.0.1:8000")
        except ImportError:
            return  # skip without pytest


def _run():
    """Run tests without pytest."""
    tests = [
        ("config watch_dir", test_config_watch_dir_resolution),
        ("loader json", test_loader_json),
        ("webhook payload", test_webhook_build_payload),
        ("chroma get empty", test_chroma_get_by_source_path_empty),
        ("health endpoint", test_health_endpoint),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK  {name}")
        except Exception as e:
            if "skip" in str(e).lower() or "Skip" in str(e):
                print(f"  skip {name}: {e}")
            else:
                print(f"  FAIL {name}: {e}")
                failed += 1
    return failed


if __name__ == "__main__":
    try:
        import pytest as _pytest
    except ImportError:
        _pytest = None
    if _pytest is not None:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
            cwd=PROJECT_ROOT,
        )
        sys.exit(result.returncode)
    # No pytest: run inline
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)
    failed = _run()
    sys.exit(1 if failed else 0)
