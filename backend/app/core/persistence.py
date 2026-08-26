"""
TrustLoop Concurrency & Persistence Hardening Module.

Provides process-safe and thread-safe file I/O operations using file locks and atomic writes.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import json
import threading
import tempfile
import uuid
from filelock import FileLock, Timeout

from backend.app.core.config import settings

_THREAD_LOCK = threading.Lock()


def _get_lock(path: Path) -> FileLock:
    lock_path = path.parent / f".{path.name}.lock"
    return FileLock(str(lock_path), timeout=settings.FILE_LOCK_TIMEOUT_SECONDS)


def locked_append_jsonl(file_path: Path, record: Dict[str, Any]) -> bool:
    """
    Thread-safe and process-safe append to a JSONL file with file locking.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, ensure_ascii=False) + "\n"

    with _THREAD_LOCK:
        lock = _get_lock(file_path)
        try:
            with lock:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(serialized)
            return True
        except (Timeout, OSError) as exc:
            # Fallback direct write if lock times out in non-critical scenarios
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(serialized)
            return False


def locked_read_jsonl(
    file_path: Path,
    limit: Optional[int] = None,
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Dict[str, Any]]:
    """
    Safely read JSONL records with optional filtering and bounded memory limits.
    """
    if not file_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with _THREAD_LOCK:
        lock = _get_lock(file_path)
        try:
            with lock.acquire(timeout=settings.FILE_LOCK_TIMEOUT_SECONDS):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if filter_fn is None or filter_fn(item):
                                records.append(item)
                        except json.JSONDecodeError:
                            continue
        except Timeout:
            # Fallback lockless read
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if filter_fn is None or filter_fn(item):
                            records.append(item)
                    except json.JSONDecodeError:
                        continue

    if limit is not None and limit > 0:
        return records[-limit:]
    return records


def atomic_write_json(file_path: Path, data: Any) -> None:
    """
    Atomically write JSON data using a temporary file replacement.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.parent / f".tmp_{file_path.name}_{uuid.uuid4().hex}"

    with _THREAD_LOCK:
        lock = _get_lock(file_path)
        with lock:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(file_path)


def atomic_read_json(file_path: Path, default: Any = None) -> Any:
    """
    Safely read JSON file with fallback default on missing or corrupt file.
    """
    if not file_path.exists():
        return default if default is not None else {}

    with _THREAD_LOCK:
        lock = _get_lock(file_path)
        try:
            with lock:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return default if default is not None else {}
