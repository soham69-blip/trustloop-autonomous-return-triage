"""
Tests for TrustLoop Concurrency, Persistence, Thread Safety, and Atomic Writes.
"""

import unittest
import threading
import tempfile
import json
from pathlib import Path

from backend.app.core.persistence import (
    locked_append_jsonl,
    locked_read_jsonl,
    atomic_write_json,
    atomic_read_json,
)


class TestPersistenceAndConcurrency(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_locked_append_and_read(self):
        test_file = self.temp_dir / "test_records.jsonl"
        data1 = {"id": 1, "value": "first"}
        data2 = {"id": 2, "value": "second"}

        locked_append_jsonl(test_file, data1)
        locked_append_jsonl(test_file, data2)

        records = locked_read_jsonl(test_file)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["value"], "first")
        self.assertEqual(records[1]["value"], "second")

    def test_concurrent_multithreaded_appends(self):
        test_file = self.temp_dir / "concurrent_log.jsonl"
        num_threads = 10
        writes_per_thread = 20

        def worker(thread_idx: int):
            for i in range(writes_per_thread):
                locked_append_jsonl(test_file, {
                    "thread_idx": thread_idx,
                    "write_idx": i,
                    "msg": f"data from thread {thread_idx} write {i}",
                })

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = locked_read_jsonl(test_file)
        # Verify exact total count with zero corrupted/interleaved lines
        self.assertEqual(len(records), num_threads * writes_per_thread)

    def test_atomic_write_and_read_json(self):
        json_file = self.temp_dir / "state.json"
        payload = {"version": "v1.3.0", "active": True, "count": 42}

        atomic_write_json(json_file, payload)
        loaded = atomic_read_json(json_file)
        self.assertEqual(loaded, payload)

    def test_bounded_read_limits_memory(self):
        test_file = self.temp_dir / "large_log.jsonl"
        for i in range(100):
            locked_append_jsonl(test_file, {"idx": i})

        # Read with limit 10
        records = locked_read_jsonl(test_file, limit=10)
        self.assertEqual(len(records), 10)
        self.assertEqual(records[-1]["idx"], 99)


if __name__ == "__main__":
    unittest.main()
