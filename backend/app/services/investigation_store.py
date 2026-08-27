"""Durable JSONL storage for live investigation summaries and trace provenance."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


STORE_PATH = Path("data/investigations/history.jsonl")


def store_investigation(result: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "case_id": result.get("case_id"),
        "investigation": result.get("investigation"),
        "decision": result.get("decision"),
        "score_fusion": result.get("score_fusion"),
        "responsibility": result.get("responsibility"),
        "vision_analysis": result.get("vision_analysis"),
        "timestamp": result.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "agents": result.get("agents", []),
        "communications": result.get("communications", []),
    }
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return record


def list_investigations(case_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not STORE_PATH.exists():
        return []
    records: List[Dict[str, Any]] = []
    with STORE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("case_id", "")).upper() == case_id.upper():
                records.append(record)
    return records[-max(1, min(limit, 100)) :][::-1]
