from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import json
import uuid
import shutil
from datetime import datetime, timezone
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BASE_DATASET = PROJECT_ROOT / "data" / "processed" / "trustloop" / "model_ready.csv"
VERIFIED_FEEDBACK_FILE = PROJECT_ROOT / "data" / "feedback" / "verified_feedback.jsonl"
QUARANTINED_FEEDBACK_FILE = PROJECT_ROOT / "data" / "feedback" / "quarantine_feedback.jsonl"

SNAPSHOT_INDEX_FILE = SNAPSHOTS_DIR / "snapshots_manifest.json"


class SnapshotMetadata(BaseModel):
    snapshot_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_version: str
    source_paths: List[str]
    dataset_sha256: str
    row_count: int
    column_list: List[str]
    feature_configuration: Dict[str, Any]
    target_distribution: Dict[str, int]
    feedback_count: int
    eligible_feedback_count: int
    quarantined_feedback_count: int
    model_training_config: Dict[str, Any]
    snapshot_filename: str


def _calculate_file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest() -> Dict[str, Dict[str, Any]]:
    from backend.app.core.persistence import atomic_read_json
    return atomic_read_json(SNAPSHOT_INDEX_FILE, default={})


def _save_manifest(manifest: Dict[str, Dict[str, Any]]) -> None:
    from backend.app.core.persistence import atomic_write_json
    atomic_write_json(SNAPSHOT_INDEX_FILE, manifest)


def create_training_snapshot(
    base_dataset_path: Optional[Path] = None,
    include_verified_feedback: bool = True,
    training_config: Optional[Dict[str, Any]] = None,
    custom_snapshot_id: Optional[str] = None,
    dataset_version: str = "v1.0",
) -> SnapshotMetadata:
    """
    Create an immutable, versioned snapshot of the exact dataset and metadata used for training.
    """
    base_path = base_dataset_path or DEFAULT_BASE_DATASET
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found at {base_path}")

    # 1. Read base dataset
    df = pd.read_csv(base_path, low_memory=False)
    sources = [str(base_path.relative_to(PROJECT_ROOT))]

    eligible_feedback_count = 0
    quarantined_feedback_count = 0
    total_feedback_count = 0

    # 2. Append eligible verified feedback if requested
    if include_verified_feedback and VERIFIED_FEEDBACK_FILE.exists():
        sources.append(str(VERIFIED_FEEDBACK_FILE.relative_to(PROJECT_ROOT)))
        feedback_rows = []
        with open(VERIFIED_FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_feedback_count += 1
                try:
                    record = json.loads(line)
                    if record.get("training_eligible", False):
                        eligible_feedback_count += 1
                        raw_feat = dict(record.get("raw_features", {}))
                        label_str = record.get("human_verified_label")
                        label_map = {
                            "Legitimate": 0,
                            "Policy Abuser": 1,
                            "Fraudulent Return": 2,
                            "Wardrobing": 3,
                        }
                        if label_str in label_map:
                            raw_feat["abuse_label"] = label_map[label_str]
                            feedback_rows.append(raw_feat)
                except Exception:
                    continue

        if QUARANTINED_FEEDBACK_FILE.exists():
            with open(QUARANTINED_FEEDBACK_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        quarantined_feedback_count += 1
                        total_feedback_count += 1

        if feedback_rows:
            feedback_df = pd.DataFrame(feedback_rows)
            # Align common columns
            common_cols = [c for c in df.columns if c in feedback_df.columns]
            if common_cols:
                df = pd.concat([df, feedback_df[common_cols]], ignore_index=True)

    # 3. Generate snapshot identifier
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    hash_bytes = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    content_hash = hashlib.sha256(hash_bytes).hexdigest()[:8]
    snapshot_id = custom_snapshot_id or f"snap_{timestamp_str}_{content_hash}"

    manifest = _load_manifest()
    if snapshot_id in manifest or (SNAPSHOTS_DIR / f"{snapshot_id}.csv").exists() or (SNAPSHOTS_DIR / f"{snapshot_id}.json").exists():
        raise ValueError(f"Snapshot '{snapshot_id}' already exists. Overwrites are disallowed.")

    # 4. Save immutable CSV snapshot
    snapshot_filename = f"{snapshot_id}.csv"
    snapshot_path = SNAPSHOTS_DIR / snapshot_filename
    df.to_csv(snapshot_path, index=False)

    # 5. Compute exact artifact hash
    file_sha256 = _calculate_file_sha256(snapshot_path)

    # 6. Target and feature distributions
    target_dist: Dict[str, int] = {}
    if "abuse_label" in df.columns:
        counts = df["abuse_label"].value_counts().to_dict()
        label_names = {0: "Legitimate", 1: "Policy Abuser", 2: "Fraudulent Return", 3: "Wardrobing"}
        target_dist = {label_names.get(k, str(k)): v for k, v in counts.items()}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "abuse_label"]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    feature_config = {
        "numeric_features": numeric_cols,
        "categorical_features": cat_cols,
        "target_column": "abuse_label" if "abuse_label" in df.columns else None,
    }


    model_config = training_config or {
        "model_type": "LightGBM",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "class_weight": "balanced",
        "random_state": 42,
    }

    metadata = SnapshotMetadata(
        snapshot_id=snapshot_id,
        dataset_version=dataset_version,
        source_paths=sources,
        dataset_sha256=file_sha256,
        row_count=len(df),
        column_list=list(df.columns),
        feature_configuration=feature_config,
        target_distribution=target_dist,
        feedback_count=total_feedback_count,
        eligible_feedback_count=eligible_feedback_count,
        quarantined_feedback_count=quarantined_feedback_count,
        model_training_config=model_config,
        snapshot_filename=snapshot_filename,
    )

    # 7. Persist to manifest
    manifest[snapshot_id] = metadata.model_dump()
    _save_manifest(manifest)

    # Also save snapshot metadata JSON sidecar
    sidecar_path = SNAPSHOTS_DIR / f"{snapshot_id}.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)

    return metadata


def get_snapshot_metadata(snapshot_id: str) -> Optional[SnapshotMetadata]:
    manifest = _load_manifest()
    data = manifest.get(snapshot_id)
    if not data:
        sidecar = SNAPSHOTS_DIR / f"{snapshot_id}.json"
        if sidecar.exists():
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return None
    if data:
        return SnapshotMetadata(**data)
    return None


def list_snapshots() -> List[SnapshotMetadata]:
    manifest = _load_manifest()
    results = []
    for s_id, data in manifest.items():
        try:
            results.append(SnapshotMetadata(**data))
        except Exception:
            continue
    return sorted(results, key=lambda s: s.created_at, reverse=True)


def verify_snapshot_integrity(snapshot_id: str) -> Tuple[bool, str]:
    meta = get_snapshot_metadata(snapshot_id)
    if not meta:
        return False, f"Snapshot '{snapshot_id}' not found in registry"

    snapshot_file = SNAPSHOTS_DIR / meta.snapshot_filename
    if not snapshot_file.exists():
        return False, f"Snapshot file '{meta.snapshot_filename}' is missing from disk"

    current_sha256 = _calculate_file_sha256(snapshot_file)
    if current_sha256 != meta.dataset_sha256:
        return False, f"SHA256 mismatch! Expected {meta.dataset_sha256}, got {current_sha256}"

    return True, "Snapshot integrity verified (SHA256 matches)"
