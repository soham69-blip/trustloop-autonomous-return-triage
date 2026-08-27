"""
TrustLoop Centralized Configuration Management.

Provides strongly typed, environment-driven configuration across all TrustLoop services.
"""

from pathlib import Path
import os
from typing import Dict, Any
from pydantic import BaseModel, Field

# Root paths
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

# Load local .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        pass


class Settings(BaseModel):
    # Service Information
    SERVICE_NAME: str = "TrustLoop"
    API_VERSION: str = "v1"
    BACKEND_VERSION: str = "1.3.0"
    ENVIRONMENT: str = Field(default_factory=lambda: os.getenv("TRUSTLOOP_ENV", "development"))
    DEBUG: bool = Field(default_factory=lambda: os.getenv("TRUSTLOOP_DEBUG", "false").lower() in ("true", "1", "yes"))

    # Filesystem Paths
    ROOT_PATH: Path = PROJECT_ROOT
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    BACKUPS_DIR: Path = PROJECT_ROOT / "models" / "backups"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    FEEDBACK_DIR: Path = PROJECT_ROOT / "data" / "feedback"
    SNAPSHOTS_DIR: Path = PROJECT_ROOT / "data" / "snapshots"
    EVIDENCE_DIR: Path = PROJECT_ROOT / "data" / "evidence"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    RAG_INDEX_DIR: Path = PROJECT_ROOT / "rag" / "index"

    # Specific Model Artifact Paths
    PROD_MODEL_PATH: Path = PROJECT_ROOT / "models" / "lightgbm_model.pkl"
    PROD_BACKUP_PATH: Path = PROJECT_ROOT / "models" / "lightgbm_model_backup.pkl"
    CANDIDATE_MODEL_PATH: Path = PROJECT_ROOT / "models" / "lightgbm_candidate.pkl"
    CATEGORICAL_MAPPINGS_PATH: Path = PROJECT_ROOT / "models" / "categorical_mappings.pkl"

    # Reference Artifact SHA256 Hashes
    PROD_MODEL_SHA256: str = "db3a6c03149fa096c7df0d2df214043236f9cb8a8b0fe8ea198d1c24ea94d485"
    CANDIDATE_MODEL_SHA256: str = "6dec9ceeb10b2f9c7148b531bb1306e38344803250b162d0b5f0ff3ae0296f04"
    CATEGORICAL_MAPPINGS_SHA256: str = "432e9539c295367d700c3307dbb439a5e674f17432e84d2b3d52089c852997ad"

    # Security & Access Control
    AUTH_ENABLED: bool = Field(default_factory=lambda: os.getenv("TRUSTLOOP_AUTH_ENABLED", "false").lower() in ("true", "1", "yes"))
    API_KEY: str = Field(default_factory=lambda: os.getenv("TRUSTLOOP_API_KEY", "trustloop-operator-key-v1"))
    ADMIN_API_KEY: str = Field(default_factory=lambda: os.getenv("TRUSTLOOP_ADMIN_API_KEY", "trustloop-admin-secret-key-v1"))
    MAX_REQUEST_SIZE_BYTES: int = 1_048_576  # 1 MB

    # Model & Decision Thresholds
    POLICY_ABUSER_THRESHOLD: float = 0.30
    DECISION_CONFIDENCE_AUTO_APPROVE: float = 0.85
    DECISION_CONFIDENCE_AUTO_REJECT: float = 0.90

    # Promotion Gate Thresholds
    MIN_MACRO_F1_IMPROVEMENT: float = 0.005
    MAX_CLASS_RECALL_REGRESSION: float = 0.02
    MAX_PRECISION_REGRESSION: float = 0.02

    # Drift Thresholds
    PSI_STABLE_THRESHOLD: float = 0.10
    PSI_CRITICAL_THRESHOLD: float = 0.25
    MAX_QUARANTINE_RATIO_WARN: float = 0.30

    # Telemetry & Concurrency
    TELEMETRY_LOG_MAX_LINES: int = 10_000
    FILE_LOCK_TIMEOUT_SECONDS: float = 5.0


# Global singleton instance
settings = Settings()
