from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid


class FeedbackSubmission(BaseModel):
    """
    Submission schema for human auditor or warehouse inspection feedback.
    """
    case_id: str = Field(..., description="Unique case identifier")
    human_verified_label: str = Field(
        ...,
        description="Ground-truth label verified by human reviewer: Legitimate, Policy Abuser, Fraudulent Return, Wardrobing"
    )
    human_decision: Optional[str] = Field(
        default=None,
        description="Action taken by human: Approved, Rejected, Overturned"
    )
    reviewer_id: str = Field(..., description="Identifier of the human reviewer or auditor")
    feedback_source: str = Field(
        default="human_auditor",
        description="Source of feedback: human_auditor, customer_dispute, warehouse_inspection"
    )
    notes: Optional[str] = Field(default=None, description="Auditor review notes")
    raw_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Original case payload features for training export"
    )
    traffic_type: str = Field(
        default="production",
        description="Traffic classification: production, test, demo, synthetic"
    )


class FeedbackRecord(BaseModel):
    """
    Complete persistent feedback record with audit trails and quality gate status.
    """
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_version: str = "production-v1.3.0"
    feature_version: str = "33-feat-prod"

    predicted_class: Optional[str] = None
    prediction_probabilities: Optional[Dict[str, float]] = None
    risk_score: Optional[float] = None
    decision: Optional[str] = None
    deterministic_signals: List[str] = []

    policy_status: Optional[str] = None
    vision_verified: Optional[bool] = None
    shap_summary: Optional[str] = None

    human_verified_label: str
    human_decision: Optional[str] = None
    reviewer_id: str
    feedback_source: str = "human_auditor"
    traffic_type: str = "production"
    feedback_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    training_eligible: bool = False
    quality_status: str = "PENDING_VALIDATION"  # ELIGIBLE, QUARANTINED, REJECTED
    quarantine_reasons: List[str] = []

    raw_features: Dict[str, Any] = {}
    notes: Optional[str] = None


class FeedbackStats(BaseModel):
    total_records: int = 0
    eligible_count: int = 0
    quarantined_count: int = 0
    rejected_count: int = 0
    class_distribution: Dict[str, int] = {}
    source_distribution: Dict[str, int] = {}


class ModelVersionMetadata(BaseModel):
    model_version: str
    model_role: str  # production, candidate, archived
    feature_version: str
    feature_names: List[str]
    training_dataset_version: str
    training_timestamp: str
    artifact_sha256: str
    parent_version: Optional[str] = None
    metrics: Dict[str, float] = {}
    class_distribution: Dict[str, int] = {}
    is_active: bool = False


class PromotionGateChecklist(BaseModel):
    candidate_version: str
    production_version: str
    all_tests_passed: bool
    feature_contract_compatible: bool
    no_data_leakage: bool
    macro_f1_improved: bool
    macro_f1_delta: float
    fraud_recall_maintained: bool
    fraud_recall_delta: float
    policy_abuser_recall_maintained: bool
    policy_abuser_recall_delta: float
    wardrobing_recall_maintained: bool
    wardrobing_recall_delta: float
    legitimate_precision_maintained: bool
    legitimate_precision_delta: float
    promoted: bool
    rejection_reasons: List[str] = []
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
