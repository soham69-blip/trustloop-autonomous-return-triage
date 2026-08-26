from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


class ReturnCase(BaseModel):
    """
    Standard Return & Refund Case schema.
    Fully backwards-compatible with all Phase 4 API payloads.
    Supports additional optional multi-modal evidence paths.
    """
    case_id: str = Field(..., description="Unique case identifier")

    customer_id: Optional[str] = None
    order_id: Optional[str] = None

    age: int = 0
    account_age_days: int = 0

    customer_segment: Optional[str] = None
    country: Optional[str] = None
    platform: Optional[str] = None
    device_type: Optional[str] = None
    payment_method: Optional[str] = None
    product_category: Optional[str] = None

    avg_order_value_usd: float = 0.0
    is_high_value_item: int = 0
    discount_used: int = 0

    days_to_return: float = 0.0
    return_reason: Optional[str] = None
    shipping_carrier: Optional[str] = None

    multiple_accounts_flag: int = 0
    wishlist_to_cart_time_hrs: float = 0.0

    customer_return_count_prior: int = 0
    returns_last_30d_prior: int = 0
    returns_last_90d_prior: int = 0
    total_returns_lifetime_prior: int = 0

    # Safe decision-time profile & claim features (Experiment A candidate compatible)
    total_returns_lifetime: int = 0
    total_orders_lifetime: int = 0
    return_rate_pct: float = 0.0
    customer_support_contacts: int = 0
    previous_dispute_count: int = 0
    refund_amount_requested_usd: Optional[float] = None

    order_date: Optional[str] = None
    return_date: Optional[str] = None

    item_condition: Optional[str] = None
    refund_amount: float = 0.0

    # Optional Multi-Modal Evidence Fields
    image_path: Optional[str] = None
    claim_notes: Optional[str] = None


class PolicyFlagSchema(BaseModel):
    rule: str
    status: str
    reason: str


class RetrievedChunkSchema(BaseModel):
    index: int
    score: float
    text: str
    source: Optional[str] = None


class RAGPolicyResult(BaseModel):
    available: bool = True
    policy_status: str
    flags: List[PolicyFlagSchema] = []
    retrieved_policy: List[RetrievedChunkSchema] = []
    query: Optional[str] = None
    reason: Optional[str] = None


class VisionEvidenceResult(BaseModel):
    available: bool = False
    verified: bool = False
    reason: Optional[str] = None
    image_quality: Optional[str] = None
    product_condition: Optional[str] = None
    damage_detected: Optional[bool] = None
    packaging_condition: Optional[str] = None
    evidence_consistent: Optional[bool] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    image_path: Optional[str] = None


class SHAPDriver(BaseModel):
    feature: str
    value: Any
    attribution: float


class SHAPExplanationResult(BaseModel):
    available: bool = False
    predicted_class: Optional[str] = None
    top_positive_drivers: List[SHAPDriver] = []
    top_negative_drivers: List[SHAPDriver] = []
    explanation_summary: Optional[str] = None


class RiskComponentsSchema(BaseModel):
    deterministic_risk: float
    ml_risk: float
    ml_weight: float = 0.70
    deterministic_weight: float = 0.30


class FeatureContractAuditSchema(BaseModel):
    target_feature_set: str
    contract_valid: bool
    missing_candidate_fields: List[str] = []
    derived_fields: List[str] = []
    inconsistencies: List[str] = []
    warnings: List[str] = []


class ModelStatusSchema(BaseModel):
    model_loaded: bool
    model_name: Optional[str] = None
    model_role: str
    model_type: Optional[str] = None
    feature_count: int = 0
    features: List[str] = []
    classes: List[str] = []
    status: str
    error: Optional[str] = None


class TriageResponse(BaseModel):
    """
    Standard full triage analysis response.
    Maintains 100% backward compatibility with Phase 4 response contract.
    """
    case_id: str

    risk_score: float
    deterministic_risk: float
    ml_risk: float
    decision_confidence: float
    decision: str
    signals: List[str] = []

    ml_prediction: Union[int, str]
    ml_label: str
    ml_confidence: float
    class_probabilities: Dict[str, float]

    risk_components: Dict[str, Any]
    feature_contract: FeatureContractAuditSchema
    model_status: ModelStatusSchema
    status: str = "ml_analysis_complete"

    # Enriched Service Layer Fields (Optional for Phase 5)
    policy_analysis: Optional[RAGPolicyResult] = None
    vision_analysis: Optional[VisionEvidenceResult] = None
    shap_explanation: Optional[SHAPExplanationResult] = None
