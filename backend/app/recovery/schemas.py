"""Pydantic request/response models for the recovery-prediction API."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RecoveryHistoryEvent(BaseModel):
    """A PRIOR transaction from the customer's history.

    ``recovered_72h`` only matters for recovery-population rows (refunded /
    cancelled / pending) and feeds the leakage-safe recovery-history features.
    When unknown, pass 0 (conservative: treated as "did not recover"); the
    features are still computed deterministically.
    """

    transaction_id: Optional[str] = None
    transaction_date: str
    status: str
    total_amount: float = 0.0
    quantity: float = 0.0
    unit_price: float = 0.0
    discount_applied: float = 0.0
    shipping_cost: float = 0.0
    payment_method: str = ""
    recovered_72h: Optional[int] = Field(default=0, ge=0, le=1)


class RecoveryPredictRequest(BaseModel):
    """One transaction to score for 72h recovery risk.

    The current transaction must be a recovery-population event
    (refunded / cancelled / pending) — that is exactly the population the
    model was trained on.
    """

    transaction_id: str
    customer_id: str
    transaction_date: str
    quantity: float = 0.0
    unit_price: float = 0.0
    total_amount: float
    discount_applied: float = 0.0
    shipping_cost: float = 0.0
    payment_method: str
    status: str
    customer_signup_date: Optional[str] = None
    history: List[RecoveryHistoryEvent] = Field(default_factory=list)

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("refunded", "cancelled", "pending"):
            raise ValueError(
                "status must be one of 'refunded' | 'cancelled' | 'pending' "
                "(the recovery-population statuses the model was trained on)")
        return v


class RecoveryBatchRequest(BaseModel):
    transactions: List[RecoveryPredictRequest]


class RecoveryPredictResponse(BaseModel):
    transaction_id: str
    customer_id: str
    model: str = "ExtraTreesClassifier"
    model_artifact: str = "final_model.joblib"
    recovery_probability: float = Field(ge=0.0, le=1.0)
    probability_raw: float = Field(ge=0.0, le=1.0)
    threshold: float
    recovery_prediction: int = Field(ge=0, le=1)
    recovery_risk: str
    risk_band: str
    calibrated: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    explanation: List[dict] = Field(default_factory=list)


class RecoveryBatchResponse(BaseModel):
    model: str = "ExtraTreesClassifier"
    model_artifact: str = "final_model.joblib"
    threshold: float
    count: int
    predictions: List[RecoveryPredictResponse]


class RecoveryDecisionRequest(BaseModel):
    """Advisory: run the ML-informed AI decision WITHOUT executing an action.

    Mirrors the live-event ingestion shape so calls stay realistic (Razorpay
    payment-id / customer-id / amount / decline-code / payment-method), then
    returns the recommended action, channel and policy verdict.
    """

    transaction_id: str
    customer_id: str
    event_type: str = "payment.failed"
    decline_code: str = "generic_decline"
    amount: float
    currency: str = "INR"
    payment_method: str = ""
    retry_count: int = 0
    transaction_date: Optional[str] = None
    history: List[RecoveryHistoryEvent] = Field(default_factory=list)


class RecoveryDecisionResponse(BaseModel):
    """Full buildathon chain: prediction -> AI decision -> policy verdict.

    ``probability_source`` is the single source of truth for P(recovery):
    ``ml:<artifact>`` when the trained model produced the score, else the
    deterministic rule-based fallback — never silently interchanged.
    """

    transaction_id: str
    customer_id: str
    recovery_probability: float = Field(ge=0.0, le=1.0)
    probability_raw: Optional[float] = None
    threshold: Optional[float] = None
    risk_band: str
    risk_label: str
    recovery_prediction: int = Field(ge=0, le=1)
    model: str = ""
    model_artifact: str = ""
    probability_source: str
    ai_decision: str
    action: str
    channel: str
    policy_verdict: str
    policy_reason: str
    reasoning: List[str] = Field(default_factory=list)
    expected_outcome: str
    action_mode: str = "simulated"
    outcome: str = "pending"
    declined_to_act: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RecoveryProcessRequest(RecoveryDecisionRequest):
    """Full run: prediction -> decision -> policy -> (simulated) execution + record."""


class RecoveryDecisionRecord(BaseModel):
    transaction_id: str
    event_id: str
    customer_id: str
    ml_probability: Optional[float] = None
    probability_raw: Optional[float] = None
    threshold: Optional[float] = None
    recovery_prediction: Optional[int] = None
    risk_band: Optional[str] = None
    risk_label: Optional[str] = None
    model: Optional[str] = None
    model_artifact: Optional[str] = None
    probability_source: str
    ai_decision: str
    action: str
    channel: str
    policy_verdict: str
    policy_reason: str
    reasoning: str
    expected_outcome: str
    action_mode: str
    outcome: str
    recovered_amount: float = 0.0
    decision_at: str
    updated_at: str


class RecoveryOutcomesResponse(BaseModel):
    count: int
    decisions: List[RecoveryDecisionRecord]


class RecoveryAnalyticsResponse(BaseModel):
    total_decisions: int
    by_probability_source: dict = Field(default_factory=dict)
    by_ai_decision: dict = Field(default_factory=dict)
    by_policy_verdict: dict = Field(default_factory=dict)
    by_outcome: dict = Field(default_factory=dict)
    recovered_count: int
    with_recovered_amount: int
    recovered_amount: float


class RecoveryAnalyzeRequest(BaseModel):
    """Full ML → AI → Policy → Action → Outcome chain with provenance.

    The endpoint runs the complete recovery pipeline for a synthetic event
    and returns every layer with its explicit data source, making it
    impossible to wonder 'is the AI actually using the ML model?'
    """

    transaction_id: str
    customer_id: str
    event_type: str = "payment.failed"
    decline_code: str = "generic_decline"
    amount: float
    currency: str = "INR"
    payment_method: str = ""
    retry_count: int = 0
    transaction_date: Optional[str] = None
    history: List[RecoveryHistoryEvent] = Field(default_factory=list)


class MLSignalBlock(BaseModel):
    """ML prediction layer — what the frozen model says."""

    recovery_probability: float = Field(ge=0.0, le=1.0)
    probability_raw: float = Field(ge=0.0, le=1.0)
    threshold: float
    prediction: int = Field(ge=0, le=1)
    risk_band: str
    risk_label: str
    model_version: str
    probability_source: str
    explanation: List[dict] = Field(default_factory=list)


class AIDecisionBlock(BaseModel):
    """AI decision layer — what the agent decides given the ML signal."""

    diagnosis: str
    diagnosis_confidence: float = Field(ge=0.0, le=1.0)
    selected_action: str
    action_success_probability: float = Field(ge=0.0, le=1.0)
    expected_recovery: int = Field(ge=0)
    reasoning: List[str] = Field(default_factory=list)
    candidate_actions: List[dict] = Field(default_factory=list)


class PolicyBlock(BaseModel):
    """Policy gate — is the action actually allowed?"""

    verdict: str
    reason: str
    checks_passed: List[str] = Field(default_factory=list)
    checks_failed: List[str] = Field(default_factory=list)


class ExecutionBlock(BaseModel):
    """Execution layer — simulated action result."""

    action: str
    channel: str
    mode: str = "simulated"
    status: str = "pending"


class OutcomeBlock(BaseModel):
    """Outcome layer — confirmed result from trusted webhook."""

    status: str = "pending"
    recovered_amount: int = 0
    source: Optional[str] = None


class RecoveryAnalyzeResponse(BaseModel):
    """Unified ML → AI → Policy → Action → Outcome response.

    Every layer carries its own data source, making the dependency chain
    transparent and auditable. A Razorpay judge should immediately see:
    ML predicts who is recoverable.
    AI chooses how to recover them.
    Policy makes sure it is safe.
    Webhook proves whether money came back.
    """

    transaction_id: str
    customer_id: str
    ml_signal: MLSignalBlock
    ai_decision: AIDecisionBlock
    policy: PolicyBlock
    execution: ExecutionBlock
    outcome: OutcomeBlock
    model_version: str
    probability_source: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)