from pydantic import BaseModel, Field
from typing import Optional, List, Any
from enum import Enum
from datetime import datetime


class EventType(str, Enum):
    CARD_PAYMENT_FAILURE = "card_payment_failure"
    RECURRING_PAYMENT_FAILURE = "recurring_payment_failure"
    CHECKOUT_ABANDONMENT = "checkout_abandonment"
    OVERDUE_INVOICE = "overdue_invoice"
    PROMISE_TO_PAY = "promise_to_pay"


class DeclineCode(str, Enum):
    INSUFFICIENT_FUNDS = "insufficient_funds"
    EXPIRED_CARD = "expired_card"
    DO_NOT_HONOR = "do_not_honor"
    BANK_TIMEOUT = "bank_timeout"
    INCORRECT_CVC = "incorrect_cvc"
    PROCESSING_ERROR = "processing_error"
    MANDATE_AFA_REQUIRED = "mandate_afa_required"
    MANDATE_SIMPLE_RETRY = "mandate_simple_retry"
    PAYMENT_LINK_EXPIRED = "payment_link_expired"
    INVOICE_OVERDUE = "invoice_overdue"
    GENERIC_DECLINE = "generic_decline"


class Customer(BaseModel):
    id: str
    name: str
    phone: str
    email: str
    language_pref: str = "hi"
    opted_out: bool = False
    dnd_registered: bool = False


class TransactionMetadata(BaseModel):
    card_last4: Optional[str] = None
    bank: Optional[str] = None
    mandate_id: Optional[str] = None
    subscription_id: Optional[str] = None
    invoice_id: Optional[str] = None
    cart_value: Optional[int] = None
    days_overdue: Optional[int] = None
    previous_attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    discount_hint: Optional[int] = None


class RevenueEvent(BaseModel):
    id: str
    type: EventType
    customer: Customer
    amount: int
    currency: str = "INR"
    root_cause: DeclineCode
    decline_code: DeclineCode
    failed_at: datetime
    metadata: TransactionMetadata = TransactionMetadata()
    ground_truth: str = "uncertain"
    recovered_amount: int = 0
    retry_count: int = 0
    last_attempt_at: Optional[datetime] = None
    status: str = "pending"
    # Normalized-event fields (populated by the webhook normalization layer).
    transaction_id: Optional[str] = None
    occurred_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    source: str = "unknown"
    received_at: Optional[datetime] = None
    # Closed-loop sequence fields.
    recovery_key: Optional[str] = None
    attempt_number: int = 1
    max_steps: Optional[int] = None


class WorkflowStatus(str, Enum):
    READY_FOR_POLICY = "READY_FOR_POLICY"
    STOPPED = "STOPPED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    RESOLVED = "RESOLVED"
    PENDING_WEBHOOK = "PENDING_WEBHOOK"


class RiskFlag(BaseModel):
    code: str
    severity: str
    message: str


class SpecialistCall(BaseModel):
    agent: str
    input_summary: str
    output_summary: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvidenceItem(BaseModel):
    field: str
    value: str
    interpretation: str


class DiagnosisOutput(BaseModel):
    classification: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[EvidenceItem] = []
    likely_recoverability: str
    recommended_wait_hours: int = 0
    recommended_action_family: str
    uncertainties: List[str] = []
    requires_afa: bool = False
    max_retries: int = 3
    risk_score: float = 0.3
    optimal_delay_hours: int = 0


class ContactFrequency(BaseModel):
    contacts_last_24h: int = 0
    contacts_last_7d: int = 0
    last_contact_at: Optional[str] = None


class CustomerContextOutput(BaseModel):
    customer_id: str
    consent_status: str
    contact_frequency: ContactFrequency = ContactFrequency()
    active_dispute: bool = False
    active_ptp: bool = False
    preferred_channel: str
    risk_flags: List[str] = []
    safe_to_contact: bool = True


class ProposedAction(BaseModel):
    action: str
    channel: str = "razorpay_api"
    amount: int = 0
    discount_percent: int = 0
    message_tone_level: int = 1
    message_text: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    reason: str = ""


class RecoveryCandidate(BaseModel):
    """A single candidate intervention with its Expected Recovery Value.

    The scorer evaluates the FULL candidate matrix for an event and marks each
    action eligible/ineligible *advisably* — the Deterministic Policy Engine
    downstream remains the authoritative safety boundary and is never bypassed.
    """
    strategy: str
    action: str
    channel: str
    probability: float = Field(ge=0.0, le=1.0)
    probability_source: str = "rule-based-v1"
    expected_value: int = Field(ge=0)
    reason: str = ""
    priority: str = "MEDIUM"
    requires_human_approval: bool = False
    from_learning: bool = False
    model_version: str = "rule-based-v1"
    probability_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    customer_friction: int = Field(default=0, ge=0, le=10)
    contact_cost: int = Field(default=0, ge=0)
    policy_eligible: bool = True
    # Advisory ineligibility marking (policy still makes the final decision).
    eligible: bool = True
    ineligibility_reason: str = ""
    # Structured, concise explanation codes for auditability ("why this action").
    reason_codes: List[str] = []
    # Outcome-informed stats: raw historical n/m for (strategy, decline_code),
    # so the candidate view shows the empirical evidence behind each estimate.
    empirical_attempts: int = Field(default=0, ge=0)
    empirical_successes: int = Field(default=0, ge=0)


class RecoveryObjective(BaseModel):
    """Score a single candidate across the multi-factor objective."""
    recovery_probability: float = Field(ge=0.0, le=1.0)
    expected_recovery_value: int = Field(ge=0)
    risk_score: float = Field(ge=0.0, le=1.0)
    customer_friction: int = Field(ge=0, le=10)
    contact_cost: int = Field(ge=0)
    policy_eligible: bool = True
    composite_score: float = Field(ge=0.0)
    explanation: str = ""


class RecoveryStrategyOutput(BaseModel):
    strategy: str
    priority: str
    reason: str
    proposed_delay_hours: int = 0
    channel: str
    discount_percent: int = 0
    requires_human_approval: bool = False
    expected_value: int = 0


class OptimizerOutput(BaseModel):
    """Ranked candidate set plus the elected best policy-eligible candidate."""
    candidates: List[RecoveryCandidate] = []
    elected: Optional[RecoveryCandidate] = None
    selection_reason: str = ""
    strategy: Optional[RecoveryStrategyOutput] = None
    # Concise structured decision factors (no chain-of-thought) for the audit trail.
    decision_factors: List[str] = []


class PolicyVerdict(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    MODIFY = "MODIFY"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class PolicyCheckDetail(BaseModel):
    rule: str
    result: str
    explanation: str
    regulatory_basis: Optional[str] = None


class PolicyDecision(BaseModel):
    verdict: PolicyVerdict
    reason: str
    checks_passed: List[str] = []
    checks_failed: List[str] = []
    detailed_results: List[PolicyCheckDetail] = []
    modified_action: Optional[str] = None
    requires_human_approval: bool = False
    original_request: Optional[dict] = None
    modified_request: Optional[dict] = None


class AgentRecommendation(BaseModel):
    strategy: str = Field(description="Recommended strategy")
    priority: str = Field(description="LOW, MEDIUM, HIGH, CRITICAL")
    proposed_delay_hours: int = Field(ge=0, le=720)
    channel: str = Field(description="RAZORPAY_API, WHATSAPP, SMS, EMAIL, NONE")
    discount_percent: float = Field(ge=0, le=100)
    requires_human_approval: bool
    reason: str


class ComplianceExplanation(BaseModel):
    verdict: str
    summary: str
    rules_triggered: List[PolicyCheckDetail] = []
    customer_safe_explanation: str
    operator_explanation: str


class SupervisorOutput(BaseModel):
    event_id: str
    workflow_status: WorkflowStatus
    specialist_calls: List[SpecialistCall] = []
    risk_flags: List[RiskFlag] = []
    next_step: str
    proposed_action: Optional[ProposedAction] = None
    diagnosis: Optional[DiagnosisOutput] = None
    customer_context: Optional[CustomerContextOutput] = None
    compliance_explanation: Optional[ComplianceExplanation] = None
    policy_decision: Optional[PolicyDecision] = None
    optimizer: Optional[dict] = None


class ExecutionResult(BaseModel):
    event_id: str
    action: str
    result: str
    amount_recovered: int = 0
    reason: str = ""
    channel: str = ""
    message_sent: Optional[str] = None
    payment_link: Optional[str] = None
    execution_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AuditEntry(BaseModel):
    id: str
    timestamp: datetime
    event_id: str
    customer_id: str
    workflow_status: str
    specialist_calls: List[dict] = []
    proposed_action: Optional[dict] = None
    policy_decision: Optional[dict] = None
    execution_result: Optional[dict] = None
    risk_flags: List[dict] = []
    rule_version: str = "1.0"


class BatchResult(BaseModel):
    batch_id: str
    total_records: int
    attempted: int
    recovered: int
    recovered_amount: int
    baseline_amount: int
    blocked_by_policy: int
    human_review: int = 0
    pending_webhook: int = 0
    errors: int = 0
    records: List[ExecutionResult] = []
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class BatchSummary(BaseModel):
    total: int
    processed: int
    recovered: int
    recovered_amount: int
    pending: int
    denied: int
    human_review: int
    errors: int
    blocked_by_policy: int
    contact_rate: float = 0.0
    opt_out_violations: int = 0
    afa_violations: int = 0
    excess_retries: int = 0
    false_recovery_rate: float = 0.0


class WebhookPayload(BaseModel):
    event_id: str
    webhook_type: str
    payload: dict = {}


class ContactEvent(BaseModel):
    id: Optional[int] = None
    customer_id: str
    event_id: Optional[str] = None
    channel: str
    sent_at: datetime
    status: str
    message_type: Optional[str] = None


class PipelineError(BaseModel):
    event_id: str
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StrategyEffectiveness(BaseModel):
    """Rolled-up effectiveness of one strategy across historical outcomes."""
    strategy: str
    attempts: int = 0
    successes: int = 0
    recovery_amount: int = 0
    contact_rate: float = 0.0
    empirical_probability: float = 0.0
    avg_recovery_per_attempt: int = 0
