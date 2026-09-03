"""ML -> AI bridge: the single source of truth for the recovery probability.

The trained Recovery ML model answers "how likely is this transaction to
recover within 72h?" (calibrated probability). The AI/agent layer answers
"what should we do about it?" (decision, policy, action). This module is the
seam that feeds the ML prediction into the agent layer without collapsing the
two responsibilities.

* `recovery_prediction_for_request`: score a full transaction (44-feature path),
  exactly like `/api/recovery/predict` but returning an `MLPrediction` object
  the agent layer can consume.
* `recovery_prediction_for_event`: score a live `RevenueEvent` (the automated
  loop / webhook path). The live failure is mapped to the recovery-population
  status `pending` (money at risk), which is the population the model was
  trained on.
* `build_event_from_request`: build a `RevenueEvent` from a
  `RecoveryPredictRequest` so the decision/process endpoints run through the
  *same* optimizer + policy engine as the live loop.

Authoritative-source contract:
  * The ML probability is the canonical recovery probability. The rule-based
    estimator is retained as the documented fallback (`probability_source`).
  * If the model is unavailable, `ok=False` + `fallback_reason` lets the caller
    fall back to the rule-based estimator and mark it as a fallback, never as
    the ML score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from app.models import Customer, DeclineCode, EventType, RevenueEvent
from app.recovery.feature_builder import ALL_FEATURES, build_raw_features
from app.recovery.model_service import get_model_service, risk_band, risk_label
from app.recovery.schemas import RecoveryPredictRequest

MODEL_VERSION = "ml:ExtraTreesClassifier-500-final"

RULE_BASED_VERSION = "rule-based-v1"

# Live failure events map to the recovery-population "pending" status — the
# transaction has an open, unsettled balance that could still be recovered.
LIVE_EVENT_POPULATION_STATUS = "pending"


@dataclass
class MLPrediction:
    """Canonical ML recovery prediction consumed by the agent/AI layer."""

    ok: bool
    fallback_reason: Optional[str] = None
    recovery_probability: Optional[float] = None
    probability_raw: Optional[float] = None
    threshold: Optional[float] = None
    recovery_prediction: Optional[int] = None
    risk_band: Optional[str] = None
    risk_label: Optional[str] = None
    model: str = "ExtraTreesClassifier"
    model_artifact: str = "final_model.joblib"
    model_version: str = MODEL_VERSION
    probability_source: str = MODEL_VERSION
    explanation: list[dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def available(self) -> bool:
        return self.ok and self.recovery_probability is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "recovery_probability": self.recovery_probability,
            "probability_raw": self.probability_raw,
            "threshold": self.threshold,
            "recovery_prediction": self.recovery_prediction,
            "risk_band": self.risk_band,
            "risk_label": self.risk_label,
            "model": self.model,
            "model_artifact": self.model_artifact,
            "model_version": self.model_version,
            "probability_source": self.probability_source,
            "explanation": self.explanation,
            "available": self.available,
            "fallback_reason": self.fallback_reason,
        }


def _unavailable(reason: str, **kwargs: Any) -> MLPrediction:
    return MLPrediction(ok=False, fallback_reason=reason,
                        probability_source=RULE_BASED_VERSION, **kwargs)


def recovery_prediction_for_request(
    request: RecoveryPredictRequest,
    service: Any = None,
    raw_features: Optional[dict] = None,
) -> MLPrediction:
    """Canonical ML prediction for a RecoveryPredictRequest (44 -> 49 -> model)."""
    service = service or get_model_service()
    try:
        raw = raw_features if raw_features is not None else build_raw_features(request)
        raw_df = pd.DataFrame([raw], columns=ALL_FEATURES)
        encoded = service.encode(raw_df)
        prob_raw, prob_cal = service.predict_proba(encoded)
        pred = int(service.decide(prob_cal)[0])
        cal = float(prob_cal[0])
        return MLPrediction(
            ok=True,
            recovery_probability=round(cal, 6),
            probability_raw=round(float(prob_raw[0]), 6),
            threshold=service.selected_threshold,
            recovery_prediction=pred,
            risk_band=risk_band(cal),
            risk_label=risk_label(cal),
            explanation=service.explain(raw_df),
            probability_source=MODEL_VERSION,
        )
    except Exception as exc:  # surface as a clean fallback, never a crash
        return _unavailable(f"ML prediction failed: {exc}")


def recovery_prediction_for_event(
    event: RevenueEvent,
    history: Optional[list] = None,
    payment_method: str = "",
    signup_date: Optional[str] = None,
) -> MLPrediction:
    """Canonical ML prediction for a live RevenueEvent (automated-loop path)."""
    transaction_date = event.failed_at.isoformat()
    request = RecoveryPredictRequest(
        transaction_id=event.transaction_id or event.id,
        customer_id=event.customer.id,
        transaction_date=transaction_date,
        quantity=0.0,
        unit_price=0.0,
        total_amount=float(event.amount or 0),
        discount_applied=0.0,
        shipping_cost=0.0,
        payment_method=payment_method or "",
        status=LIVE_EVENT_POPULATION_STATUS,
        customer_signup_date=signup_date or None,
        history=list(history or []),
    )
    return recovery_prediction_for_request(request)


def build_event_from_request(request: RecoveryPredictRequest) -> RevenueEvent:
    """Build a RevenueEvent from a RecoveryPredictRequest.

    The decision/process endpoints push the request through the SAME
    optimizer + policy engine as the live loop. The synthetic event uses the
    generic decline code and a card-payment failure type; task-specific
    policies (AFA, retries, contact windows) still apply unchanged.
    """
    return RevenueEvent(
        id=request.transaction_id,
        type=EventType.CARD_PAYMENT_FAILURE,
        customer=Customer(id=request.customer_id, name=request.customer_id,
                          phone="", email=""),
        amount=int(request.total_amount or 0),
        currency="INR",
        root_cause=DeclineCode.GENERIC_DECLINE,
        decline_code=DeclineCode.GENERIC_DECLINE,
        failed_at=datetime.fromisoformat(request.transaction_date),
        status=LIVE_EVENT_POPULATION_STATUS,
        retry_count=0,
        transaction_id=request.transaction_id,
    )


__all__ = [
    "MLPrediction", "MODEL_VERSION", "RULE_BASED_VERSION",
    "recovery_prediction_for_request", "recovery_prediction_for_event",
    "build_event_from_request",
]