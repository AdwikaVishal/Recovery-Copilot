"""Recovery Probability Estimator abstraction.

Clean seam so that a trained ML model (XGBoost / LightGBM / logistic / neural)
can replace the deterministic rule-based estimator later, once real historical
outcomes exist. Nothing here claims to be ML — it is transparent, rule-based,
refined over time by the outcome->learning loop via app.database.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod

from app.models import RevenueEvent, CustomerContextOutput, RecoveryCandidate


@dataclass
class ProbabilityEstimate:
    probability: float = 0.0
    confidence: float = 0.0
    model_version: str = ""
    features_used: list[str] | None = None


class RecoveryProbabilityEstimator(ABC):
    """Estimates P(recovery | event, customer_context, candidate_action)."""

    @abstractmethod
    async def estimate(
        self,
        event: RevenueEvent,
        context: CustomerContextOutput,
        candidate: RecoveryCandidate,
    ) -> ProbabilityEstimate:
        ...


# Transparent, deterministic base recovery probabilities per strategy/decline_code.
# These are the "before any learning data" priors. Refined over time by the
# outcome->learning loop (see app.database.get_recovery_probability).
BASE_PROBABILITIES = {
    "RETRY": {
        "insufficient_funds": 0.62, "bank_timeout": 0.74, "processing_error": 0.70,
        "do_not_honor": 0.38, "mandate_simple_retry": 0.58, "generic_decline": 0.42,
    },
    "PAYMENT_LINK": {
        "expired_card": 0.50, "incorrect_cvc": 0.42, "payment_link_expired": 0.40,
        "invoice_overdue": 0.30,
    },
    "MESSAGE": {
        "insufficient_funds": 0.25, "invoice_overdue": 0.28, "payment_link_expired": 0.22,
    },
    "REAUTHORIZE": {
        "mandate_afa_required": 0.55, "mandate_expired": 0.30, "mandate_simple_retry": 0.40,
    },
}

DEFAULT_PROB = 0.25
MODEL_VERSION = "rule-based-v1"


class RuleBasedRecoveryProbabilityEstimator(RecoveryProbabilityEstimator):
    """Deterministic, transparent probability estimator.

    - Cold prior from BASE_PROBABILITIES keyed on (strategy, decline_code).
    - Refined by empirical recovery probability from the outcome->learning loop
      once >= 3 recorded outcomes exist for that (strategy, decline_code).
    - Confidence scales with diagnosis/customer evidencing signal.
    """

    async def estimate(
        self,
        event: RevenueEvent,
        context: CustomerContextOutput,
        candidate: RecoveryCandidate,
    ) -> ProbabilityEstimate:
        from app.database import get_recovery_probability

        base = _base_probability(candidate.strategy, event)

        # Replay/benchmark events (txn_*) are deterministic on purpose: they run
        # on cold priors only, never on empirical outcomes, so the offline
        # benchmark stays byte-identical across runs AND live/eval learning can
        # never leak into it. Isolation of the learning loop per environment.
        if event.id.startswith("txn_"):
            return ProbabilityEstimate(
                probability=round(base, 4),
                confidence=round(0.5, 2),
                model_version=MODEL_VERSION,
                features_used=["decline_code", "strategy", "safe_to_contact"],
            )

        # Live events: empirical recovery probability scoped to live outcomes,
        # so live learning never influences the benchmark or the evaluator.
        learned = await get_recovery_probability(
            candidate.strategy, event.decline_code.value, base, source="live"
        )
        from_learning = learned != base

        # Confidence: model has more faith when learned from real data and when
        # the customer is safe to contact with clear context.
        confidence = 0.8 if from_learning else 0.5
        if not context.safe_to_contact:
            confidence = 0.2

        return ProbabilityEstimate(
            probability=round(learned, 4),
            confidence=round(confidence, 2),
            model_version=MODEL_VERSION,
            features_used=["decline_code", "strategy", "safe_to_contact", "empirical_outcomes"],
        )


def _base_probability(strategy: str, event: RevenueEvent) -> float:
    table = BASE_PROBABILITIES.get(strategy, {})
    return table.get(event.decline_code.value, DEFAULT_PROB)


_default_estimator = RuleBasedRecoveryProbabilityEstimator()


def get_probability_estimator() -> RecoveryProbabilityEstimator:
    """Return the configured estimator (currently the rule-based one)."""
    return _default_estimator


__all__ = [
    "RecoveryProbabilityEstimator", "ProbabilityEstimate",
    "RuleBasedRecoveryProbabilityEstimator", "get_probability_estimator",
]
