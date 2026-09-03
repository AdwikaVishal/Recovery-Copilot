"""Recovery Optimizer agent — Intelligent Recovery Action Scoring.

Upgrades the single-strategy recovery decision into a ranked, multi-candidate
optimization with Expected Recovery Value (probability of recovery x amount at
risk). The optimizer evaluates the FULL matrix of permitted action families for
the event (retry / payment link / reminder / reauthorize / human review),
scores each with P(recovery) and Expected Recovery Value, ranks them, marks
advisably-ineligible candidates, and elects the best one.

Architecture guarantees:
  * The optimizer ONLY proposes/optimizes. The Deterministic Policy Engine
    downstream remains the FINAL AUTHORITY and is never bypassed.
  * On a cold DB (no empirical learning data) the ELECTED strategy is the exact
    deterministic output of `build_strategy` — keeping the batch benchmark and
    the scenario suite byte-identical. The ranked candidate set is an additive
    decision layer on top.
  * The outcome->learning loop can overturn the deterministic primary only when
    real recorded outcomes (>=3 samples) materially favor an eligible
    alternative with higher Expected Recovery Value.
  * No ML, no random probabilities. The deterministic `RecoveryProbabilityEstimator`
    behind a clean seam can be swapped for a trained `MLRecoveryScorer` later.
"""
from datetime import datetime

from app.config import PolicyConfig
from app.models import (
    RevenueEvent, DiagnosisOutput, CustomerContextOutput,
    RecoveryCandidate, RecoveryObjective, OptimizerOutput, RecoveryStrategyOutput,
)
from agents.probability_estimator import (
    get_probability_estimator, BASE_PROBABILITIES, DEFAULT_PROB,
)
from agents.recovery_strategy_agent import (
    build_strategy, strategy_to_proposed_action,
)
from app.recovery.bridge import MLPrediction, MODEL_VERSION, RULE_BASED_VERSION

CONFIG = PolicyConfig()

# Action families the scorer evaluates for every event.
_RETRY_BASE_CODES = set(BASE_PROBABILITIES.get("RETRY", {}).keys())
_MANDATE_INVOLVED = {"mandate_afa_required", "mandate_simple_retry"}


async def build_optimizer_output(
    event: RevenueEvent,
    diagnosis: DiagnosisOutput,
    context: CustomerContextOutput,
    now: datetime | None = None,
    ml_estimate: MLPrediction | None = None,
) -> OptimizerOutput:
    """Generate, score, rank and elect a recovery candidate.

    The elected strategy is the deterministic primary from `build_strategy`
    (preserving the exact original decision/contract so the batch benchmark and
    scenarios are unchanged on a cold DB). The candidate set is the additive
    optimizer layer: every permitted action family is evaluated with Expected
    Recovery Value, ineligible ones are marked (advisable — policy decides),
    and the outcome->learning loop can overturn the primary only when real data
    materially favors an eligible alternative.

    When `ml_estimate` is provided, the calibrated ML recovery probability is
    the authoritative P(recovery) for every candidate (single source of truth);
    the rule-based estimator is the documented fallback otherwise.
    """
    primary_strategy = build_strategy(event, diagnosis, context)

    candidates = []
    if primary_strategy.strategy != "STOP":
        candidates.append(await _from_strategy(
            event, diagnosis, context, primary_strategy, now, ml_estimate))

    candidates.extend(await _full_matrix(
        event, diagnosis, context, primary_strategy, now, ml_estimate))

    for c in candidates:
        _score_objective(event, c)

    # Rank by Expected Recovery Value (desc), eligible candidates first so a blocked
    # option can never displace a runnable recommendation in the visible ordering.
    # Ineligible candidates stay visible (ranked by EV within their set) with an
    # explicit reason. Deterministic tie-break by strategy.
    candidates.sort(key=lambda c: (not c.eligible, -c.expected_value, c.strategy))

    # Elect the deterministic primary candidate unless empirical learning
    # materially favors an eligible alternative (higher EV AND real data).
    primary_candidate = next(
        (c for c in candidates if c.strategy == primary_strategy.strategy), None
    )
    elected_candidate = primary_candidate
    if selected := _better_than_primary(candidates, primary_strategy):
        elected_candidate = selected

    same_as_primary = bool(
        elected_candidate and elected_candidate.strategy == primary_strategy.strategy
    )
    consensus = [c.strategy for c in candidates if c.eligible]
    prob_source = (
        f"ML probability ({ml_estimate.recovery_probability:.4f}, "
        f"{ml_estimate.risk_band})"
        if ml_estimate is not None and ml_estimate.available
        else "rule-based probability"
    )
    if same_as_primary:
        selection_reason = (
            f"Primary {primary_strategy.strategy} kept (eligible candidates ranked "
            f"by EV using {prob_source}: {consensus})."
        )
    elif elected_candidate:
        selection_reason = (
            f"Empirical data favors {elected_candidate.strategy} (EV ₹{elected_candidate.expected_value // 100:,}) "
            f"over primary {primary_strategy.strategy}."
        )
    else:
        selection_reason = "No viable candidate."

    # Return the exact deterministic primary when unchanged (byte-identical output),
    # otherwise a candidate-derived strategy.
    strategy = (
        primary_strategy
        if same_as_primary
        else (_candidate_to_strategy(event, elected_candidate) if elected_candidate else None)
    )

    return OptimizerOutput(
        candidates=candidates,
        elected=elected_candidate,
        selection_reason=selection_reason,
        strategy=strategy,
        decision_factors=_decision_factors(event, diagnosis, context, ml_estimate),
    )


def _better_than_primary(candidates, primary_strategy) -> RecoveryCandidate | None:
    """Return an eligible alternative that empirically beats the primary."""
    primary_cand = next((c for c in candidates if c.strategy == primary_strategy.strategy), None)
    if primary_cand is None:
        return None
    for c in candidates:
        if c.strategy == primary_strategy.strategy or c.strategy in ("STOP", "HUMAN_REVIEW"):
            continue
        if not c.eligible:
            continue
        # Only overturn the primary on real learned evidence, not a cold prior.
        if c.from_learning and c.expected_value > primary_cand.expected_value:
            return c
    return None


async def _from_strategy(event, diagnosis, context, strategy: RecoveryStrategyOutput,
                         now, ml_estimate=None) -> RecoveryCandidate:
    cand = RecoveryCandidate(
        strategy=strategy.strategy,
        action=strategy_to_proposed_action(event, strategy).action,
        channel=strategy.channel,
        probability=DEFAULT_PROB,
        expected_value=0,
        reason=strategy.reason,
        priority=strategy.priority,
        requires_human_approval=strategy.requires_human_approval,
    )
    await _apply_estimation(event, context, cand, ml_estimate)
    _apply_eligibility(event, diagnosis, context, cand, now)
    cand.reason_codes = _reason_codes(cand, event, diagnosis, context)
    return cand


async def _matrix_candidate(event, diagnosis, context, strategy, action, channel,
                            reason, now, human=False,
                            ml_estimate=None) -> RecoveryCandidate | None:
    """One candidate from the full action-family matrix, scored + eligibility-marked."""
    cand = RecoveryCandidate(
        strategy=strategy,
        action=action,
        channel=channel,
        probability=DEFAULT_PROB,
        expected_value=0,
        reason=reason,
        requires_human_approval=human,
    )
    await _apply_estimation(event, context, cand, ml_estimate)
    _apply_eligibility(event, diagnosis, context, cand, now)
    cand.reason_codes = _reason_codes(cand, event, diagnosis, context)
    return cand


async def _full_matrix(event, diagnosis, context, primary, now,
                       ml_estimate=None) -> list[RecoveryCandidate]:
    """Evaluate ALL permitted recovery action families for this event.

    This is the expected-recovery-value scoring layer: rather than silently
    omitting options, every family a reasonable operator would consider is
    represented — ineligible ones carry `eligible: false` + `ineligibility_reason`
    so the decision is explainable. The Policy Engine downstream still makes the
    authoritative call; nothing here authorizes execution.
    """
    alts = []
    seen = {primary.strategy} if primary.strategy != "STOP" else set()
    decline = event.decline_code.value
    mandate_event = (
        event.type.value == "recurring_payment_failure" or decline in _MANDATE_INVOLVED
    )
    payment_event = event.type.value in ("card_payment_failure", "recurring_payment_failure")

    async def maybe(strategy, action, channel, reason, human=False):
        if strategy in seen:
            return
        seen.add(strategy)
        alts.append(await _matrix_candidate(
            event, diagnosis, context, strategy, action, channel, reason, now,
            human=human, ml_estimate=ml_estimate,
        ))

    # Retry — only meaningful for payment declines; marked ineligible otherwise,
    # and ineligible when AFA blocks a blind retry or the retry budget is spent.
    if payment_event or decline in _RETRY_BASE_CODES:
        await maybe("RETRY", "retry_payment", "razorpay_api",
                    "Considered: immediate retry charge (highest conversion for transient failures).")

    # Payment link — direct, low-friction, control of timing; contact-based.
    await maybe("PAYMENT_LINK", "send_payment_link", "whatsapp",
                "Considered: payment link (customer controls timing); used for invalid/expired instruments.")

    # Reminder / nudge message — gentlest contact; lowest value but zero charge risk.
    await maybe("MESSAGE", "send_dunning_message", "whatsapp",
                "Considered: reminder message (gentlest; used for abandonment/overdue).")

    # Mandate reauthorization — the only AFA-compliant path for high-value recurring.
    if mandate_event or diagnosis.requires_afa:
        await maybe("REAUTHORIZE", "re_authorize_mandate", "whatsapp",
                    "Considered: mandate reauthorization (RBI AFA-compliant for high-value recurring).")

    # Human review — a safe, always-eligible escalation; never auto-executes money.
    if diagnosis.risk_score > 0.8 or diagnosis.confidence < 0.4:
        await maybe("HUMAN_REVIEW", "escalate_to_human", "internal",
                    "Low confidence / high risk — human review is a safe alternative.", human=True)

    return alts


# ---------------------------------------------------------------------------
# Advisory eligibility (NOT policy). Policy remains the authoritative boundary.
# ---------------------------------------------------------------------------

def _apply_eligibility(event: RevenueEvent, diagnosis: DiagnosisOutput,
                       context: CustomerContextOutput, cand: RecoveryCandidate,
                       now: datetime | None) -> None:
    """Mark a candidate ineligible using context/diagnosis signals only.

    These are advisory, decision-visibility flags. The Policy Engine runs the
    authoritative checks (opt-out, AFA, retries, cooling, discounts, window,
    amount, risk, confidence) and its verdict is what actually gates execution.
    """
    strategy = cand.strategy

    if strategy in ("RETRY", "PAYMENT_LINK", "MESSAGE", "REAUTHORIZE"):
        if strategy == "RETRY":
            if event.decline_code.value in ("expired_card", "incorrect_cvc"):
                cand.eligible, cand.ineligibility_reason = False, (
                    f"Payment method invalid ({event.decline_code.value}) — a blind retry "
                    "on an unworkable instrument cannot succeed; collect updated details first.")
                return
            if diagnosis.requires_afa:
                cand.eligible, cand.ineligibility_reason = False, (
                    "RBI AFA required for recurring debit — a blind retry is blocked; "
                    "reauthorization is the compliant path.")
                return
            max_r = diagnosis.max_retries if (diagnosis.max_retries or 0) > 0 else CONFIG.max_retries
            if event.retry_count >= max_r:
                cand.eligible, cand.ineligibility_reason = False, (
                    f"Retry budget exhausted (retry_count={event.retry_count} >= max {max_r}).")
                return
            if event.type.value not in ("card_payment_failure", "recurring_payment_failure"):
                cand.eligible, cand.ineligibility_reason = False, (
                    f"Retry not applicable to event type '{event.type.value}'.")
                return
            return  # initialized eligible

        # Contact-based strategies (link / reminder / reauth): gate on consent
        # and contact budget. Frequency counts come from the Customer Context Agent.
        if not context.safe_to_contact:
            cand.eligible, cand.ineligibility_reason = False, (
                f"Customer not safe to contact: {context.risk_flags or 'consent'}"
            )
            return
        weekly = context.contact_frequency.contacts_last_7d
        max_week = CONFIG.max_contacts_per_week
        if weekly >= max_week:
            cand.eligible, cand.ineligibility_reason = False, (
                f"Contact-frequency limit reached ({weekly}/{max_week} this week).")
            return
        if strategy == "REAUTHORIZE":
            if not (diagnosis.requires_afa or event.type.value == "recurring_payment_failure"):
                cand.eligible, cand.ineligibility_reason = False, (
                    "Mandate reauthorization applies to recurring mandates only.")
                return
            return
        if now is not None:
            try:
                hour = now.hour
                start = int(CONFIG.contact_window_start.split(":")[0])
                end = int(CONFIG.contact_window_end.split(":")[0])
                if not (start <= hour < end):
                    cand.eligible, cand.ineligibility_reason = False, (
                        f"Outside contact window ({CONFIG.contact_window_start}–{CONFIG.contact_window_end}).")
                    return
            except Exception:
                pass


def _reason_codes(cand: RecoveryCandidate, event: RevenueEvent,
                  diagnosis: DiagnosisOutput, context: CustomerContextOutput) -> list[str]:
    """Structured, concise explanation codes for the audit trail (no chain-of-thought)."""
    codes = []
    if not cand.eligible:
        # Ineligibility codes keyed off the generated reason (advisory only).
        if "AFA required" in cand.ineligibility_reason:
            codes.append("AFA_REQUIRED")
        elif "budget exhausted" in cand.ineligibility_reason:
            codes.append("RETRY_LIMIT_REACHED")
        elif "Payment method invalid" in cand.ineligibility_reason:
            codes.append("PAYMENT_METHOD_INVALID")
        elif "not safe to contact" in cand.ineligibility_reason:
            codes.append("OPT_OUT_OR_SAFETY")
        elif "frequency" in cand.ineligibility_reason:
            codes.append("CONTACT_FREQUENCY_LIMIT")
        elif "contact window" in cand.ineligibility_reason:
            codes.append("OUTSIDE_CONTACT_WINDOW")
        elif "not applicable" in cand.ineligibility_reason:
            codes.append("NOT_APPLICABLE")
        else:
            codes.append("INELIGIBLE")
    elif cand.strategy == "RETRY":
        codes += ["TEMPORARY_FAILURE"]
        codes.append("HIGH_RECOVERABILITY" if diagnosis.likely_recoverability == "HIGH" else "RECOVERABILITY_MEDIUM")
        if event.retry_count == 0:
            codes.append("NO_RETRY_LIMIT_BREACH")
    elif cand.strategy == "PAYMENT_LINK":
        if event.decline_code.value in ("expired_card", "incorrect_cvc"):
            codes += ["PAYMENT_METHOD_INVALID", "LINK_PREFERRED"]
        else:
            codes.append("PAYMENT_LINK_AVAILABLE")
    elif cand.strategy == "MESSAGE":
        codes.append("NUDGE_AVAILABLE")
    elif cand.strategy == "REAUTHORIZE":
        codes.append("AFA_COMPLIANT_ACTION" if diagnosis.requires_afa else "MANDATE_REAUTH")
    elif cand.strategy == "HUMAN_REVIEW":
        if diagnosis.confidence < 0.4:
            codes.append("LOW_CONFIDENCE")
        elif diagnosis.risk_score > 0.8:
            codes.append("HIGH_RISK")
        else:
            codes.append("SAFE_ESCALATION")
    return codes


def _decision_factors(event: RevenueEvent, diagnosis: DiagnosisOutput,
                      context: CustomerContextOutput,
                      ml_estimate: MLPrediction | None = None) -> list[str]:
    """Concise, structured decision inputs for the trace (no chain-of-thought)."""
    factors = [
        f"event_type={event.type.value}",
        f"diagnosis={diagnosis.classification}",
        f"recoverability={diagnosis.likely_recoverability}",
        f"confidence={diagnosis.confidence:.2f}",
        f"decline_code={event.decline_code.value}",
        f"retry_count={event.retry_count}",
        f"amount={event.amount}",
        f"safe_to_contact={context.safe_to_contact}",
        f"afa_required={diagnosis.requires_afa}",
    ]
    if ml_estimate is not None and ml_estimate.available:
        factors += [
            f"ml_probability={ml_estimate.recovery_probability:.4f}",
            f"ml_risk_band={ml_estimate.risk_band}",
            f"ml_prediction={ml_estimate.recovery_prediction}",
            f"ml_threshold={ml_estimate.threshold}",
            f"ml_source={ml_estimate.probability_source}",
        ]
    return factors


async def _apply_estimation(event, context, candidate: RecoveryCandidate,
                            ml_estimate: MLPrediction | None = None):
    """Estimate P(recovery).

    Authoritative source: the calibrated ML probability (shared model service)
    when an `ml_estimate` is available. Otherwise the deterministic rule-based
    estimator (cold prior + learned), which remains the documented fallback and
    benchmark path — never silently overrides the ML score.
    """
    if ml_estimate is not None and ml_estimate.available:
        candidate.probability = round(float(ml_estimate.recovery_probability), 4)
        candidate.probability_confidence = 0.9
        candidate.model_version = ml_estimate.model_version or MODEL_VERSION
        candidate.probability_source = ml_estimate.probability_source or "ml"
        candidate.from_learning = False
    else:
        estimator = get_probability_estimator()
        est = await estimator.estimate(event, context, candidate)
        base = BASE_PROBABILITIES.get(candidate.strategy, {}).get(
            event.decline_code.value, DEFAULT_PROB
        )
        candidate.probability = round(est.probability, 4)
        candidate.probability_confidence = est.confidence
        candidate.model_version = est.model_version or RULE_BASED_VERSION
        candidate.probability_source = RULE_BASED_VERSION
        candidate.from_learning = est.probability != base

    candidate.expected_value = int(event.amount * candidate.probability)

    # Outcome-informed stats: expose the raw historical n/m behind the estimate
    # for this (strategy, decline_code). Scoped to live outcomes only — batch
    # replay (txn_*) stays on cold priors and never surfaces learning data.
    candidate.empirical_attempts = 0
    candidate.empirical_successes = 0
    if not event.id.startswith("txn_"):
        try:
            from app.database import get_strategy_outcome_counts
            n, k = await get_strategy_outcome_counts(
                candidate.strategy, event.decline_code.value, source="live"
            )
            candidate.empirical_attempts = n
            candidate.empirical_successes = k
        except Exception:
            pass


def _score_objective(event: RevenueEvent, candidate: RecoveryCandidate) -> RecoveryObjective:
    """Score a candidate across risk / friction / cost / eligibility for ranking."""
    risk = 0.0
    friction = 0
    cost = 0
    if candidate.strategy == "RETRY":
        risk, friction, cost = 0.15, 0, 0
    elif candidate.strategy == "REAUTHORIZE":
        risk, friction, cost = 0.25, 2, 0
    elif candidate.strategy == "PAYMENT_LINK":
        risk, friction, cost = 0.05, 1, 2
    elif candidate.strategy == "MESSAGE":
        risk, friction, cost = 0.02, 0, 1
    elif candidate.strategy == "HUMAN_REVIEW":
        risk, friction, cost = 0.0, 10, 5

    candidate.risk_score = round(risk, 3)
    candidate.customer_friction = friction
    candidate.contact_cost = cost

    # Composite: ERV dominance, mildly penalized by risk/friction/cost.
    composite = max(candidate.expected_value - int(candidate.expected_value * risk) - cost * 100, 0)
    return RecoveryObjective(
        recovery_probability=candidate.probability,
        expected_recovery_value=candidate.expected_value,
        risk_score=candidate.risk_score,
        customer_friction=candidate.customer_friction,
        contact_cost=candidate.contact_cost,
        policy_eligible=candidate.policy_eligible,
        composite_score=float(composite),
        explanation=f"p={candidate.probability:.2f}, risk={candidate.risk_score:.2f}, friction={friction}, cost={cost}",
    )


def _candidate_to_strategy(event: RevenueEvent, cand: RecoveryCandidate) -> RecoveryStrategyOutput:
    return RecoveryStrategyOutput(
        strategy=cand.strategy,
        priority=cand.priority,
        reason=cand.reason or f"Elected by optimizer (EV ₹{cand.expected_value // 100:,})",
        proposed_delay_hours=0,
        channel=cand.channel,
        discount_percent=0,
        requires_human_approval=cand.requires_human_approval,
        expected_value=cand.expected_value,
    )


def score_candidates(candidates: list[RecoveryCandidate]) -> list[dict]:
    """Return each candidate's multi-factor objective, for SSE/analytics display."""
    ranked = []
    for i, c in enumerate(candidates, start=1):
        ranked.append({
            "rank": i,
            "strategy": c.strategy,
            "action": c.action,
            "channel": c.channel,
            "probability": c.probability,
            "expected_value": c.expected_value,
            "risk_score": c.risk_score,
            "customer_friction": c.customer_friction,
            "contact_cost": c.contact_cost,
            "policy_eligible": c.policy_eligible,
            "eligible": c.eligible,
            "ineligibility_reason": c.ineligibility_reason,
            "reason_codes": c.reason_codes,
            "confidence": c.probability_confidence,
            "model_version": c.model_version,
            "probability_source": c.probability_source,
            "empirical_attempts": c.empirical_attempts,
            "empirical_successes": c.empirical_successes,
        })
    return ranked


__all__ = [
    "build_optimizer_output", "OptimizerOutput", "RecoveryCandidate",
    "score_candidates",
]