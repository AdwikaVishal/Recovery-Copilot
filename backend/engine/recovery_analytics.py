"""Recovery Analytics / learning loop.

Aggregates historical strategy outcomes into effectiveness tables and exposes the
"what worked?" view. The learned empirical probabilities are consumed by the
Recovery Optimizer to refine Expected Recovery Value over time (closed loop).
"""
from app.database import get_strategy_effectiveness, db_session


def calculate_baseline(items) -> int:
    """Single canonical counterfactual baseline.

    Sums the PER-EVENT truncated counterfactual value (int(amount * rate)), so
    every consumer — batch pipeline, dashboard hero, summary, evaluation — yields
    byte-identical numbers regardless of grouping or storage shape. (Summing
    per-event truncations is NOT the same as truncating the per-category sum, which
    is how the 9-paise hero-vs-benchmark drift was introduced.)

    Accepts RevenueEvent-like objects or dicts with `amount` and `decline_code`.
    """
    from data.scenarios import COUNTERFACTUAL_RATES
    total = 0
    for item in items:
        if isinstance(item, dict):
            amount = item.get("amount") or 0
            dc = item.get("decline_code") or ""
        else:
            amount = getattr(item, "amount", 0) or 0
            dc = getattr(item, "decline_code", "") or ""
            if hasattr(dc, "value"):
                dc = dc.value
        rate = COUNTERFACTUAL_RATES.get(dc, 0.05)
        total += int(amount * rate)
    return total


async def strategy_effectiveness_report() -> dict:
    rows = await get_strategy_effectiveness()
    total_attempts = sum(r["attempts"] for r in rows)
    total_successes = sum(r["successes"] for r in rows)
    total_recovery = sum(r["recovery_amount"] for r in rows)
    overall = (total_successes / total_attempts) if total_attempts else 0.0

    # Best strategy by empirical recovery amount (non-empty).
    best = max(rows, key=lambda r: r["recovery_amount"]) if rows else None

    return {
        "strategies": rows,
        "overall": {
            "attempts": total_attempts,
            "successes": total_successes,
            "recovery_amount": total_recovery,
            "success_rate": round(overall, 4),
        },
        "best_strategy": best["strategy"] if best and best["attempts"] > 0 else None,
    }


async def recovery_optimizer_insights(batch_id: str = None) -> list[dict]:
    """Per-candidate selection trace from the most recent executed outcomes."""
    async with db_session() as db:
        cursor = await db.execute(
            """SELECT event_id, strategy, action, success, recovered_amount,
                      probability, expected_value, executed_at
               FROM strategy_outcomes
               ORDER BY executed_at DESC LIMIT 50"""
        )
        rows = [dict(r) for r in await cursor.fetchall()]
    return rows


async def scoring_comparison_report(batch_id: str = None) -> dict:
    """Honest A/B: what the scorer EXPECTED vs what execution ACTUALLY recovered.

    Compares the expected recovery value recorded for each executed attempt
    (EV = P(recovery) x amount at decision time) against the real recovered
    amount from the outcome ledger. This is a calibration view of the scoring
    layer over the Policy-Engine-authorized subset of attempts.

    Honesty guard: this never claims the ranked-scorer layer lifted results.
    Batch evaluation elects the deterministic rule-based primary by design, so
    optimizer rankings are decision-visible but did not alter batch outputs.
    """
    async with db_session() as db:
        cursor = await db.execute(
            """SELECT strategy, COUNT(*) AS attempts,
                      SUM(CASE WHEN success THEN 1 ELSE 0 END) AS successes,
                      SUM(expected_value) AS expected_paise,
                      SUM(recovered_amount) AS recovered_paise
               FROM strategy_outcomes
               GROUP BY strategy
               ORDER BY recovered_paise DESC"""
        )
        rows = [dict(r) for r in await cursor.fetchall()]

    def _cash(paise: int | None) -> str:
        return f"₹{int(paise or 0) // 100:,}"

    per_strategy = []
    for r in rows:
        expected = int(r["expected_paise"] or 0)
        actual = int(r["recovered_paise"] or 0)
        per_strategy.append({
            "strategy": r["strategy"],
            "attempts": r["attempts"],
            "successes": r["successes"],
            "success_rate": round((r["successes"] / r["attempts"]) if r["attempts"] else 0.0, 4),
            "expected": expected,
            "expected_display": _cash(expected),
            "actual": actual,
            "actual_display": _cash(actual),
            "surplus": actual - expected,
            "surplus_display": _cash(actual - expected),
        })

    total_expected = sum(r["expected_paise"] or 0 for r in rows)
    total_actual = sum(r["recovered_paise"] or 0 for r in rows)

    return {
        "per_strategy": per_strategy,
        "totals": {
            "attempts": sum(r["attempts"] for r in rows),
            "expected": total_expected,
            "expected_display": _cash(total_expected),
            "actual": total_actual,
            "actual_display": _cash(total_actual),
            "surplus": total_actual - total_expected,
            "surplus_display": _cash(total_actual - total_expected),
        },
        "honesty": (
            "Expected vs actual compares the scoring layer against the "
            "Policy-Engine-authorized outcomes ledger. On the canonical batch "
            "benchmark the deterministic rule-based primary is elected by design, "
            "so the ranked scorer is decision-visible but did NOT change batch "
            "results — no uplift is claimed from synthetic ranking data."
        ),
    }
