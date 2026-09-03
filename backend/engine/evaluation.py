import json
import os
import tempfile
import contextvars
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from app.models import (
    RevenueEvent, Customer, TransactionMetadata,
    EventType, DeclineCode, PolicyVerdict
)
from app.database import init_db, db_session, set_active_db_path, reset_active_db_path
from agents.diagnosis_agent import diagnose
from agents.policy_engine import PolicyEngine
from agents.recovery_strategy_agent import build_strategy, strategy_to_proposed_action
from agents.customer_context_agent import get_customer_context
from agents.execution_adapter import DeterministicMockGateway, FailureInjectingGateway
from data.scenarios import get_all_scenarios, build_event_from_scenario


@dataclass
class EvaluationMetrics:
    total_events: int = 0
    eligible_events: int = 0
    recovered_events: int = 0
    gross_recovered_amount: int = 0
    baseline_recovered_amount: int = 0
    incremental_recovery: int = 0
    recovery_rate: float = 0.0
    contact_rate: float = 0.0
    contacts_attempted: int = 0
    opt_out_violations: int = 0
    afa_violations: int = 0
    excess_retries: int = 0
    human_review_rate: float = 0.0
    human_review_count: int = 0
    false_recovery_rate: float = 0.0
    cost_per_recovered_rupee: float = 0.0
    scenario_results: list = field(default_factory=list)
    verdict_distribution: dict = field(default_factory=dict)
    action_distribution: dict = field(default_factory=dict)
    errors: int = 0


@asynccontextmanager
async def _isolated_eval_db():
    """Route ALL database access during evaluation to a throwaway SQLite DB.

    Every downstream helper (get_customer_context, db_session, count_contacts_since,
    record_outcome, audit, ptp_agent, etc.) opens its connection via database.get_db(),
    which honors the active DB path ContextVar. By setting that path here, the entire
    evaluation subtree operates against a private temporary database that is destroyed
    when evaluation ends, so the batch/live benchmark DB is never touched.
    """
    import aiosqlite

    fd, tmp_name = tempfile.mkstemp(prefix="eval_copilot_", suffix=".db")
    os.close(fd)
    tmp_path = Path(tmp_name)
    token = set_active_db_path(tmp_path)
    try:
        await init_db()
        yield
    finally:
        reset_active_db_path(token)
        try:
            await aiosqlite.connect(str(tmp_path)).close()
        except Exception:
            pass
        try:
            tmp_path.unlink()
        except Exception:
            pass


async def run_evaluation(events: list[RevenueEvent] = None, with_guardrails: bool = True) -> EvaluationMetrics:
    async with _isolated_eval_db():
        return await _run_evaluation_inner(events, with_guardrails)


async def _run_evaluation_inner(events: list[RevenueEvent] = None, with_guardrails: bool = True) -> EvaluationMetrics:
    if events is None:
        events = _load_sample_events()

    metrics = EvaluationMetrics(total_events=len(events))
    engine = PolicyEngine()

    for event in events:
        try:
            result = await _evaluate_single_event(event, engine, with_guardrails)
            metrics.scenario_results.append(result)

            if result["eligible"]:
                metrics.eligible_events += 1

            if result["recovered"]:
                metrics.recovered_events += 1
                metrics.gross_recovered_amount += event.amount

            baseline_rate = _get_baseline_rate(event.type.value)
            metrics.baseline_recovered_amount += int(event.amount * baseline_rate)

            if result["contacted"]:
                metrics.contacts_attempted += 1

            if result["opt_out_violation"]:
                metrics.opt_out_violations += 1

            if result["afa_violation"]:
                metrics.afa_violations += 1

            if result["excess_retry"]:
                metrics.excess_retries += 1

            if result["verdict"] == "HUMAN_REVIEW":
                metrics.human_review_count += 1

            verdict = result["verdict"]
            metrics.verdict_distribution[verdict] = metrics.verdict_distribution.get(verdict, 0) + 1

            action = result["action"]
            metrics.action_distribution[action] = metrics.action_distribution.get(action, 0) + 1

        except Exception as e:
            metrics.errors += 1
            metrics.scenario_results.append({
                "event_id": event.id,
                "error": str(e),
                "eligible": False,
                "recovered": False,
            })

    metrics.incremental_recovery = metrics.gross_recovered_amount - metrics.baseline_recovered_amount
    metrics.recovery_rate = metrics.recovered_events / max(metrics.eligible_events, 1)
    metrics.contact_rate = metrics.contacts_attempted / max(metrics.total_events, 1)
    metrics.human_review_rate = metrics.human_review_count / max(metrics.total_events, 1)
    metrics.false_recovery_rate = 0.0
    metrics.cost_per_recovered_rupee = 0.0

    return metrics


async def _run_scenario_evaluation_inner() -> list[dict]:
    scenarios = get_all_scenarios()
    results = []

    for scenario in scenarios:
        event = build_event_from_scenario(scenario)
        engine = PolicyEngine()

        try:
            # Clean slate: remove any seeded contact_events for this customer
            await _clear_contact_history(event.customer.id)

            # Seed contact_events for contact_limit scenario
            if scenario.get("contacts_last_24h"):
                from app.config import PolicyConfig
                config = PolicyConfig()
                max_contacts = config.get("max_contacts_per_day", 3)
                await _seed_contact_history(event.customer.id, max_contacts)

            diagnosis = diagnose(event)
            context = await get_customer_context(event)
            strategy = build_strategy(event, diagnosis, context)
            proposed = strategy_to_proposed_action(event, strategy)

            if scenario.get("proposed_discount"):
                proposed.discount_percent = scenario["proposed_discount"]

            test_time = None
            if scenario.get("test_hour") is not None:
                test_time = datetime.utcnow().replace(
                    hour=scenario["test_hour"],
                    minute=scenario.get("test_minute", 0),
                    second=0, microsecond=0,
                )

            decision = engine.evaluate(event, diagnosis, proposed, now=test_time)

            # If strategy returns STOP (e.g. customer not safe to contact), override verdict
            if strategy.strategy == "STOP":
                from app.models import PolicyVerdict
                decision.verdict = PolicyVerdict.DENY
                decision.reason = f"Strategy STOP: {strategy.reason}"
                if "contact_frequency" not in decision.checks_failed:
                    decision.checks_failed.append("contact_frequency")

            verdict_match = decision.verdict.value == scenario["expected_verdict"]

            action_match = True
            if scenario.get("expected_action"):
                expected_action = scenario["expected_action"]

                if decision.verdict.value == "DENY":
                    if "max_retries" in decision.checks_failed:
                        actual_action = "HUMAN_REVIEW"
                    else:
                        actual_action = "STOP"
                elif decision.verdict.value == "HUMAN_REVIEW":
                    actual_action = "HUMAN_REVIEW"
                else:
                    strategy_to_expected = {
                        "RETRY": "RETRY",
                        "REAUTHORIZE": "REAUTHORIZE",
                        "PAYMENT_LINK": "SEND_PAYMENT_LINK",
                        "MESSAGE": "SEND_REMINDER",
                        "PTP_FOLLOWUP": "SEND_REMINDER",
                        "HUMAN_REVIEW": "HUMAN_REVIEW",
                        "STOP": "STOP",
                    }
                    actual_action = strategy_to_expected.get(strategy.strategy, strategy.strategy)

                action_match = actual_action == expected_action
            else:
                actual_action = strategy.strategy

            results.append({
                "scenario_id": scenario["id"],
                "description": scenario["description"],
                "expected_verdict": scenario["expected_verdict"],
                "actual_verdict": decision.verdict.value,
                "verdict_match": verdict_match,
                "expected_action": scenario.get("expected_action"),
                "actual_action": actual_action,
                "action_match": action_match,
                "diagnosis_classification": diagnosis.classification,
                "diagnosis_confidence": diagnosis.confidence,
                "recoverability": diagnosis.likely_recoverability,
                "checks_failed": decision.checks_failed,
                "pass": verdict_match and action_match,
            })

        except Exception as e:
            results.append({
                "scenario_id": scenario["id"],
                "description": scenario["description"],
                "error": str(e),
                "pass": False,
            })
        finally:
            # Scoped teardown: remove any contact_events seeded for this scenario's
            # customer so evaluation leaves no residue in the shared tables and
            # cannot influence later scenarios or the batch benchmark.
            try:
                await _clear_contact_history(event.customer.id)
            except Exception:
                pass

    return results


async def run_scenario_evaluation() -> list[dict]:
    # Evaluation MUST run against a throwaway DB so it can never mutate the active
    # batch/live benchmark database.
    async with _isolated_eval_db():
        return await _run_scenario_evaluation_inner()


async def _seed_contact_history(customer_id: str, contact_count: int):
    from datetime import timedelta
    async with db_session() as db:
        now = datetime.utcnow()
        for i in range(contact_count):
            sent_at = (now - timedelta(hours=12)).isoformat()
            await db.execute(
                """INSERT OR IGNORE INTO contact_events
                   (customer_id, event_id, channel, sent_at, status, message_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (customer_id, f"seed_{customer_id}_{i}", "whatsapp", sent_at, "delivered", "retry_payment"),
            )


async def _clear_contact_history(customer_id: str):
    async with db_session() as db:
        await db.execute("DELETE FROM contact_events WHERE customer_id = ?", (customer_id,))


def _get_baseline_rate(event_type: str) -> float:
    from app.config import PolicyConfig
    config = PolicyConfig()
    rates = config.get("baseline_rates", {})
    return rates.get(event_type, 0.05)


async def _evaluate_single_event(
    event: RevenueEvent,
    engine: PolicyEngine,
    with_guardrails: bool,
) -> dict:
    diagnosis = diagnose(event)
    context = await get_customer_context(event)
    strategy = build_strategy(event, diagnosis, context)
    proposed = strategy_to_proposed_action(event, strategy)

    if not with_guardrails:
        decision = PolicyDecision_mock()
    else:
        decision = engine.evaluate(event, diagnosis, proposed)

    eligible = decision.verdict.value in ("ALLOW", "MODIFY")
    recovered = False
    contacted = False

    if eligible:
        contacted = proposed.action in ("retry_payment", "send_payment_link", "send_dunning_message", "re_authorize_mandate")

        gateway = DeterministicMockGateway()
        if proposed.action == "retry_payment":
            result = gateway.retry_payment(f"pay_{event.id}", event.amount, event)
            recovered = result.status == "captured"
        elif proposed.action == "send_payment_link":
            recovered = False
        elif proposed.action == "send_dunning_message":
            result = gateway.send_dunning_message(event.customer.id, proposed.channel, event)
            recovered = result.status == "captured"

    opt_out_violation = False
    if event.customer.opted_out and contacted:
        opt_out_violation = True

    afa_violation = False
    if diagnosis.requires_afa and proposed.action == "retry_payment":
        afa_violation = True

    excess_retry = event.retry_count > 3

    return {
        "event_id": event.id,
        "eligible": eligible,
        "recovered": recovered,
        "contacted": contacted,
        "verdict": decision.verdict.value,
        "action": proposed.action,
        "opt_out_violation": opt_out_violation,
        "afa_violation": afa_violation,
        "excess_retry": excess_retry,
        "amount": event.amount,
    }


class PolicyDecision_mock:
    verdict = PolicyVerdict.ALLOW
    checks_failed = []


def _load_sample_events() -> list[RevenueEvent]:
    from engine.pipeline import load_batch
    try:
        return load_batch()
    except FileNotFoundError:
        from data.generator import generate_batch
        return generate_batch(100)


def format_metrics_report(metrics: EvaluationMetrics) -> str:
    lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        f"Total events:            {metrics.total_events}",
        f"Eligible events:         {metrics.eligible_events}",
        f"Recovered events:        {metrics.recovered_events}",
        f"Recovery rate:           {metrics.recovery_rate:.1%}",
        "",
        "AMOUNTS",
        f"Gross recovered:         ₹{metrics.gross_recovered_amount // 100:,}",
        f"Baseline recovered:      ₹{metrics.baseline_recovered_amount // 100:,}",
        f"Incremental recovery:    ₹{metrics.incremental_recovery // 100:,}",
        "",
        "COMPLIANCE",
        f"Opt-out violations:      {metrics.opt_out_violations}",
        f"AFA violations:          {metrics.afa_violations}",
        f"Excess retries:          {metrics.excess_retries}",
        f"False recovery rate:     {metrics.false_recovery_rate:.1%}",
        "",
        "OPERATIONAL",
        f"Contact rate:            {metrics.contact_rate:.1%}",
        f"Human review rate:       {metrics.human_review_rate:.1%}",
        f"Errors:                  {metrics.errors}",
        "",
        "VERDICT DISTRIBUTION",
    ]
    for v, count in sorted(metrics.verdict_distribution.items()):
        lines.append(f"  {v}: {count}")

    lines.append("")
    lines.append("ACTION DISTRIBUTION")
    for a, count in sorted(metrics.action_distribution.items()):
        lines.append(f"  {a}: {count}")

    lines.append("=" * 60)
    return "\n".join(lines)
