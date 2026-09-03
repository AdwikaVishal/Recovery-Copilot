from app.models import RevenueEvent, PolicyDecision, SupervisorOutput


CRITIC_RULES = [
    {
        "id": "retry_exhausted",
        "description": "Retrying after max retries already exhausted",
        "check": lambda event, decision, supervisor: (
            supervisor.proposed_action and
            supervisor.proposed_action.action == "retry_payment" and
            event.retry_count >= 3
        ),
        "objection": "Retrying payment but customer has already failed 3 times — historically <2% succeed after exhaustion. Consider HUMAN_REVIEW or STOP.",
    },
    {
        "id": "opted_out_message",
        "description": "Sending message to opted-out customer",
        "check": lambda event, decision, supervisor: (
            event.customer.opted_out and
            supervisor.proposed_action and
            supervisor.proposed_action.action in ("send_dunning_message", "send_reminder")
        ),
        "objection": "Customer has opted out of communications. Sending a message violates their preference and RBI Fair Practices Code.",
    },
    {
        "id": "dnd_registered_contact",
        "description": "Contacting DND-registered customer",
        "check": lambda event, decision, supervisor: (
            event.customer.dnd_registered and
            supervisor.proposed_action and
            supervisor.proposed_action.channel in ("sms", "voice") and
            supervisor.proposed_action.action != "retry_payment"
        ),
        "objection": "Customer is DND-registered. SMS/voice contact violates TRAI regulations unless transactional.",
    },
    {
        "id": "high_value_no_afa",
        "description": "Retrying high-value mandate without AFA",
        "check": lambda event, decision, supervisor: (
            event.amount >= 1500000 and
            event.type.value == "recurring_payment_failure" and
            supervisor.proposed_action and
            supervisor.proposed_action.action == "retry_payment"
        ),
        "objection": "High-value recurring mandate (≥₹15,000) requires fresh AFA per RBI e-Mandate Framework. Blind retry violates regulation.",
    },
    {
        "id": "discount_over_ceiling",
        "description": "Offering discount above policy ceiling",
        "check": lambda event, decision, supervisor: (
            supervisor.proposed_action and
            supervisor.proposed_action.discount_percent > 10
        ),
        "objection": f"Discount offered exceeds 10% policy ceiling. This requires explicit human approval.",
    },
    {
        "id": "outside_contact_window",
        "description": "Contacting outside allowed hours",
        "check": lambda event, decision, supervisor: (
            supervisor.proposed_action and
            supervisor.proposed_action.action in ("send_dunning_message", "send_reminder") and
            hasattr(event.failed_at, 'hour') and
            (event.failed_at.hour < 8 or event.failed_at.hour >= 21)
        ),
        "objection": "Contact attempt outside 08:00–21:00 window. RBI Fair Practices Code requires communication only during business hours.",
    },
    {
        "id": "blind_retry_network",
        "description": "Retrying on network timeout without waiting",
        "check": lambda event, decision, supervisor: (
            event.decline_code.value == "network_timeout" and
            supervisor.proposed_action and
            supervisor.proposed_action.action == "retry_payment" and
            supervisor.proposed_action.proposed_delay_hours == 0
        ),
        "objection": "Network timeout typically resolves within minutes. Immediate retry may waste an attempt. Consider short delay.",
    },
    {
        "id": "amount_very_high",
        "description": "Pursuing very high-value transaction without escalation",
        "check": lambda event, decision, supervisor: (
            event.amount >= 5000000 and
            supervisor.proposed_action and
            supervisor.proposed_action.action != "escalate_to_human" and
            supervisor.workflow_status.value != "STOPPED"
        ),
        "objection": "Transaction value is very high (≥₹50,000). Recommend human escalation.",
    },
]


def run_critic(
    event: RevenueEvent,
    decision: PolicyDecision,
    supervisor: SupervisorOutput,
) -> list[dict]:
    objections = []
    for rule in CRITIC_RULES:
        try:
            if rule["check"](event, decision, supervisor):
                objections.append({
                    "rule_id": rule["id"],
                    "description": rule["description"],
                    "objection": rule["objection"],
                })
        except Exception:
            pass
    return objections
