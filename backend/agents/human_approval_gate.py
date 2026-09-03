from app.models import RevenueEvent, ProposedAction, PolicyDecision, PolicyVerdict


class HumanApprovalGate:
    THRESHOLDS = {
        "max_discount_percent": 10,
        "high_value_paise": 500000,
        "max_retries_for_escalation": 3,
    }

    def evaluate(
        self,
        event: RevenueEvent,
        proposed: ProposedAction,
        policy_decision: PolicyDecision,
    ) -> dict:
        reasons = []

        if proposed.discount_percent > self.THRESHOLDS["max_discount_percent"]:
            reasons.append(f"Discount {proposed.discount_percent}% exceeds {self.THRESHOLDS['max_discount_percent']}% cap")

        if event.amount > self.THRESHOLDS["high_value_paise"] and proposed.discount_percent > 0:
            reasons.append(f"High-value transaction (₹{event.amount // 100:,}) with discount needs approval")

        if policy_decision.requires_human_approval:
            reasons.append("Policy engine flagged for human review")

        if event.retry_count >= self.THRESHOLDS["max_retries_for_escalation"]:
            reasons.append(f"Retries exhausted ({event.retry_count} attempts)")

        needs_approval = len(reasons) > 0

        return {
            "needs_approval": needs_approval,
            "reasons": reasons,
            "can_auto_proceed": not needs_approval,
            "escalation_path": "account_manager" if event.amount > self.THRESHOLDS["high_value_paise"] else "support_lead",
        }
