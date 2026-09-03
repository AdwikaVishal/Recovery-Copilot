from datetime import datetime
from app.models import (
    RevenueEvent, DiagnosisOutput, ProposedAction,
    PolicyDecision, PolicyVerdict, PolicyCheckDetail
)
from app.config import PolicyConfig


class PolicyEngine:
    def __init__(self, config: PolicyConfig = None):
        self.config = config or PolicyConfig()

    def evaluate(
        self,
        event: RevenueEvent,
        diagnosis: DiagnosisOutput,
        proposed: ProposedAction,
        now: datetime = None,
    ) -> PolicyDecision:
        if now is None:
            now = datetime.utcnow()

        checks_passed = []
        checks_failed = []
        detailed_results = []

        all_checks = [
            self._check_opt_out,
            self._check_max_retries,
            self._check_cooling_period,
            self._check_afa_threshold,
            self._check_discount_ceiling,
            self._check_contact_frequency,
            self._check_time_of_day,
            self._check_amount_ceiling,
            self._check_risk_score,
            self._check_diagnosis_confidence,
        ]

        for check in all_checks:
            result = check(event, diagnosis, proposed, now)
            detail = PolicyCheckDetail(
                rule=result["name"],
                result="PASS" if result["passed"] else "FAIL",
                explanation=result["explanation"],
                regulatory_basis=result.get("regulatory_basis"),
            )
            detailed_results.append(detail)
            if result["passed"]:
                checks_passed.append(result["name"])
            else:
                checks_failed.append(result["name"])

        if not checks_failed:
            return PolicyDecision(
                verdict=PolicyVerdict.ALLOW,
                reason="All checks passed",
                checks_passed=checks_passed,
                checks_failed=[],
                detailed_results=detailed_results,
            )

        modify_candidates = {"discount_ceiling"}
        hard_failures = {"opt_out", "afa_check", "max_retries", "cooling_period", "contact_frequency"}
        failed_set = set(checks_failed)

        has_hard_failure = bool(failed_set & hard_failures)
        has_modify_candidate = bool(failed_set & modify_candidates)
        has_human_review = any(
            r.rule in ("risk_score", "amount_ceiling", "diagnosis_confidence")
            for r in detailed_results
            if r.result == "FAIL"
        )

        if has_hard_failure:
            verdict = PolicyVerdict.DENY
            reason = f"Hard failure: {', '.join(failed_set & hard_failures)}"
            return PolicyDecision(
                verdict=verdict,
                reason=reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                detailed_results=detailed_results,
                original_request=_action_to_dict(proposed),
            )

        if has_modify_candidate:
            modified = _apply_modifications(proposed, event, self.config)
            non_modify_failures = failed_set - modify_candidates
            if not non_modify_failures:
                return PolicyDecision(
                    verdict=PolicyVerdict.MODIFY,
                    reason=f"Modified action: {', '.join(failed_set & modify_candidates)}",
                    checks_passed=checks_passed,
                    checks_failed=checks_failed,
                    detailed_results=detailed_results,
                    modified_action=str(modified),
                    original_request=_action_to_dict(proposed),
                    modified_request=modified.model_dump(),
                )

        if has_human_review:
            return PolicyDecision(
                verdict=PolicyVerdict.HUMAN_REVIEW,
                reason=f"Requires human review: {', '.join(checks_failed)}",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                detailed_results=detailed_results,
                requires_human_approval=True,
                original_request=_action_to_dict(proposed),
            )

        return PolicyDecision(
            verdict=PolicyVerdict.DENY,
            reason=f"Failed: {', '.join(checks_failed)}",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            detailed_results=detailed_results,
            original_request=_action_to_dict(proposed),
        )

    def _check_opt_out(self, event, diagnosis, proposed, now):
        passed = not event.customer.opted_out
        if passed:
            explanation = f"Customer {event.customer.id} has not opted out of communications."
        else:
            explanation = f"Customer {event.customer.id} has opted out. Contact is prohibited under policy."
        return {"name": "opt_out", "passed": passed, "explanation": explanation, "regulatory_basis": "TRAI regulations on telemarketing; customer consent required under TCPA-equivalent provisions"}

    def _check_max_retries(self, event, diagnosis, proposed, now):
        max_r = diagnosis.max_retries if diagnosis.max_retries > 0 else self.config.max_retries
        passed = event.retry_count < max_r
        if passed:
            explanation = f"Retry count {event.retry_count}/{max_r} — within limit."
        else:
            explanation = f"Retry count {event.retry_count}/{max_r} — exceeded. Further retries require human approval."
        return {"name": "max_retries", "passed": passed, "explanation": explanation, "regulatory_basis": "RBI Fair Practices Code — lenders must not pursue recovery beyond reasonable frequency"}

    def _check_cooling_period(self, event, diagnosis, proposed, now):
        if event.last_attempt_at is None:
            return {"name": "cooling_period", "passed": True, "explanation": "No prior attempt recorded — cooling period N/A."}

        last = event.last_attempt_at
        if isinstance(last, str):
            last = datetime.fromisoformat(last)

        elapsed = (now - last).total_seconds() / 3600
        min_hours = max(self.config.min_cooling_hours, diagnosis.optimal_delay_hours)
        passed = elapsed >= min_hours
        if passed:
            explanation = f"Elapsed {elapsed:.1f}h since last attempt (minimum {min_hours}h) — cooling period satisfied."
        else:
            explanation = f"Elapsed {elapsed:.1f}h since last attempt (minimum {min_hours}h) — too soon to retry."
        return {"name": "cooling_period", "passed": passed, "explanation": explanation, "regulatory_basis": "RBI Ombudsman guidelines — minimum cooling period between recovery attempts"}

    def _check_afa_threshold(self, event, diagnosis, proposed, now):
        if not diagnosis.requires_afa:
            if event.amount > self.config.rbi_afa_threshold_paise:
                return {"name": "afa_check", "passed": True, "explanation": f"Amount ₹{event.amount // 100:,} exceeds AFA threshold ₹{self.config.rbi_afa_threshold_paise // 100:,} but AFA not required for this payment type.", "regulatory_basis": "RBI e-Mandate Framework (Jul 2025) — Additional Factor Authentication mandatory for recurring debits >₹15,000"}
            return {"name": "afa_check", "passed": True, "explanation": f"Amount ₹{event.amount // 100:,} below RBI AFA threshold ₹{self.config.rbi_afa_threshold_paise // 100:,} — no AFA required.", "regulatory_basis": "RBI e-Mandate Framework (Jul 2025) — Additional Factor Authentication mandatory for recurring debits >₹15,000"}

        if proposed.action == "re_authorize_mandate":
            return {"name": "afa_check", "passed": True, "explanation": f"RBI AFA required for ₹{event.amount // 100:,}. Action is re_authorize_mandate — compliant.", "regulatory_basis": "RBI e-Mandate Framework (Jul 2025) — Additional Factor Authentication mandatory for recurring debits >₹15,000"}

        return {"name": "afa_check", "passed": False, "explanation": f"RBI AFA required for ₹{event.amount // 100:,} (threshold ₹{self.config.rbi_afa_threshold_paise // 100:,}). Action '{proposed.action}' is not AFA-compliant.", "regulatory_basis": "RBI e-Mandate Framework (Jul 2025) — Additional Factor Authentication mandatory for recurring debits >₹15,000"}

    def _check_discount_ceiling(self, event, diagnosis, proposed, now):
        if proposed.discount_percent == 0:
            return {"name": "discount_ceiling", "passed": True, "explanation": "No discount offered — within ceiling.", "regulatory_basis": "Internal policy — merchant-funded discount ceiling to prevent margin erosion"}
        passed = proposed.discount_percent <= self.config.max_discount_percent
        if passed:
            explanation = f"Discount {proposed.discount_percent}% within ceiling {self.config.max_discount_percent}%."
        else:
            explanation = f"Discount {proposed.discount_percent}% exceeds ceiling {self.config.max_discount_percent}% — will be reduced to {self.config.max_discount_percent}%."
        return {"name": "discount_ceiling", "passed": passed, "explanation": explanation, "regulatory_basis": "Internal policy — merchant-funded discount ceiling to prevent margin erosion"}

    def _check_contact_frequency(self, event, diagnosis, proposed, now):
        max_week = self.config.max_contacts_per_week
        return {"name": "contact_frequency", "passed": True, "explanation": f"Weekly contact limit {max_week} — checked by Customer Context Agent before reaching Policy Engine.", "regulatory_basis": "TRAI DND / RBI consumer protection — frequency caps on financial communications"}

    def _check_time_of_day(self, event, diagnosis, proposed, now):
        try:
            hour = now.hour
            start = int(self.config.contact_window_start.split(":")[0])
            end = int(self.config.contact_window_end.split(":")[0])
            passed = start <= hour < end
            if passed:
                explanation = f"Current hour {hour}:00 within allowed window {self.config.contact_window_start}–{self.config.contact_window_end}."
            else:
                explanation = f"Current hour {hour}:00 outside allowed window {self.config.contact_window_start}–{self.config.contact_window_end}. Contact deferred."
        except Exception:
            passed = True
            explanation = "Contact window config invalid — defaulting to ALLOW."
        return {"name": "time_of_day", "passed": passed, "explanation": explanation, "regulatory_basis": "TRAI telemarketing regulations — communications restricted to 08:00–21:00 IST"}

    def _check_amount_ceiling(self, event, diagnosis, proposed, now):
        ceiling = self.config.get("escalation_threshold_paise", 500000)
        if event.amount > ceiling and proposed.discount_percent > 0:
            return {"name": "amount_ceiling", "passed": False, "explanation": f"High-value ₹{event.amount // 100:,} (>{ceiling // 100:,}) with discount {proposed.discount_percent}% — requires human approval.", "regulatory_basis": "Internal policy — high-value transactions with discount require human approval"}
        return {"name": "amount_ceiling", "passed": True, "explanation": f"Amount ₹{event.amount // 100:,} — ceiling applies only when discount is offered on high-value transactions (>{ceiling // 100:,}).", "regulatory_basis": "Internal policy — high-value transactions with discount require human approval"}

    def _check_risk_score(self, event, diagnosis, proposed, now):
        if diagnosis.risk_score > 0.8:
            return {"name": "risk_score", "passed": False, "explanation": f"Risk score {diagnosis.risk_score:.2f} exceeds threshold 0.80 — requires human review.", "regulatory_basis": "Internal policy — high-risk cases (>0.80 score) require human review before execution"}
        return {"name": "risk_score", "passed": True, "explanation": f"Risk score {diagnosis.risk_score:.2f} within threshold 0.80.", "regulatory_basis": "Internal policy — high-risk cases (>0.80 score) require human review before execution"}

    def _check_diagnosis_confidence(self, event, diagnosis, proposed, now):
        if diagnosis.confidence < 0.7:
            return {"name": "diagnosis_confidence", "passed": False, "explanation": f"Diagnosis confidence {diagnosis.confidence:.0%} below 70% threshold — requires human review.", "regulatory_basis": "Internal policy — low-confidence diagnoses must be reviewed by human operator"}
        return {"name": "diagnosis_confidence", "passed": True, "explanation": f"Diagnosis confidence {diagnosis.confidence:.0%} within acceptable range.", "regulatory_basis": "Internal policy — low-confidence diagnoses must be reviewed by human operator"}


def _action_to_dict(proposed: ProposedAction) -> dict:
    return {
        "action": proposed.action,
        "channel": proposed.channel,
        "amount": proposed.amount,
        "discount_percent": proposed.discount_percent,
    }


def _apply_modifications(proposed: ProposedAction, event: RevenueEvent, config: PolicyConfig) -> ProposedAction:
    modified = proposed.model_copy(deep=True)

    if proposed.discount_percent > config.max_discount_percent:
        modified.discount_percent = config.max_discount_percent

    ceiling = config.get("escalation_threshold_paise", 500000)
    if event.amount > ceiling and modified.discount_percent > 0:
        modified.discount_percent = 0

    return modified
