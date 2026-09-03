"""Construction of the 44 RAW model features, reproducing the training notebook.

This module replicates the exact, leakage-safe feature engineering from
``recovery_model_final.ipynb`` (cells 2–4 and 7). The model is defined on
44 raw features grouped as Base(6) + Historical(10) + Behavioral(9) +
Advanced(5) + Temporal(6) + Recovery History(8).

Only information available BEFORE the current transaction is ever used:
cumulative counts use ``shift(1)``, rolling windows use ``closed="left"`` and
streaks are "streak_before", exactly like training. The current transaction's
own row contributes its base/temporal values and everything else is derived
from the caller-supplied PRIOR transaction history.

The reader is expected to already understand the notebook; this module is its
production twin, not a re-derivation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.recovery.schemas import RecoveryHistoryEvent, RecoveryPredictRequest

RECOVERY_STATUSES = ("refunded", "cancelled", "pending")

# ---------------------------------------------------------------------------
# Feature group definitions — kept identical to feature_lists.json
# ---------------------------------------------------------------------------

BASE_FEATURES = ["quantity", "unit_price", "total_amount", "discount_applied",
                 "shipping_cost", "payment_method"]
HISTORICAL_FEATURES = [
    "historical_transactions", "historical_completed", "historical_failed",
    "historical_refunded", "historical_cancelled", "historical_success_rate",
    "historical_failure_rate", "historical_total_spend",
    "historical_avg_transaction", "customer_tenure_days",
]
BEHAVIORAL_FEATURES = [
    "transactions_last_7d", "completed_last_30d", "failed_last_30d",
    "failure_streak_before", "success_streak_before", "failed_last_7d",
    "transactions_last_30d", "transactions_last_90d", "spend_last_30d",
]
ADVANCED_FEATURES = [
    "days_since_previous_transaction", "days_since_previous_success",
    "days_since_previous_failure", "previous_avg_amount", "amount_vs_previous_avg",
]
TEMPORAL_FEATURES = [
    "transaction_year", "transaction_month", "transaction_day",
    "transaction_dayofweek", "transaction_hour", "is_weekend",
]
RECOVERY_HISTORY_FEATURES = [
    "customer_transactions_before", "customer_recoveries_before",
    "customer_recovery_rate_before", "previous_recovery_date",
    "days_since_previous_recovery", "recovery_streak_before",
    "recoveries_last_30d", "recoveries_last_90d",
]

ALL_FEATURES = (BASE_FEATURES + HISTORICAL_FEATURES + BEHAVIORAL_FEATURES
                + ADVANCED_FEATURES + TEMPORAL_FEATURES + RECOVERY_HISTORY_FEATURES)


def _to_frame(request: RecoveryPredictRequest) -> pd.DataFrame:
    """Build the per-customer ordered transaction frame.

    The frame is the customer's prior transactions plus the current transaction
    as the final row, sorted exactly like the notebook
    (customer_id, transaction_date, transaction_id).
    """
    rows: list[dict[str, Any]] = []
    for h in request.history or []:
        rows.append({
            "transaction_id": h.transaction_id or f"hist_{len(rows)}",
            "customer_id": request.customer_id,
            "transaction_date": pd.to_datetime(h.transaction_date),
            "quantity": float(h.quantity or 0),
            "unit_price": float(h.unit_price or 0),
            "total_amount": float(h.total_amount or 0),
            "discount_applied": float(h.discount_applied or 0),
            "shipping_cost": float(h.shipping_cost or 0),
            "payment_method": h.payment_method or "",
            "status": h.status,
            "recovered_72h": int(h.recovered_72h or 0),
        })
    rows.append({
        "transaction_id": request.transaction_id,
        "customer_id": request.customer_id,
        "transaction_date": pd.to_datetime(request.transaction_date),
        "quantity": float(request.quantity or 0),
        "unit_price": float(request.unit_price or 0),
        "total_amount": float(request.total_amount or 0),
        "discount_applied": float(request.discount_applied or 0),
        "shipping_cost": float(request.shipping_cost or 0),
        "payment_method": request.payment_method,
        "status": request.status,
        "recovered_72h": np.nan,  # target placeholder; never a model input
    })
    df = pd.DataFrame(rows)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return (df.sort_values(["transaction_date", "transaction_id"])
              .reset_index(drop=True))


def _streak_before(values: np.ndarray) -> np.ndarray:
    """Count of consecutive 1s immediately BEFORE each position (notebook logic)."""
    out = np.zeros(len(values), dtype=int)
    streak = 0
    for i, value in enumerate(values):
        out[i] = streak
        if value:
            streak += 1
        else:
            streak = 0
    return out


def _base_and_historical(df: pd.DataFrame, signup_date: pd.Timestamp) -> pd.DataFrame:
    """Cells 2a/2b — base columns carried through + cumulative historical features."""
    out = df.copy()

    out["previous_avg_amount"] = out.groupby("customer_id")["total_amount"].shift(1)
    out["amount_vs_previous_avg"] = np.where(
        out["previous_avg_amount"] > 0,
        out["total_amount"] / out["previous_avg_amount"],
        1.0,
    )

    out["historical_transactions"] = out.groupby("customer_id", sort=False).cumcount()
    out["_completed"] = out["status"].eq("completed").astype(int)
    out["_failed"] = out["status"].eq("failed").astype(int)
    out["_refunded"] = out["status"].eq("refunded").astype(int)
    out["_cancelled"] = out["status"].eq("cancelled").astype(int)

    out["historical_completed"] = out.groupby("customer_id")["_completed"].cumsum().shift(1).fillna(0)
    out["historical_failed"] = out.groupby("customer_id")["_failed"].cumsum().shift(1).fillna(0)
    out["historical_refunded"] = out.groupby("customer_id")["_refunded"].cumsum().shift(1).fillna(0)
    out["historical_cancelled"] = out.groupby("customer_id")["_cancelled"].cumsum().shift(1).fillna(0)

    out["_historical_amount"] = out.groupby("customer_id")["total_amount"].cumsum().shift(1).fillna(0)
    out["historical_total_spend"] = out["_historical_amount"]

    out["historical_avg_transaction"] = np.where(
        out["historical_transactions"] > 0,
        out["historical_total_spend"] / out["historical_transactions"], 0,
    )
    out["historical_success_rate"] = np.where(
        out["historical_transactions"] > 0,
        out["historical_completed"] / out["historical_transactions"], 0,
    )
    out["historical_failure_rate"] = np.where(
        out["historical_transactions"] > 0,
        (out["historical_failed"] + out["historical_refunded"] + out["historical_cancelled"])
        / out["historical_transactions"], 0,
    )

    tenure = (
        (out["transaction_date"] - pd.to_datetime(signup_date)).dt.total_seconds() / 86400
        if signup_date is not None and not pd.isna(signup_date)
        else np.nan
    )
    out["customer_tenure_days"] = tenure

    out = out.drop(columns=["_completed", "_failed", "_refunded", "_cancelled",
                            "_historical_amount"])
    return out


def _behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 2c — rolling-window counts and streak_before features (leakage-safe)."""
    work = df.copy()
    work["_row"] = np.arange(len(work))
    work["_one"] = 1
    work["_completed"] = work["status"].eq("completed").astype(int)
    work["_failed"] = work["status"].isin(["failed", "cancelled", "refunded"]).astype(int)

    def _roll(frame: pd.DataFrame, values_col: str, window_days: str,
              feature: str) -> pd.DataFrame:
        rolled = (
            frame.set_index("transaction_date").groupby("customer_id")[values_col]
            .rolling(window_days, closed="left").sum().reset_index()
            .rename(columns={values_col: feature})
        )
        # closed="left" makes every same-timestamp row see the identical
        # prior-window sum, so duplicate (customer, date) keys can be collapsed
        # before joining. Without this, two duplicated keys merge many-to-many
        # and the cross-product explodes (see tests/test_recovery_model.py).
        rolled = rolled.drop_duplicates(["customer_id", "transaction_date"],
                                        keep="last")
        return frame.merge(rolled, on=["customer_id", "transaction_date"],
                           how="left")

    for days, feature in [(7, "transactions_last_7d"), (30, "transactions_last_30d"),
                          (90, "transactions_last_90d")]:
        work = _roll(work, "_one", f"{days}D", feature)
    for source, feature in [("_completed", "completed_last_30d"),
                            ("_failed", "failed_last_30d")]:
        work = _roll(work, source, "30D", feature)
    work = _roll(work, "_failed", "7D", "failed_last_7d")
    work = _roll(work, "total_amount", "30D", "spend_last_30d")

    work["success_streak_before"] = work.groupby("customer_id")["_completed"].transform(
        lambda x: _streak_before(x.to_numpy()))
    work["failure_streak_before"] = work.groupby("customer_id")["_failed"].transform(
        lambda x: _streak_before(x.to_numpy()))

    work = work.sort_values("_row").reset_index(drop=True)
    work = work.drop(columns=["_row", "_one", "_completed", "_failed"])
    for feature in BEHAVIORAL_FEATURES:
        work[feature] = work[feature].fillna(0)
    return work


def _advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 2d — authoritative final pass for recency + momentum features."""
    out = df.copy()
    out = out.sort_values(["customer_id", "transaction_date", "transaction_id"]).reset_index(drop=True)

    out["previous_transaction_date"] = out.groupby("customer_id")["transaction_date"].shift(1)
    out["days_since_previous_transaction"] = (
        out["transaction_date"] - out["previous_transaction_date"]
    ).dt.total_seconds() / 86400

    success_dates = out["transaction_date"].where(out["status"].eq("completed"))
    out["previous_success_date"] = success_dates.groupby(out["customer_id"]).ffill().shift(1)
    out["days_since_previous_success"] = (
        out["transaction_date"] - out["previous_success_date"]
    ).dt.total_seconds() / 86400

    failed_dates = out["transaction_date"].where(
        out["status"].isin(["cancelled", "refunded"]))
    out["previous_failure_date"] = failed_dates.groupby(out["customer_id"]).ffill().shift(1)
    out["days_since_previous_failure"] = (
        out["transaction_date"] - out["previous_failure_date"]
    ).dt.total_seconds() / 86400

    out["previous_avg_amount"] = out.groupby("customer_id")["total_amount"].transform(
        lambda x: x.shift(1).expanding().mean())
    out["amount_vs_previous_avg"] = out["total_amount"] / out["previous_avg_amount"].replace(0, np.nan)

    for f in ["days_since_previous_transaction", "days_since_previous_success",
              "days_since_previous_failure"]:
        out[f] = out[f].fillna(0)
    out["previous_avg_amount"] = out["previous_avg_amount"].fillna(out["total_amount"])
    out["amount_vs_previous_avg"] = (out["amount_vs_previous_avg"]
                                     .replace([np.inf, -np.inf], np.nan).fillna(1.0))
    return out


def _temporal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["transaction_date"].dt
    out["transaction_year"] = dt.year
    out["transaction_month"] = dt.month
    out["transaction_day"] = dt.day
    out["transaction_dayofweek"] = dt.dayofweek
    out["transaction_hour"] = dt.hour
    out["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    return out


def _recovery_history(df: pd.DataFrame, current_date: pd.Timestamp,
                      request_transaction_id: str) -> dict[str, Any]:
    """Cell 3b — cumulative/recovery-history features, strictly prior rows only.

    Only the customer's PRIOR recovery-population transactions (refunded /
    cancelled / pending) feed these features; the current row is the reference
    point and is never counted. ``recovered_72h`` for prior rows is supplied by
    the caller (0 when unknown/conservative).
    """
    recovery_rows = df[df["status"].isin(RECOVERY_STATUSES)]
    # Keep only rows STRICTLY before the current transaction.
    strictly_before = recovery_rows[
        (recovery_rows["transaction_date"] < current_date)
        | ((recovery_rows["transaction_date"] == current_date)
           & (recovery_rows["transaction_id"] != request_transaction_id))
    ].reset_index(drop=True)

    n_before = len(strictly_before)
    recovered_series = strictly_before["recovered_72h"] if n_before else pd.Series(dtype=int)

    recoveries_before = int(recovered_series.sum()) if n_before else 0
    recovery_rate = (recoveries_before / n_before) if n_before else 0.0

    previous_recovery_date = pd.NaT
    if n_before:
        rec_dates = strictly_before.loc[recovered_series.eq(1), "transaction_date"]
        if len(rec_dates):
            previous_recovery_date = rec_dates.iloc[-1]

    days_since_previous_recovery = (
        (current_date - previous_recovery_date).total_seconds() / 86400
        if pd.notna(previous_recovery_date) else np.nan
    )

    streak = 0
    if n_before:
        vals = recovered_series.to_numpy(dtype=int)
        for v in vals[::-1]:
            if v == 1:
                streak += 1
            else:
                break

    if n_before:
        p30 = pd.Timestamp(current_date) - pd.Timedelta(days=30)
        p90 = pd.Timestamp(current_date) - pd.Timedelta(days=90)
        prev_dates = strictly_before.loc[recovered_series.eq(1), "transaction_date"]
        recoveries_last_30d = int(((prev_dates >= p30)).sum()) if len(prev_dates) else 0
        recoveries_last_90d = int(((prev_dates >= p90)).sum()) if len(prev_dates) else 0
    else:
        recoveries_last_30d = 0
        recoveries_last_90d = 0

    return {
        "customer_transactions_before": n_before,
        "customer_recoveries_before": recoveries_before,
        "customer_recovery_rate_before": float(recovery_rate),
        "previous_recovery_date": previous_recovery_date,
        "days_since_previous_recovery": days_since_previous_recovery,
        "recovery_streak_before": streak,
        "recoveries_last_30d": recoveries_last_30d,
        "recoveries_last_90d": recoveries_last_90d,
    }


def _current_id(df: pd.DataFrame) -> str:
    """Return the current transaction id (the last row in the frame)."""
    return df["transaction_id"].iloc[-1]


def build_raw_features(request: RecoveryPredictRequest) -> dict[str, Any]:
    """Compute the 44 raw features for one prediction request.

    Returns a dict keyed by the exact 44 feature names from
    ``recovery_model_artifacts/feature_lists.json``. ``previous_recovery_date``
    is kept as a datetime (NaT when there is no prior recovery) so the model
    service can apply the notebook's exact datetime→numeric conversion before
    imputation.
    """
    frame = _to_frame(request)
    current_date = pd.to_datetime(request.transaction_date)

    signup = pd.to_datetime(request.customer_signup_date) if request.customer_signup_date else None

    enriched = _temporal(_advanced(_behavioral(_base_and_historical(frame, signup))))
    current = enriched.iloc[-1]

    base_row = {
        "quantity": float(current["quantity"]),
        "unit_price": float(current["unit_price"]),
        "total_amount": float(current["total_amount"]),
        "discount_applied": float(current["discount_applied"]),
        "shipping_cost": float(current["shipping_cost"]),
        "payment_method": current["payment_method"],
    }

    historical_row = {f: float(current[f]) for f in HISTORICAL_FEATURES}
    behavioral_row = {f: float(current[f]) for f in BEHAVIORAL_FEATURES}
    advanced_row = {f: float(current[f]) for f in ADVANCED_FEATURES
                    if f in current.index}
    temporal_row = {f: float(current[f]) for f in TEMPORAL_FEATURES}

    recovery_row = _recovery_history(frame, current_date, request.transaction_id)

    raw = {}
    raw.update(base_row)
    raw.update(historical_row)
    raw.update(behavioral_row)
    raw.update(advanced_row)
    raw.update(temporal_row)
    raw.update(recovery_row)

    assert sorted(raw.keys()) == sorted(ALL_FEATURES), (
        f"Feature set mismatch: got {sorted(raw.keys())}, expected {sorted(ALL_FEATURES)}")
    return raw