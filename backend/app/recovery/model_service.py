"""Recovery model service — loads frozen artifacts once and serves predictions.

Production contract (see docs/recovery_model.md):

    transaction
        -> 44 raw features          (app.recovery.feature_builder)
        -> 49 encoded features      (this module: ref_date conversion + imputer + one-hot)
        -> ExtraTreesClassifier     (final_model.joblib)
        -> raw probability
        -> sigmoid calibrator       (sigmoid_calibrator.joblib)
        -> calibrated probability
        -> threshold                (0.04, from final_metrics.json)
        -> risk decision

The service NEVER retrains, never refits preprocessing, and never executes the
research notebook. All objects are loaded lazily once and reused, so the
102 MB ExtraTrees model is read from disk a single time per process.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from app.recovery.feature_builder import ALL_FEATURES

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "recovery_model_artifacts"

CATEGORICAL = ["payment_method"]
NUMERIC = [f for f in ALL_FEATURES if f not in CATEGORICAL]

RISK_BANDS = [
    (0.01, "<1%"),
    (0.02, "1-2%"),
    (0.03, "2-3%"),
    (0.04, "3-4%"),
    (0.05, "4-5%"),
    (0.075, "5-7.5%"),
    (0.10, "7.5-10%"),
    (np.inf, ">=10%"),
]


def risk_band(probability: float) -> str:
    for upper, label in RISK_BANDS:
        if probability < upper:
            return label
    return ">=10%"


def risk_label(probability: float) -> str:
    """Application-level risk string derived from the calibrated probability."""
    if probability < 0.02:
        return "Very Low"
    if probability < 0.04:
        return "Low"
    if probability < 0.075:
        return "Medium"
    if probability < 0.10:
        return "High"
    return "Very High"


class RecoveryModelService:
    """Holds every trained artifact and exposes vocabulary-consistent inference."""

    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        # Populated lazily on first access (model file is ~100 MB).
        self._loaded = False
        self._lock = threading.Lock()

        self.model = None
        self.imputer = None
        self.encoder = None
        self.calibrator = None
        self.feature_lists: dict[str, Any] = {}
        self.model_feature_names: list[str] = []
        self.ref_date: pd.Timestamp = pd.Timestamp("2023-01-01")
        self.metrics: dict[str, Any] = {}
        self.selected_threshold: float = 0.04
        self.feature_importance: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "RecoveryModelService":
        with self._lock:
            if self._loaded:
                return self
            try:
                import warnings
                import sklearn as _sk
                _artifact_version = "1.8.0"
                _runtime_version = _sk.__version__
                if _runtime_version != _artifact_version:
                    warnings.warn(
                        f"scikit-learn runtime {_runtime_version} differs from "
                        f"training version {_artifact_version}. Artifacts are NOT "
                        f"retrained. Inference is expected safe for minor patches "
                        f"but results should be validated.",
                        UserWarning, stacklevel=2,
                    )
                self.model = joblib.load(self.artifact_dir / "final_model.joblib")
                self.imputer = joblib.load(self.artifact_dir / "imputer.joblib")
                self.encoder = joblib.load(self.artifact_dir / "encoder.joblib")
                self.calibrator = joblib.load(self.artifact_dir / "sigmoid_calibrator.joblib")

                with open(self.artifact_dir / "feature_lists.json") as f:
                    self.feature_lists = json.load(f)
                with open(self.artifact_dir / "model_feature_names.json") as f:
                    self.model_feature_names = json.load(f)
                with open(self.artifact_dir / "ref_date.json") as f:
                    self.ref_date = pd.to_datetime(json.load(f)["ref_date"])
                with open(self.artifact_dir / "final_metrics.json") as f:
                    self.metrics = json.load(f)

                self.selected_threshold = float(self.metrics.get("selected_threshold", 0.04))

                importance_csv = self.artifact_dir / "feature_importance.csv"
                if importance_csv.exists():
                    imp = pd.read_csv(importance_csv)
                    self.feature_importance = (
                        imp.sort_values("importance", ascending=False)
                           .head(20).to_dict("records"))

                # Hard invariants from the artifact audit — fail loudly rather
                # than silently serving a mismatched pipeline.
                assert len(self.feature_lists.get("ALL_FEATURES", [])) == 44
                assert len(self.model_feature_names) == 49
                assert self.model.n_features_in_ == 49
                assert len(self.imputer.statistics_) == 43
                expected_cats = [f"payment_method_{c}" for c in self.encoder.categories_[0]]
                assert self.model_feature_names[43:] == expected_cats
                self._loaded = True
            except Exception:
                self._loaded = False
                raise
            return self

    def ensure_loaded(self) -> "RecoveryModelService":
        if not self._loaded:
            return self.load()
        return self

    # ------------------------------------------------------------------
    # Preprocessing (deterministic replay of notebook cell 5)
    # ------------------------------------------------------------------

    def encode(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Map a DataFrame of 44 raw features to the 49 model features.

        ``previous_recovery_date`` datetime is converted to days since the
        saved training reference date BEFORE imputation (exact notebook order).
        """
        self.ensure_loaded()
        assert list(raw_df.columns) == ALL_FEATURES, "Raw frame must be in ALL_FEATURES order"
        assert raw_df.shape[1] == 44

        conv = raw_df.copy()
        dt = pd.to_datetime(conv["previous_recovery_date"])
        conv["previous_recovery_date"] = (dt - self.ref_date).dt.total_seconds() / 86400

        num = self.imputer.transform(conv[NUMERIC])
        num_df = pd.DataFrame(num, columns=NUMERIC, index=conv.index)
        cat = self.encoder.transform(conv[CATEGORICAL])
        cat_df = pd.DataFrame(
            cat, columns=list(self.encoder.get_feature_names_out(CATEGORICAL)),
            index=conv.index,
        )
        encoded = pd.concat([num_df, cat_df], axis=1)
        assert encoded.shape[1] == 49
        assert list(encoded.columns) == self.model_feature_names
        return encoded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, encoded: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (raw_probability, calibrated_probability) arrays in [0, 1]."""
        self.ensure_loaded()
        raw = self.model.predict_proba(encoded)[:, 1]
        calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return raw, calibrated

    def decide(self, calibrated: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        return (calibrated >= self.selected_threshold).astype(int)

    def explain(self, raw_df: pd.DataFrame, top: int = 5) -> list[dict]:
        """Lightweight, static-importance explanation for a single raw row.

        Singleton caller: always pass a 1-row frame. Uses the saved impurity
        feature-importance artifact — no per-request SHAP.
        """
        self.ensure_loaded()
        if not self.feature_importance or raw_df.shape[0] != 1:
            return []
        row = raw_df.iloc[0]
        out: list[dict] = []
        seen: set[str] = set()
        for item in self.feature_importance:
            encoded_name = item["feature"]
            raw_name = (encoded_name[len("payment_method_"):]
                        if encoded_name.startswith("payment_method_")
                        else encoded_name)
            if raw_name not in row.index or raw_name in seen:
                continue
            value = row[raw_name]
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif pd.isna(value) and not isinstance(value, str):
                value = None
            out.append({
                "feature": raw_name,
                "encoded_feature": encoded_name,
                "importance": float(item.get("importance", 0.0)),
                "value": (value.isoformat() if value is not None
                          and not isinstance(value, (int, float, str))
                          else value),
            })
            seen.add(raw_name)
            if len(out) >= top:
                break
        return out

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def model_info(self) -> dict[str, Any]:
        self.ensure_loaded()
        return {
            "model": "ExtraTreesClassifier",
            "model_artifact": "final_model.joblib",
            "estimators": 500,
            "raw_features": 44,
            "encoded_features": 49,
            "target": "recovered_72h",
            "recovery_rate_training": self.metrics.get("baseline_recovery_rate"),
            "calibration": "Sigmoid (LogisticRegression, fit on validation only)",
            "calibrator_artifact": "sigmoid_calibrator.joblib",
            "threshold": self.selected_threshold,
            "threshold_artifact": "final_metrics.json (selected_threshold)",
            "imputer_artifact": "imputer.joblib",
            "encoder_artifact": "encoder.joblib",
            "ref_date_artifact": "ref_date.json",
            "feature_group_breakdown": {
                "base": len(self.feature_lists.get("BASE_FEATURES", [])),
                "historical": len(self.feature_lists.get("HISTORICAL_FEATURES", [])),
                "behavioral": len(self.feature_lists.get("BEHAVIORAL_FEATURES", [])),
                "advanced": len(self.feature_lists.get("ADVANCED_FEATURES", [])),
                "temporal": len(self.feature_lists.get("TEMPORAL_FEATURES", [])),
                "recovery_history": len(self.feature_lists.get("RECOVERY_HISTORY_FEATURES", [])),
            },
            "test_metrics": {
                "roc_auc": self.metrics.get("test_roc_auc_calibrated"),
                "pr_auc": self.metrics.get("test_pr_auc_calibrated"),
                "brier": self.metrics.get("test_brier_calibrated"),
                "logloss": self.metrics.get("test_logloss_calibrated"),
                "precision": self.metrics.get("test_precision"),
                "recall": self.metrics.get("test_recall"),
                "f1": self.metrics.get("test_f1"),
                "lift": self.metrics.get("lift"),
            },
        }

    def features_spec(self) -> dict[str, Any]:
        self.ensure_loaded()
        return {
            "raw_feature_count": 44,
            "encoded_feature_count": 49,
            "categorical": list(CATEGORICAL),
            "numeric": list(NUMERIC),
            "groups": {
                "base": self.feature_lists.get("BASE_FEATURES", []),
                "historical": self.feature_lists.get("HISTORICAL_FEATURES", []),
                "behavioral": self.feature_lists.get("BEHAVIORAL_FEATURES", []),
                "advanced": self.feature_lists.get("ADVANCED_FEATURES", []),
                "temporal": self.feature_lists.get("TEMPORAL_FEATURES", []),
                "recovery_history": self.feature_lists.get("RECOVERY_HISTORY_FEATURES", []),
            },
            "encoded_feature_names": self.model_feature_names,
        }


_service: Optional[RecoveryModelService] = None
_service_lock = threading.Lock()


def get_model_service() -> RecoveryModelService:
    """Process-wide singleton: artifacts load once, then every request reuses them."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = RecoveryModelService().load()
    return _service


__all__ = [
    "RecoveryModelService", "get_model_service", "risk_band", "risk_label",
    "ARTIFACT_DIR", "NUMERIC", "CATEGORICAL",
]