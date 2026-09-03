"""Recovery ML model — production inference package.

This package loads the trained ExtraTreesClassifier artifacts (produced by
recovery_model_final.ipynb) and serves calibrated recovery probabilities for a
single transaction or in batch.

The research notebook stays the training/evaluation source of truth; the backend
never executes the notebook and never refits the preprocessing. It only replays
the exact trained preprocessing (imputer + one-hot encoder + sigmoid calibrator)
on top of a leakage-safe feature construction.
"""