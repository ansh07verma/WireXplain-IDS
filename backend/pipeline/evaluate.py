"""
pipeline/evaluate.py
Load trained models and run evaluation on saved test data.
Also generates feature importance ranking.
"""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_feature_names(path: Path) -> list[str]:
    """Load the list of selected feature names from JSON."""
    return json.loads(Path(path).read_text())


def feature_importance(rf_model, feature_names: list[str], top_n: int = 15) -> list[dict]:
    """
    Return top-N feature importances from a trained RandomForest.
    Returns list of {feature, importance, rank}.
    """
    importances = rf_model.feature_importances_
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {"rank": i + 1, "feature": feat, "importance": round(float(imp), 6)}
        for i, (feat, imp) in enumerate(ranked[:top_n])
    ]


def model_summary(rf_model_path: Path, feature_names_path: Path) -> dict:
    """
    Load model and return a summary dict for the frontend.
    """
    if not Path(rf_model_path).exists():
        return {"trained": False}

    model = joblib.load(rf_model_path)
    feature_names = load_feature_names(feature_names_path)

    return {
        "trained": True,
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_importance": feature_importance(model, feature_names),
    }
