"""
pipeline/feature_selection.py
Select top N features using Mutual Information.
Saves selected feature names to disk so all other modules use the same set.
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif

from pipeline.feature_engineering import LABEL_BINARY

logger = logging.getLogger(__name__)

# Default number of top features
DEFAULT_TOP_N = 15


class FeatureSelector:
    """Selects top-N features using Mutual Information scoring."""

    def __init__(self, top_n: int = DEFAULT_TOP_N):
        self.top_n = top_n
        self.scores_: pd.Series | None = None
        self.selected_features_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, emit=None) -> list[str]:
        """
        Compute MI scores and select top_n features.
        Returns list of selected feature names.
        """
        def log(msg):
            logger.info(msg)
            if emit:
                emit(msg)

        X = df.drop(columns=[LABEL_BINARY])
        y = df[LABEL_BINARY]

        log(f"Feature selection: computing Mutual Information for {X.shape[1]} features ...")
        log(f"  (This may take 1–3 minutes for large datasets)")

        # Subsample for speed if dataset is very large
        if len(X) > 200_000:
            sample_idx = X.sample(200_000, random_state=42).index
            X_sample = X.loc[sample_idx]
            y_sample = y.loc[sample_idx]
            log(f"  Subsampled to 200,000 rows for MI computation speed")
        else:
            X_sample, y_sample = X, y

        mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
        self.scores_ = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        # Select top N
        self.selected_features_ = list(self.scores_.head(self.top_n).index)

        # Log ranking
        log(f"  Top {self.top_n} features selected (by Mutual Information):")
        for i, (feat, score) in enumerate(self.scores_.head(self.top_n).items(), 1):
            log(f"    {i:2d}. {feat:<35s}  MI={score:.4f}")

        reduction = (1 - self.top_n / X.shape[1]) * 100
        log(f"  Dimensionality reduction: {X.shape[1]} → {self.top_n} features ({reduction:.1f}% reduction)")

        return self.selected_features_

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection to a DataFrame (keeps label_binary if present)."""
        if not self.selected_features_:
            raise RuntimeError("Call fit() before transform()")
        keep = self.selected_features_.copy()
        if LABEL_BINARY in df.columns:
            keep = [LABEL_BINARY] + keep
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")
        return df[keep]

    def save(self, path: Path):
        """Persist selected feature names + MI scores as JSON."""
        path = Path(path)
        data = {
            "selected_features": self.selected_features_,
            "top_n": self.top_n,
            "mi_scores": {k: float(v) for k, v in self.scores_.items()},
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Feature selection results saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "FeatureSelector":
        """Restore a FeatureSelector from a saved JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        obj = cls(top_n=data["top_n"])
        obj.selected_features_ = data["selected_features"]
        obj.scores_ = pd.Series(data["mi_scores"])
        return obj
