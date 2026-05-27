"""
detection/anomaly_detector.py
Wraps the trained IsolationForest model to flag potential zero-day threats.
"""
import logging
import joblib
import pandas as pd
from pathlib import Path
from config import ISOLATION_MODEL_PATH

logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self):
        self.model = None

    def load_model(self):
        if not ISOLATION_MODEL_PATH.exists():
            raise FileNotFoundError("IsolationForest model not trained yet.")
        if self.model is None:
            self.model = joblib.load(ISOLATION_MODEL_PATH)

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Returns anomaly flags (-1 means anomaly, 1 means normal).
        We map -1 -> True (is_anomaly), 1 -> False.
        """
        self.load_model()
        
        if df.empty:
            return {"is_anomaly": [], "anomaly_scores": []}

        # Isolation forest returns 1 for inliers, -1 for outliers
        preds = self.model.predict(df.values)
        
        # Decision function: lower values mean more anomalous
        scores = self.model.decision_function(df.values)

        is_anomaly = [bool(p == -1) for p in preds]

        return {
            "is_anomaly": is_anomaly,
            "anomaly_scores": scores.tolist(),
        }
