"""
detection/ml_detector.py
Wraps trained ML classifiers (RF, XGBoost, LightGBM).
"""
import logging
import joblib
import pandas as pd
from pathlib import Path
from config import RF_MODEL_PATH, XGB_MODEL_PATH, LGB_MODEL_PATH, MODEL_TYPE

logger = logging.getLogger(__name__)

MODEL_PATHS = {
    "rf": RF_MODEL_PATH,
    "xgb": XGB_MODEL_PATH,
    "lgb": LGB_MODEL_PATH,
}


class MLDetector:
    def __init__(self, model_type: str | None = None):
        self.model_type = (model_type or MODEL_TYPE).lower()
        self.model = None

    def _resolve_path(self) -> Path:
        path = MODEL_PATHS.get(self.model_type, RF_MODEL_PATH)
        if not path.exists():
            if self.model_type != "rf" and RF_MODEL_PATH.exists():
                logger.warning(f"{self.model_type} model missing, falling back to RF")
                return RF_MODEL_PATH
            raise FileNotFoundError(
                f"Model '{self.model_type}' not trained. Run the pipeline first."
            )
        return path

    def load_model(self):
        path = self._resolve_path()
        if self.model is None:
            self.model = joblib.load(path)
            logger.info(f"Loaded ML model: {path.name}")

    def predict(self, df: pd.DataFrame) -> dict:
        self.load_model()

        if df.empty:
            return {"labels": [], "probabilities": []}

        labels = self.model.predict(df.values)
        probas = self.model.predict_proba(df.values)
        max_probas = probas.max(axis=1)

        return {
            "labels": labels.tolist(),
            "probabilities": max_probas.tolist(),
            "model_type": self.model_type,
        }
