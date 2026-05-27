"""
pipeline/train.py
Train RandomForest, XGBoost, LightGBM + IsolationForest.
"""
import json
import logging
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

from pipeline.feature_engineering import LABEL_BINARY
from config import XGB_MODEL_PATH, LGB_MODEL_PATH

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains RF, XGBoost, LightGBM classifiers and IsolationForest."""

    def __init__(
        self,
        n_estimators: int = 100,
        test_size: float = 0.2,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.contamination = contamination
        self.random_state = random_state
        self.rf_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.iso_model = None
        self.feature_names_: list[str] = []
        self.X_test_ = None
        self.y_test_ = None

    def run(
        self,
        df: pd.DataFrame,
        rf_model_path: Path,
        iso_model_path: Path,
        feature_names_path: Path,
        xgb_model_path: Path | None = None,
        lgb_model_path: Path | None = None,
        emit=None,
    ) -> dict:
        def log(msg):
            logger.info(msg)
            if emit:
                emit(msg)

        feature_cols = [c for c in df.columns if c != LABEL_BINARY]
        X = df[feature_cols].values
        y = df[LABEL_BINARY].values
        self.feature_names_ = feature_cols

        log(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        self.X_test_ = X_test
        self.y_test_ = y_test

        model_metrics = {}

        # RandomForest
        log(f"Training RandomForest ({self.n_estimators} trees) ...")
        t0 = time.time()
        self.rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            n_jobs=-1,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.rf_model.fit(X_train, y_train)
        log(f"  RandomForest trained in {time.time() - t0:.1f}s")
        joblib.dump(self.rf_model, rf_model_path)
        model_metrics["random_forest"] = self._evaluate_classifier(self.rf_model, X_test, y_test, log)

        # XGBoost
        if xgb_model_path:
            try:
                from xgboost import XGBClassifier
                log("Training XGBoost ...")
                t0 = time.time()
                self.xgb_model = XGBClassifier(
                    n_estimators=self.n_estimators,
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                self.xgb_model.fit(X_train, y_train)
                log(f"  XGBoost trained in {time.time() - t0:.1f}s")
                joblib.dump(self.xgb_model, xgb_model_path)
                model_metrics["xgboost"] = self._evaluate_classifier(self.xgb_model, X_test, y_test, log)
            except ImportError:
                log("  XGBoost not installed — skipping")

        # LightGBM
        if lgb_model_path:
            try:
                from lightgbm import LGBMClassifier
                log("Training LightGBM ...")
                t0 = time.time()
                self.lgb_model = LGBMClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
                self.lgb_model.fit(X_train, y_train)
                log(f"  LightGBM trained in {time.time() - t0:.1f}s")
                joblib.dump(self.lgb_model, lgb_model_path)
                model_metrics["lightgbm"] = self._evaluate_classifier(self.lgb_model, X_test, y_test, log)
            except ImportError:
                log("  LightGBM not installed — skipping")

        # IsolationForest
        log(f"Training IsolationForest (contamination={self.contamination}) ...")
        t0 = time.time()
        iso_sample = min(50_000, len(X_train))
        idx = np.random.RandomState(self.random_state).choice(len(X_train), iso_sample, replace=False)
        self.iso_model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.iso_model.fit(X_train[idx])
        log(f"  IsolationForest trained in {time.time() - t0:.1f}s")
        joblib.dump(self.iso_model, iso_model_path)

        Path(feature_names_path).write_text(json.dumps(self.feature_names_))

        comparison = {k: dict(v) for k, v in model_metrics.items()}
        primary = dict(model_metrics.get("random_forest", {}))
        primary["model_comparison"] = comparison
        return primary

    def _evaluate_classifier(self, model, X_test, y_test, log) -> dict:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        log(f"  Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc:.4f}")

        return {
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
            "roc_auc": round(auc, 6),
            "confusion_matrix": cm,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "test_samples": int(len(y_test)),
        }
