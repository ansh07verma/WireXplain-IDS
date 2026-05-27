import joblib
import pandas as pd
import numpy as np
import shap
import logging
from pathlib import Path
from config import RF_MODEL_PATH, FEATURE_NAMES_PATH
import json

logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None

    def load_model(self):
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError("RandomForest model not trained yet.")
        if not FEATURE_NAMES_PATH.exists():
            raise FileNotFoundError("Feature names not found.")

        if self.model is None:
            self.model = joblib.load(RF_MODEL_PATH)
            with open(FEATURE_NAMES_PATH, 'r') as f:
                self.feature_names = json.load(f)
            
            # Use TreeExplainer for RandomForest
            # Since Random Forest can be large, we might want to approximate or just use it directly
            self.explainer = shap.TreeExplainer(self.model)

    def explain_local(self, row_df: pd.DataFrame, predicted_class: int = 1) -> dict:
        """
        Explain a single prediction.
        row_df: DataFrame with exactly one row (the processed features).
        """
        self.load_model()
        if row_df.empty:
            return {"contributions": [], "explanation": "No data provided."}

        # Calculate SHAP values for the single row
        shap_values = self.explainer.shap_values(row_df)
        
        # For sklearn RandomForest, shap_values is a list of arrays (one for each class)
        # Or a single array if binary classification depending on shap version.
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class][0]
        elif len(shap_values.shape) == 3:
            # shape (num_samples, num_features, num_classes)
            sv = shap_values[0, :, predicted_class]
        else:
            sv = shap_values[0]

        # Combine feature names with their SHAP values and original feature values
        row_values = row_df.iloc[0].values
        contributions = []
        for i, f_name in enumerate(self.feature_names):
            contributions.append({
                "feature": f_name,
                "value": float(row_values[i]),
                "contribution": float(sv[i])
            })
        
        # Sort by absolute contribution to find the most impactful features
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_features = contributions[:5]
        
        # Generate Natural Language Explanation
        if predicted_class == 1: # Attack
            explanation = "This flow was classified as an Attack because "
            reasons = []
            for feat in top_features:
                if feat["contribution"] > 0:
                    reasons.append(f"the value of `{feat['feature']}` ({feat['value']:.2f}) strongly increased the risk score")
            if reasons:
                explanation += ", ".join(reasons) + "."
            else:
                explanation += "multiple features combined slightly to cross the threshold."
        else:
            explanation = "This flow was classified as Normal. "
            top_neg = [f for f in top_features if f["contribution"] < 0]
            if top_neg:
                explanation += f"Features like `{top_neg[0]['feature']}` kept it within expected baseline behavior."
            else:
                explanation += "No single feature indicated malicious activity."

        return {
            "contributions": contributions,
            "top_features": top_features,
            "explanation": explanation
        }

    def explain_global(self, sample_df: pd.DataFrame, predicted_class: int = 1) -> dict:
        """
        Explain global feature importance for a sample dataset.
        sample_df: DataFrame containing the processed features.
        """
        self.load_model()
        if sample_df.empty:
            return {"feature_importance": []}

        # Calculate SHAP values for the sample
        shap_values = self.explainer.shap_values(sample_df)
        
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, predicted_class]
        else:
            sv = shap_values

        # Mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(sv).mean(axis=0)
        
        importance = []
        for i, f_name in enumerate(self.feature_names):
            importance.append({
                "feature": f_name,
                "importance": float(mean_abs_shap[i])
            })
            
        importance.sort(key=lambda x: x["importance"], reverse=True)
        return {"feature_importance": importance}
