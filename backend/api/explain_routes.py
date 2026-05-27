from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd
import io
import logging
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from explainability.shap_explainer import SHAPExplainer
from pipeline.feature_engineering import FeatureEngineer
from pipeline.feature_selection import FeatureSelector
from config import MODELS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Explainability"])

_explainer = None
_feature_engineer = None
_feature_selector = None

def get_explainer_and_pipeline():
    global _explainer, _feature_engineer, _feature_selector
    if not _explainer:
        _explainer = SHAPExplainer()
    if not _feature_engineer:
        _feature_engineer = FeatureEngineer()
    if not _feature_selector:
        selector_path = MODELS_DIR / "feature_selector.json"
        if not selector_path.exists():
            raise FileNotFoundError("Feature selector not trained.")
        _feature_selector = FeatureSelector.load(selector_path)
    return _explainer, _feature_engineer, _feature_selector

@router.post("/global")
async def explain_global(file: UploadFile = File(...)):
    """
    Accepts a CSV of network flows, processes them, and returns global feature importance.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if "Label" not in df.columns:
            df["Label"] = "BENIGN"
            
        explainer, engineer, selector = get_explainer_and_pipeline()
        
        # Take a sample if dataset is too large to compute SHAP quickly
        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
            
        df_eng = engineer.run(df)
        df_sel = selector.transform(df_eng)
        
        if "label_binary" in df_sel.columns:
            df_sel = df_sel.drop(columns=["label_binary"])
            
        result = explainer.explain_global(df_sel)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Global explain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/local")
async def explain_local(file: UploadFile = File(...), row_index: int = Form(0)):
    """
    Accepts a CSV and an index, explains that specific flow's prediction.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if row_index < 0 or row_index >= len(df):
            raise HTTPException(status_code=400, detail="row_index out of bounds")
            
        # Extract just the row we want to explain
        row_df = df.iloc[[row_index]].copy()
        
        if "Label" not in row_df.columns:
            row_df["Label"] = "BENIGN"
            
        explainer, engineer, selector = get_explainer_and_pipeline()
        
        df_eng = engineer.run(row_df)
        df_sel = selector.transform(df_eng)
        
        if "label_binary" in df_sel.columns:
            df_sel = df_sel.drop(columns=["label_binary"])
            
        # Let's assume we want to explain it as an attack (class 1)
        # We could also run prediction first to know what class to explain
        from detection.ml_detector import MLDetector
        ml = MLDetector()
        preds = ml.predict(df_sel)
        pred_class = int(preds["labels"][0])
        
        result = explainer.explain_local(df_sel, predicted_class=pred_class)
        result["predicted_class"] = pred_class
        result["ml_confidence"] = float(preds["probabilities"][0])
        
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Local explain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
