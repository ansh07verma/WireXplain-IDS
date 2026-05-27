"""
api/detection_routes.py
API endpoints for hybrid detection on uploaded CSV files.
"""
import logging
import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.hybrid import detect_flows

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Detection"])


@router.post("/csv")
async def detect_csv(file: UploadFile = File(...)):
    """Accepts a CSV upload and runs hybrid detection (ML + anomaly + signatures)."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()

        report = detect_flows(df, log_alerts=True, enrich_intel=True)
        return {"filename": file.filename, **report}

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
