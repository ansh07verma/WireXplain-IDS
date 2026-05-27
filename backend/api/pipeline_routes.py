"""
api/pipeline_routes.py
REST + SSE endpoints for the training pipeline.

Endpoints:
  GET  /api/pipeline/status        — which models exist + dataset present
  POST /api/pipeline/run           — kick off training (returns run_id)
  GET  /api/pipeline/stream/{id}   — SSE stream of live log lines
  GET  /api/pipeline/metrics       — last training metrics
  GET  /api/pipeline/features      — feature importance ranking
"""
import json
import asyncio
import threading
import queue
import time
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add parent to path for sibling imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    RF_MODEL_PATH, ISOLATION_MODEL_PATH, XGB_MODEL_PATH, LGB_MODEL_PATH,
    FEATURE_NAMES_PATH, MODELS_DIR, DATA_DIR,
    TOP_N_FEATURES, CONTAMINATION, N_ESTIMATORS, TEST_SIZE, DEFAULT_DATASET,
)
from pipeline.datasets.registry import list_datasets, load_dataset, get_dataset_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Pipeline"])

# ── In-memory state ──────────────────────────────────────────────────────────
# Each training run gets a unique ID and its own log queue
_runs: dict[str, dict] = {}   # run_id → {queue, status, metrics, started_at}
_latest_run_id: str | None = None


# ── Pydantic models ──────────────────────────────────────────────────────────
class RunConfig(BaseModel):
    n_estimators: int = N_ESTIMATORS
    test_size: float = TEST_SIZE
    contamination: float = CONTAMINATION
    top_n_features: int = TOP_N_FEATURES
    dataset: str = DEFAULT_DATASET


# ── Helpers ──────────────────────────────────────────────────────────────────
def _make_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _save_metrics(metrics: dict):
    """Persist last metrics to disk so they survive restarts."""
    path = MODELS_DIR / "last_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=str))


def _load_metrics() -> dict | None:
    path = MODELS_DIR / "last_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ── Background training thread ───────────────────────────────────────────────
def _run_pipeline(run_id: str, config: RunConfig):
    """Runs in a background thread. Streams log lines into the run's queue."""
    run = _runs[run_id]
    q: queue.Queue = run["queue"]

    def emit(msg: str):
        """Put a log line into the SSE queue."""
        q.put({"type": "log", "data": msg})

    try:
        run["status"] = "running"
        emit("=" * 56)
        emit(f"  WireXplain IDS — Pipeline Run  [{run_id}]")
        emit("=" * 56)

        # ── Stage 1: Load data ────────────────────────────────────
        emit("")
        emit("▶ Stage 1/4 — Loading Dataset")
        emit("-" * 40)

        emit(f"  Dataset: {config.dataset}")
        df_raw = load_dataset(config.dataset, emit=emit)

        # ── Stage 2: Feature Engineering ─────────────────────────
        emit("")
        emit("▶ Stage 2/4 — Feature Engineering")
        emit("-" * 40)

        from pipeline.feature_engineering import FeatureEngineer
        df_eng = FeatureEngineer().run(df_raw, emit=emit)

        # ── Stage 3: Feature Selection ────────────────────────────
        emit("")
        emit("▶ Stage 3/4 — Feature Selection (Mutual Information)")
        emit("-" * 40)

        from pipeline.feature_selection import FeatureSelector

        selector_path = MODELS_DIR / "feature_selector.json"
        selector = FeatureSelector(top_n=config.top_n_features)
        selector.fit(df_eng, emit=emit)
        df_sel = selector.transform(df_eng)
        selector.save(selector_path)

        # Also save processed CSV for detection module to reuse
        processed_path = DATA_DIR / "processed" / "selected_features.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df_sel.to_csv(processed_path, index=False)
        emit(f"  Saved processed dataset → {processed_path.name}")

        # ── Stage 4: Train Models ─────────────────────────────────
        emit("")
        emit("▶ Stage 4/4 — Training Models")
        emit("-" * 40)

        from pipeline.train import ModelTrainer

        trainer = ModelTrainer(
            n_estimators=config.n_estimators,
            test_size=config.test_size,
            contamination=config.contamination,
        )
        metrics = trainer.run(
            df=df_sel,
            rf_model_path=RF_MODEL_PATH,
            iso_model_path=ISOLATION_MODEL_PATH,
            feature_names_path=FEATURE_NAMES_PATH,
            xgb_model_path=XGB_MODEL_PATH,
            lgb_model_path=LGB_MODEL_PATH,
            emit=emit,
        )

        # Persist metrics
        _save_metrics(metrics)
        run["metrics"] = metrics
        run["status"] = "done"

        emit("")
        emit("=" * 56)
        emit("  ✅ PIPELINE COMPLETE")
        emit(f"  Accuracy: {metrics['accuracy']*100:.2f}%  |  F1: {metrics['f1_score']*100:.2f}%")
        emit("=" * 56)

        q.put({"type": "done", "data": metrics})

    except FileNotFoundError as e:
        run["status"] = "error"
        emit(f"")
        emit(f"  ❌ ERROR: {e}")
        emit(f"  → Download 02-14-2018.csv and place it in backend/data/raw/")
        q.put({"type": "error", "data": str(e)})

    except Exception as e:
        import traceback
        run["status"] = "error"
        emit(f"")
        emit(f"  ❌ PIPELINE FAILED: {e}")
        emit(traceback.format_exc())
        q.put({"type": "error", "data": str(e)})


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/datasets")
async def pipeline_datasets():
    return {"datasets": list_datasets()}


@router.get("/status")
async def pipeline_status():
    """Check which models are trained and whether the dataset is present."""
    default_path = get_dataset_path(DEFAULT_DATASET)
    return {
        "dataset_ready": default_path.exists(),
        "dataset_path": str(default_path),
        "default_dataset": DEFAULT_DATASET,
        "datasets": list_datasets(),
        "models": {
            "random_forest":    RF_MODEL_PATH.exists(),
            "isolation_forest": ISOLATION_MODEL_PATH.exists(),
            "xgboost":          (MODELS_DIR / "xgboost.pkl").exists(),
            "lightgbm":         (MODELS_DIR / "lightgbm.pkl").exists(),
        },
        "any_model_trained": RF_MODEL_PATH.exists(),
        "feature_selector_ready": (MODELS_DIR / "feature_selector.json").exists(),
        "active_run": _latest_run_id,
        "run_status": _runs[_latest_run_id]["status"] if _latest_run_id and _latest_run_id in _runs else None,
    }


@router.post("/run")
async def run_pipeline(config: RunConfig):
    """Start a new training run in the background."""
    global _latest_run_id

    # Refuse to run if another is already in progress
    if _latest_run_id and _runs.get(_latest_run_id, {}).get("status") == "running":
        raise HTTPException(400, "A pipeline run is already in progress.")

    run_id = _make_run_id()
    _runs[run_id] = {
        "queue": queue.Queue(),
        "status": "starting",
        "metrics": None,
        "started_at": datetime.utcnow().isoformat(),
    }
    _latest_run_id = run_id

    # Launch in background thread (training is CPU-bound)
    thread = threading.Thread(
        target=_run_pipeline,
        args=(run_id, config),
        daemon=True,
        name=f"pipeline-{run_id}",
    )
    thread.start()

    return {"run_id": run_id, "status": "started"}


@router.get("/stream/{run_id}")
async def stream_logs(run_id: str):
    """
    SSE endpoint — streams log lines as the pipeline runs.
    Each event is a JSON object: {type: "log"|"done"|"error", data: ...}
    """
    if run_id not in _runs:
        raise HTTPException(404, f"Run ID '{run_id}' not found")

    run = _runs[run_id]
    q: queue.Queue = run["queue"]

    async def event_generator():
        # Initial connection acknowledgement
        yield f"data: {json.dumps({'type': 'connected', 'data': run_id})}\n\n"

        while True:
            try:
                # Non-blocking get with short timeout to stay async-friendly
                item = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=1.0)
                )
                yield f"data: {json.dumps(item)}\n\n"

                if item["type"] in ("done", "error"):
                    break

            except queue.Empty:
                # Heartbeat so the connection stays alive
                if run["status"] not in ("running", "starting"):
                    break
                yield f": heartbeat\n\n"
                await asyncio.sleep(0.1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/metrics")
async def get_metrics():
    """Return metrics from the last completed training run."""
    metrics = _load_metrics()
    if not metrics:
        raise HTTPException(404, "No training run has completed yet.")
    return metrics


@router.get("/features")
async def get_features():
    """Return feature importance from the trained RandomForest."""
    if not RF_MODEL_PATH.exists() or not FEATURE_NAMES_PATH.exists():
        raise HTTPException(404, "Model not trained yet.")

    import joblib
    from pipeline.evaluate import feature_importance, load_feature_names

    model = joblib.load(RF_MODEL_PATH)
    feature_names = load_feature_names(FEATURE_NAMES_PATH)

    return {
        "feature_names": feature_names,
        "importance": feature_importance(model, feature_names),
    }
