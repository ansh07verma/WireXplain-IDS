"""
WireXplain IDS — FastAPI Backend Entry Point
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import CORS_ORIGINS
from api.pipeline_routes import router as pipeline_router
from api.system_routes import router as system_router
from api.detection_routes import router as detection_router
from api.explain_routes import router as explain_router
from api.alert_routes import router as alert_router
from api.rule_routes import router as rule_router
from api.capture_routes import router as capture_router
from api.settings_routes import router as settings_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("  WireXplain IDS — Backend Starting")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 50)
    yield
    print("WireXplain IDS — Shutting down")


app = FastAPI(
    title="WireXplain IDS",
    description="Explainable Real-Time Hybrid Intrusion Detection System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router,   prefix="/api")
app.include_router(pipeline_router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(detection_router, prefix="/api/detect", tags=["Detection"])
app.include_router(explain_router,   prefix="/api/explain", tags=["Explainability"])
app.include_router(alert_router,     prefix="/api/alerts",  tags=["Alerts"])
app.include_router(rule_router,      prefix="/api/rules",   tags=["Rules"])
app.include_router(capture_router,   prefix="/api/capture", tags=["Capture"])
app.include_router(settings_router,  prefix="/api/settings", tags=["Settings"])


@app.get("/")
async def root():
    return {"name": "WireXplain IDS", "version": "2.0.0", "docs": "/docs"}
