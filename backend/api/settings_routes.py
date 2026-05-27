"""
api/settings_routes.py
Runtime configuration for API keys and model selection.
"""
import os
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_TYPE, ABUSEIPDB_API_KEY, VIRUSTOTAL_API_KEY, DEFAULT_DATASET

router = APIRouter(tags=["Settings"])

ENV_PATH = Path(__file__).parent.parent / ".env"


class SettingsUpdate(BaseModel):
    abuseipdb_api_key: str | None = None
    virustotal_api_key: str | None = None
    model_type: str | None = None
    default_dataset: str | None = None
    syslog_host: str | None = None
    webhook_url: str | None = None


def _read_env() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _write_env(updates: dict):
    env = _read_env()
    env.update(updates)
    lines = [f"{k}={v}" for k, v in env.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n")
    for k, v in updates.items():
        os.environ[k] = v


@router.get("")
async def get_settings():
    return {
        "model_type": MODEL_TYPE,
        "default_dataset": DEFAULT_DATASET,
        "abuseipdb_configured": bool(ABUSEIPDB_API_KEY),
        "virustotal_configured": bool(VIRUSTOTAL_API_KEY),
    }


@router.put("")
async def update_settings(body: SettingsUpdate):
    updates = {}
    if body.abuseipdb_api_key is not None:
        updates["ABUSEIPDB_API_KEY"] = body.abuseipdb_api_key
    if body.virustotal_api_key is not None:
        updates["VIRUSTOTAL_API_KEY"] = body.virustotal_api_key
    if body.model_type is not None:
        updates["MODEL_TYPE"] = body.model_type
    if body.default_dataset is not None:
        updates["DEFAULT_DATASET"] = body.default_dataset
    if body.syslog_host is not None:
        updates["SYSLOG_HOST"] = body.syslog_host
    if body.webhook_url is not None:
        updates["WEBHOOK_URL"] = body.webhook_url

    if updates:
        _write_env(updates)
        import importlib
        import config as cfg
        importlib.reload(cfg)
        if "MODEL_TYPE" in updates:
            from detection import hybrid
            hybrid._ml_detector = None

    return {"status": "updated", "keys": list(updates.keys())}
