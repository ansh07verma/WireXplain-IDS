"""
System routes — health check, backend status, system info
"""
import platform
import sys
from datetime import datetime

from fastapi import APIRouter

router = APIRouter(tags=["System"])

START_TIME = datetime.utcnow()


@router.get("/health")
async def health_check():
    """Backend health check — called by frontend on load"""
    uptime_seconds = (datetime.utcnow() - START_TIME).total_seconds()
    return {
        "status": "ok",
        "version": "2.0.0",
        "uptime_seconds": round(uptime_seconds, 1),
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/info")
async def system_info():
    """Extended system information"""
    import importlib

    packages = ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "shap", "scapy"]
    versions = {}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not installed"

    return {
        "status": "ok",
        "system": platform.system(),
        "python": sys.version.split()[0],
        "packages": versions,
    }
