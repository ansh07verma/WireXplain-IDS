from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
import logging
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from alerting.alert_manager import AlertManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Alerts"])

_alert_manager = None


def get_alert_manager():
    global _alert_manager
    if not _alert_manager:
        _alert_manager = AlertManager()
    return _alert_manager


class AlertUpdate(BaseModel):
    lifecycle_state: str


@router.get("")
async def get_alerts(
    limit: int = Query(100, ge=1, le=1000),
    severity: Optional[str] = None,
    since: Optional[str] = None,
    lifecycle_state: Optional[str] = None,
):
    try:
        am = get_alert_manager()
        alerts = am.get_alerts(
            limit=limit, severity=severity, since=since, lifecycle_state=lifecycle_state
        )
        return {"status": "success", "data": alerts}
    except Exception as e:
        logger.error(f"Failed to fetch alerts API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_alert_stats():
    try:
        am = get_alert_manager()
        stats = am.get_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Failed to fetch alert stats API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{alert_id}")
async def update_alert(alert_id: int, body: AlertUpdate):
    try:
        am = get_alert_manager()
        ok = am.update_alert(alert_id, body.lifecycle_state)
        if not ok:
            raise HTTPException(404, f"Alert {alert_id} not found")
        return {"status": "success", "id": alert_id, "lifecycle_state": body.lifecycle_state}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
