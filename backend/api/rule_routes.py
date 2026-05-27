from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from pydantic import BaseModel
from typing import Any

router = APIRouter(tags=["Rules"])

RULES_PATH = Path(__file__).parent.parent / "config" / "rules.json"


class RulesUpdate(BaseModel):
    rules: list[dict[str, Any]]


@router.get("/")
async def get_rules():
    if not RULES_PATH.exists():
        return []
    return json.loads(RULES_PATH.read_text())


@router.put("/")
async def update_rules(body: RulesUpdate):
    RULES_PATH.write_text(json.dumps(body.rules, indent=2))
    try:
        from detection.signature_detector import SignatureDetector
        from detection import hybrid
        hybrid._signature_detector = SignatureDetector()
    except Exception:
        pass
    return {"status": "updated", "count": len(body.rules)}
