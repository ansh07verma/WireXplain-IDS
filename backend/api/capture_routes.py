"""
api/capture_routes.py
Live packet capture, PCAP replay, and SSE event stream.
"""
import asyncio
import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from capture.capture_service import get_capture_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Capture"])


class StartCaptureRequest(BaseModel):
    interface: str | None = None


@router.get("/interfaces")
async def list_interfaces():
    svc = get_capture_service()
    return {"interfaces": svc.list_interfaces()}


@router.get("/status")
async def capture_status():
    return get_capture_service().get_status()


@router.post("/start")
async def start_capture(body: StartCaptureRequest):
    svc = get_capture_service()
    try:
        return svc.start(interface=body.interface)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/stop")
async def stop_capture():
    return get_capture_service().stop()


@router.post("/replay")
async def replay_pcap(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pcap", ".pcapng", ".cap")):
        raise HTTPException(400, "Only PCAP files are supported.")

    suffix = Path(file.filename).suffix or ".pcap"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    svc = get_capture_service()
    return svc.replay_pcap(tmp_path)


@router.get("/stream")
async def capture_stream():
    svc = get_capture_service()

    async def event_generator():
        yield f"data: {json.dumps({'type': 'connected', 'data': svc.get_status()})}\n\n"
        while True:
            event = await asyncio.get_event_loop().run_in_executor(
                None, lambda: svc.get_event(timeout=1.0)
            )
            if event:
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("stopped", "replay_done", "error"):
                    if event.get("type") == "error" and not svc.running:
                        break
            else:
                if not svc.running and svc._event_queue.empty():
                    yield f": heartbeat\n\n"
                    await asyncio.sleep(0.5)
                    continue
                yield f": heartbeat\n\n"
                await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
