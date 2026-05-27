"""
capture/capture_service.py
Background packet capture, flow extraction, and real-time hybrid detection.
"""
import json
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from capture.flow_extractor import LiveFlowExtractor
from capture.live_flow_mapper import map_flows_to_cicids
from detection.hybrid import detect_flows

logger = logging.getLogger(__name__)

_capture_service: "CaptureService | None" = None


class CaptureService:
    def __init__(self):
        self.extractor = LiveFlowExtractor(flush_timeout=3.0)
        self.running = False
        self.mode = "idle"  # idle | live | replay
        self.interface: str | None = None
        self.packet_count = 0
        self.flows_detected = 0
        self.threats_detected = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._event_queue: queue.Queue = queue.Queue(maxsize=500)
        self._lock = threading.Lock()
        self._recent_flows: list[dict] = []
        self._max_recent = 100

    def list_interfaces(self) -> list[dict]:
        try:
            from scapy.all import get_if_list, get_if_addr
            ifaces = []
            for name in get_if_list():
                try:
                    addr = get_if_addr(name)
                except Exception:
                    addr = ""
                ifaces.append({"name": name, "address": addr or "N/A"})
            return ifaces
        except Exception as e:
            logger.warning(f"Could not list interfaces: {e}")
            return [{"name": "default", "address": "N/A"}]

    def _emit(self, event_type: str, data: Any):
        event = {"type": event_type, "data": data, "ts": datetime.utcnow().isoformat()}
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                pass
            self._event_queue.put_nowait(event)

    def _process_flows(self, df: pd.DataFrame):
        if df.empty:
            return
        df_cic = map_flows_to_cicids(df)
        try:
            report = detect_flows(df_cic, log_alerts=True, enrich_intel=True)
        except FileNotFoundError as e:
            self._emit("error", {"message": str(e)})
            return
        except Exception as e:
            logger.error(f"Live detection failed: {e}", exc_info=True)
            self._emit("error", {"message": str(e)})
            return

        with self._lock:
            self.flows_detected += report["total_flows"]
            self.threats_detected += report["threats_detected"] + report["anomalies_detected"]

        for row in report["results"]:
            flow_event = {
                "metadata": row["metadata"],
                "status": row["status"],
                "detection_source": row["detection_source"],
                "severity": row["severity"],
                "ml_confidence": row["ml_confidence"],
                "rule_name": row["rule_name"],
            }
            with self._lock:
                self._recent_flows.insert(0, flow_event)
                self._recent_flows = self._recent_flows[: self._max_recent]

            if row["status"] in ("attack", "anomaly"):
                self._emit("alert", flow_event)
            else:
                self._emit("flow", flow_event)

    def _flush_loop(self):
        while not self._stop_event.is_set():
            time.sleep(2)
            df = self.extractor.flush_flows()
            if not df.empty:
                self._process_flows(df)

    def _packet_handler(self, pkt):
        self.packet_count += 1
        self.extractor.process_packet(pkt)
        if self.packet_count % 100 == 0:
            self._emit("stats", self.get_status())

    def start(self, interface: str | None = None):
        if self.running:
            return {"status": "already_running", **self.get_status()}

        self._stop_event.clear()
        self.running = True
        self.mode = "live"
        self.interface = interface or None
        self.packet_count = 0
        self.flows_detected = 0
        self.threats_detected = 0
        self.extractor = LiveFlowExtractor(flush_timeout=3.0)
        self._recent_flows = []

        def run_sniff():
            from scapy.all import sniff
            try:
                sniff(
                    iface=self.interface,
                    prn=self._packet_handler,
                    store=False,
                    stop_filter=lambda _: self._stop_event.is_set(),
                )
            except Exception as e:
                logger.error(f"Sniff error: {e}")
                self._emit("error", {"message": str(e)})
            finally:
                df = self.extractor.flush_flows(force=True)
                if not df.empty:
                    self._process_flows(df)
                self.running = False
                self.mode = "idle"
                self._emit("stopped", self.get_status())

        self._thread = threading.Thread(target=run_sniff, daemon=True, name="capture-sniff")
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True, name="capture-flush")
        self._thread.start()
        self._flush_thread.start()
        self._emit("started", self.get_status())
        return {"status": "started", **self.get_status()}

    def stop(self):
        if not self.running:
            return {"status": "not_running", **self.get_status()}
        self._stop_event.set()
        self.running = False
        return {"status": "stopping", **self.get_status()}

    def replay_pcap(self, pcap_path: Path):
        if self.running:
            return {"status": "error", "message": "Stop capture before replay"}

        self._stop_event.clear()
        self.running = True
        self.mode = "replay"
        self.packet_count = 0
        self.flows_detected = 0
        self.threats_detected = 0

        def run_replay():
            from scapy.all import rdpcap
            try:
                packets = rdpcap(str(pcap_path))
                for pkt in packets:
                    if self._stop_event.is_set():
                        break
                    self._packet_handler(pkt)
                df = self.extractor.flush_flows(force=True)
                if not df.empty:
                    self._process_flows(df)
                self._emit("replay_done", self.get_status())
            except Exception as e:
                logger.error(f"PCAP replay failed: {e}")
                self._emit("error", {"message": str(e)})
            finally:
                self.running = False
                self.mode = "idle"
                self._emit("stopped", self.get_status())

        self._thread = threading.Thread(target=run_replay, daemon=True, name="capture-replay")
        self._thread.start()
        return {"status": "replay_started", "file": pcap_path.name}

    def get_status(self) -> dict:
        with self._lock:
            recent = list(self._recent_flows[:20])
        return {
            "running": self.running,
            "mode": self.mode,
            "interface": self.interface,
            "packet_count": self.packet_count,
            "active_flows": len(self.extractor.active_flows),
            "flows_detected": self.flows_detected,
            "threats_detected": self.threats_detected,
            "recent_flows": recent,
        }

    def get_event(self, timeout: float = 1.0):
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def get_capture_service() -> CaptureService:
    global _capture_service
    if _capture_service is None:
        _capture_service = CaptureService()
    return _capture_service
