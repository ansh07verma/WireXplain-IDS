import sqlite3
import logging
import socket
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from config import ALERTS_DB_PATH, SYSLOG_HOST, SYSLOG_PORT, WEBHOOK_URL

logger = logging.getLogger(__name__)

SEVERITY_SCORES = {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}


class AlertManager:
    def __init__(self):
        self.db_path = ALERTS_DB_PATH
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        src_ip TEXT,
                        dst_ip TEXT,
                        dst_port INTEGER,
                        protocol TEXT,
                        status TEXT,
                        confidence REAL,
                        rule_name TEXT,
                        severity TEXT,
                        detection_source TEXT,
                        lifecycle_state TEXT DEFAULT 'open',
                        metadata TEXT
                    )
                ''')
                for col, typedef in [
                    ("detection_source", "TEXT"),
                    ("lifecycle_state", "TEXT DEFAULT 'open'"),
                ]:
                    try:
                        cursor.execute(f"ALTER TABLE alerts ADD COLUMN {col} {typedef}")
                    except sqlite3.OperationalError:
                        pass
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize alerts database: {e}")

    def _compute_severity_score(
        self,
        status: str,
        confidence: float,
        severity: str | None,
        intel: dict | None,
    ) -> str:
        base = SEVERITY_SCORES.get((severity or "medium").lower(), 3)
        if intel and intel.get("abuse_score", 0) >= 75:
            base = max(base, 5)
        if status == "attack" and confidence >= 0.9:
            base = max(base, 4)
        for label, score in SEVERITY_SCORES.items():
            if score == base:
                return label
        return severity or "medium"

    def _export_alert(self, alert_dict: dict):
        if WEBHOOK_URL:
            try:
                import httpx
                httpx.post(WEBHOOK_URL, json=alert_dict, timeout=5.0)
            except Exception as e:
                logger.debug(f"Webhook export failed: {e}")

        if SYSLOG_HOST:
            try:
                msg = json.dumps(alert_dict).encode()
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(msg, (SYSLOG_HOST, SYSLOG_PORT))
                sock.close()
            except Exception as e:
                logger.debug(f"Syslog export failed: {e}")

    def log_alert(
        self,
        flow_meta: dict,
        status: str,
        confidence: float,
        rule_name: str = None,
        severity: str = None,
        detection_source: str = None,
    ):
        try:
            intel = flow_meta.get("intel", {}) if isinstance(flow_meta.get("intel"), dict) else {}
            severity = self._compute_severity_score(status, confidence, severity, intel)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                src_ip = flow_meta.get("Src IP", flow_meta.get("src_ip", ""))
                dst_ip = flow_meta.get("Dst IP", flow_meta.get("dst_ip", ""))
                dst_port = flow_meta.get("Dst Port", flow_meta.get("dst_port", 0))
                protocol = flow_meta.get("Protocol", flow_meta.get("protocol", ""))

                cursor.execute('''
                    INSERT INTO alerts (
                        src_ip, dst_ip, dst_port, protocol, status, confidence,
                        rule_name, severity, detection_source, lifecycle_state, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
                ''', (
                    str(src_ip), str(dst_ip),
                    int(dst_port) if pd.notna(dst_port) else 0,
                    str(protocol), status, float(confidence),
                    rule_name, severity, detection_source,
                    json.dumps(flow_meta),
                ))
                alert_id = cursor.lastrowid
                conn.commit()

            alert_record = {
                "id": alert_id,
                "src_ip": str(src_ip),
                "dst_ip": str(dst_ip),
                "status": status,
                "severity": severity,
                "detection_source": detection_source,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._export_alert(alert_record)
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

    def update_alert(self, alert_id: int, lifecycle_state: str) -> bool:
        valid = {"open", "acknowledged", "false_positive", "closed"}
        if lifecycle_state not in valid:
            raise ValueError(f"Invalid state. Must be one of: {valid}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE alerts SET lifecycle_state = ? WHERE id = ?",
                (lifecycle_state, alert_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_alerts(self, limit: int = 100, severity: str = None, since: str = None, lifecycle_state: str = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM alerts"
                params = []
                conditions = []

                if severity:
                    conditions.append("severity = ?")
                    params.append(severity)
                if since:
                    conditions.append("timestamp >= ?")
                    params.append(since)
                if lifecycle_state:
                    conditions.append("lifecycle_state = ?")
                    params.append(lifecycle_state)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to fetch alerts: {e}")
            return []

    def get_stats(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM alerts")
                total = cursor.fetchone()[0]

                cursor.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status")
                status_counts = dict(cursor.fetchall())

                cursor.execute("SELECT severity, COUNT(*) FROM alerts GROUP BY severity")
                severity_counts = dict(cursor.fetchall())

                cursor.execute("SELECT lifecycle_state, COUNT(*) FROM alerts GROUP BY lifecycle_state")
                lifecycle_counts = dict(cursor.fetchall())

                cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 5")
                recent = [dict(r) for r in cursor.fetchall()]

                return {
                    "total": total,
                    "status_counts": status_counts,
                    "severity_counts": severity_counts,
                    "lifecycle_counts": lifecycle_counts,
                    "recent": recent,
                }
        except Exception as e:
            logger.error(f"Failed to get alert stats: {e}")
            return {
                "total": 0,
                "status_counts": {},
                "severity_counts": {},
                "lifecycle_counts": {},
                "recent": [],
            }
