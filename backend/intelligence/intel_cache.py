"""SQLite cache for threat intelligence lookups."""
import sqlite3
import json
from datetime import datetime, timedelta
from config import INTEL_CACHE_PATH


class IntelCache:
    TTL_HOURS = 24

    def __init__(self):
        INTEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(INTEL_CACHE_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS intel_cache (
                    ip TEXT PRIMARY KEY,
                    data TEXT,
                    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def get(self, ip: str) -> dict | None:
        with sqlite3.connect(INTEL_CACHE_PATH) as conn:
            row = conn.execute(
                "SELECT data, cached_at FROM intel_cache WHERE ip = ?", (ip,)
            ).fetchone()
            if not row:
                return None
            cached_at = datetime.fromisoformat(row[1])
            if datetime.utcnow() - cached_at > timedelta(hours=self.TTL_HOURS):
                return None
            return json.loads(row[0])

    def set(self, ip: str, data: dict):
        with sqlite3.connect(INTEL_CACHE_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO intel_cache (ip, data, cached_at) VALUES (?, ?, ?)",
                (ip, json.dumps(data), datetime.utcnow().isoformat()),
            )
            conn.commit()
