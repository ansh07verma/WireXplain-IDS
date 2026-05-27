"""AbuseIPDB IP reputation lookup."""
import logging
import httpx
from config import ABUSEIPDB_API_KEY

logger = logging.getLogger(__name__)


def check_ip(ip: str) -> dict:
    if not ABUSEIPDB_API_KEY:
        return {"available": False, "reason": "no_api_key"}

    try:
        resp = httpx.get(
            "https://api.abuseipdb.com/api/v2/check",
            headers={"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"},
            params={"ipAddress": ip, "maxAgeInDays": 90},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return {"available": False, "reason": f"http_{resp.status_code}"}

        data = resp.json().get("data", {})
        return {
            "available": True,
            "abuse_score": data.get("abuseConfidenceScore", 0),
            "country": data.get("countryCode", ""),
            "total_reports": data.get("totalReports", 0),
            "is_whitelisted": data.get("isWhitelisted", False),
        }
    except Exception as e:
        logger.debug(f"AbuseIPDB lookup failed for {ip}: {e}")
        return {"available": False, "reason": str(e)}
