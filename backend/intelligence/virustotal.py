"""VirusTotal IP reputation lookup."""
import logging
import httpx
from config import VIRUSTOTAL_API_KEY

logger = logging.getLogger(__name__)


def check_ip(ip: str) -> dict:
    if not VIRUSTOTAL_API_KEY:
        return {"available": False, "reason": "no_api_key"}

    try:
        resp = httpx.get(
            f"https://www.virustotal.com/api/v3/ip_addresses/{ip}",
            headers={"x-apikey": VIRUSTOTAL_API_KEY},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return {"available": False, "reason": f"http_{resp.status_code}"}

        stats = resp.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        malicious = stats.get("malicious", 0)
        return {
            "available": True,
            "malicious_votes": malicious,
            "suspicious_votes": stats.get("suspicious", 0),
            "harmless_votes": stats.get("harmless", 0),
        }
    except Exception as e:
        logger.debug(f"VirusTotal lookup failed for {ip}: {e}")
        return {"available": False, "reason": str(e)}
