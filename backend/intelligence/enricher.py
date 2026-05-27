"""Combine threat intel sources and cache results."""
import ipaddress
import logging
from intelligence.abuseipdb import check_ip as abuse_check
from intelligence.virustotal import check_ip as vt_check
from intelligence.intel_cache import IntelCache

logger = logging.getLogger(__name__)


def _is_private(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return True


class IntelEnricher:
    def __init__(self):
        self.cache = IntelCache()

    def enrich_ip(self, ip: str) -> dict:
        if _is_private(ip):
            return {"ip": ip, "private": True}

        cached = self.cache.get(ip)
        if cached:
            return cached

        abuse = abuse_check(ip)
        vt = vt_check(ip)
        result = {
            "ip": ip,
            "private": False,
            "abuse_score": abuse.get("abuse_score", 0) if abuse.get("available") else 0,
            "abuse": abuse,
            "virustotal": vt,
            "malicious_votes": vt.get("malicious_votes", 0) if vt.get("available") else 0,
        }
        self.cache.set(ip, result)
        return result
