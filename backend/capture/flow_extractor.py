import time
import logging
import pandas as pd
from collections import defaultdict
from scapy.all import IP, TCP, UDP

logger = logging.getLogger(__name__)

class LiveFlowExtractor:
    """
    Lightweight real-time flow extractor.
    Groups packets by 5-tuple and computes basic statistics needed by the pipeline.
    Approximates CICFlowMeter features.
    """
    def __init__(self, flush_timeout=3.0):
        self.flush_timeout = flush_timeout
        self.active_flows = defaultdict(dict)

    def _get_flow_key(self, pkt):
        if not pkt.haslayer(IP):
            return None
            
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        protocol = pkt[IP].proto
        
        src_port = 0
        dst_port = 0
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        else:
            return None # Ignore non TCP/UDP for now

        # Ensure bidirectional packets map to the same flow
        # Forward is defined as the direction of the first seen packet
        key_fwd = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        key_bwd = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        
        if key_bwd in self.active_flows:
            return key_bwd, False # Direction is backward
        return key_fwd, True      # Direction is forward

    def process_packet(self, pkt):
        key_info = self._get_flow_key(pkt)
        if not key_info:
            return
            
        key, is_forward = key_info
        pkt_len = len(pkt)
        ts = time.time()

        if key not in self.active_flows:
            self.active_flows[key] = {
                "Src IP": pkt[IP].src if is_forward else pkt[IP].dst,
                "Dst IP": pkt[IP].dst if is_forward else pkt[IP].src,
                "Src Port": pkt[TCP].sport if pkt.haslayer(TCP) else (pkt[UDP].sport if pkt.haslayer(UDP) else 0),
                "Dst Port": pkt[TCP].dport if pkt.haslayer(TCP) else (pkt[UDP].dport if pkt.haslayer(UDP) else 0),
                "Protocol": pkt[IP].proto,
                "Timestamp": ts,
                "Flow Duration": 0.0,
                "Total Fwd Packets": 0,
                "Total Backward Packets": 0,
                "Total Length of Fwd Packets": 0,
                "Total Length of Bwd Packets": 0,
                "SYN Flag Cnt": 0,
                "ACK Flag Cnt": 0,
                "FIN Flag Cnt": 0,
                "RST Flag Cnt": 0,
                "PSH Flag Cnt": 0,
                "Init Fwd Win Byts": 0,
                "Init Bwd Win Byts": 0,
                "Fwd Header Len": 0,
                "Bwd Header Len": 0,
                "_first_ts": ts,
                "_last_ts": ts
            }

        flow = self.active_flows[key]
        
        if is_forward:
            flow["Total Fwd Packets"] += 1
            flow["Total Length of Fwd Packets"] += pkt_len
        else:
            flow["Total Backward Packets"] += 1
            flow["Total Length of Bwd Packets"] += pkt_len

        if pkt.haslayer(TCP):
            flags = pkt[TCP].flags
            if flags & 0x02:
                flow["SYN Flag Cnt"] += 1
            if flags & 0x10:
                flow["ACK Flag Cnt"] += 1
            if flags & 0x01:
                flow["FIN Flag Cnt"] += 1
            if flags & 0x04:
                flow["RST Flag Cnt"] += 1
            if flags & 0x08:
                flow["PSH Flag Cnt"] += 1
            if is_forward and flow["Init Fwd Win Byts"] == 0:
                flow["Init Fwd Win Byts"] = int(pkt[TCP].window)
            if not is_forward and flow["Init Bwd Win Byts"] == 0:
                flow["Init Bwd Win Byts"] = int(pkt[TCP].window)
            hdr = int(pkt[TCP].dataofs) * 4 if pkt[TCP].dataofs else 20
            if is_forward:
                flow["Fwd Header Len"] += hdr
            else:
                flow["Bwd Header Len"] += hdr
            
        flow["_last_ts"] = max(flow["_last_ts"], ts)
        flow["Flow Duration"] = (flow["_last_ts"] - flow["_first_ts"]) * 1_000_000  # microseconds like CICIDS

    def flush_flows(self, force=False):
        """
        Returns a DataFrame of flows that have expired (or all if force=True).
        Removes flushed flows from active_flows.
        """
        now = time.time()
        expired_keys = []
        flushed_flows = []

        for key, flow in self.active_flows.items():
            if force or (now - flow["_last_ts"]) > self.flush_timeout:
                expired_keys.append(key)
                
                # Copy and clean up internal fields
                flow_copy = flow.copy()
                del flow_copy["_first_ts"]
                del flow_copy["_last_ts"]
                
                # Derive missing basic rates (FeatureEngineer will handle more advanced derivations)
                duration_us = max(flow_copy["Flow Duration"], 1.0)
                duration_sec = duration_us / 1_000_000
                total_pkts = flow_copy["Total Fwd Packets"] + flow_copy["Total Backward Packets"]
                total_bytes = flow_copy["Total Length of Fwd Packets"] + flow_copy["Total Length of Bwd Packets"]
                
                flow_copy["Flow Pkts/s"] = total_pkts / max(duration_sec, 0.0001)
                flow_copy["Flow Byts/s"] = total_bytes / max(duration_sec, 0.0001)
                flow_copy["Fwd Pkts/s"] = flow_copy["Total Fwd Packets"] / max(duration_sec, 0.0001)
                flow_copy["Bwd Pkts/s"] = flow_copy["Total Backward Packets"] / max(duration_sec, 0.0001)
                if total_pkts > 1:
                    flow_copy["Flow IAT Mean"] = duration_us / (total_pkts - 1)
                flow_copy["Flow IAT Max"] = flow_copy.get("Flow IAT Mean", 0)
                flow_copy["Fwd Pkt Len Max"] = max(
                    flow_copy["Total Length of Fwd Packets"] / max(flow_copy["Total Fwd Packets"], 1),
                    0
                )
                
                flushed_flows.append(flow_copy)

        for key in expired_keys:
            del self.active_flows[key]

        if not flushed_flows:
            return pd.DataFrame()
            
        return pd.DataFrame(flushed_flows)
