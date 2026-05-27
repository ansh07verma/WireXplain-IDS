"""
capture/live_flow_mapper.py
Map live-extracted flow rows to CICIDS2018 column names and fill missing features.
"""
import pandas as pd

LIVE_TO_CIC = {
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Src Port": "Src Port",
    "Flow Packets/s": "Flow Pkts/s",
    "Flow Bytes/s": "Flow Byts/s",
}

# Common CICIDS columns used by rules / feature engineering (defaults for live capture)
CIC_DEFAULTS = {
    "Tot Fwd Pkts": 0,
    "Tot Bwd Pkts": 0,
    "TotLen Fwd Pkts": 0,
    "TotLen Bwd Pkts": 0,
    "Flow Duration": 0,
    "Flow Pkts/s": 0,
    "Flow Byts/s": 0,
    "SYN Flag Cnt": 0,
    "ACK Flag Cnt": 0,
    "FIN Flag Cnt": 0,
    "RST Flag Cnt": 0,
    "PSH Flag Cnt": 0,
    "Fwd Pkts/s": 0,
    "Bwd Pkts/s": 0,
    "Init Fwd Win Byts": 0,
    "Init Bwd Win Byts": 0,
    "Fwd Header Len": 0,
    "Bwd Header Len": 0,
    "Fwd Seg Size Min": 0,
    "Dst Port": 0,
    "fwd_packet_rate": 0,
    "bwd_packet_rate": 0,
    "Flow IAT Mean": 0,
    "Flow IAT Max": 0,
    "Fwd Pkt Len Max": 0,
}


def map_flows_to_cicids(df: pd.DataFrame) -> pd.DataFrame:
    """Rename live columns and ensure CICIDS-compatible schema."""
    if df.empty:
        return df

    out = df.copy()
    out = out.rename(columns={k: v for k, v in LIVE_TO_CIC.items() if k in out.columns})

    for col, default in CIC_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default

    if "Protocol" in out.columns:
        out["Protocol"] = out["Protocol"].apply(
            lambda p: 6 if p == 6 or str(p).upper() == "TCP" else (17 if p == 17 or str(p).upper() == "UDP" else p)
        )

    # Recompute rate features from totals (ensures Flow Pkts/s always present)
    if "Flow Duration" in out.columns:
        duration_sec = (out["Flow Duration"].astype(float).abs() / 1e6).clip(lower=1e-9)
        fwd = out.get("Tot Fwd Pkts", 0)
        bwd = out.get("Tot Bwd Pkts", 0)
        out["Fwd Pkts/s"] = (fwd / duration_sec).clip(0, 1e6)
        out["Bwd Pkts/s"] = (bwd / duration_sec).clip(0, 1e6)
        out["Flow Pkts/s"] = ((fwd + bwd) / duration_sec).clip(0, 1e6)

    return out
