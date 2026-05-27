"""
pipeline/feature_engineering.py
Transform raw CICIDS2018 DataFrame into ML-ready features.

Steps:
  1. Binary label encoding  (Benign=0, Attack=1)
  2. Derive rate features   (fwd_packet_rate, bwd_packet_rate, etc.)
  3. Drop non-numeric columns
  4. Clip extreme outliers  (99.9th percentile cap)
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column name for binary label output
LABEL_BINARY = "label_binary"

# Labels that are considered benign (everything else = attack)
BENIGN_LABELS = {"benign", "normal"}

# Derived feature definitions: (name, numerator_col, denominator_col, offset)
DERIVED_FEATURES = [
    ("fwd_packet_rate",  "Tot Fwd Pkts",    "Flow Duration",   1e-9),
    ("bwd_packet_rate",  "Tot Bwd Pkts",    "Flow Duration",   1e-9),
    ("fwd_byte_ratio",   "TotLen Fwd Pkts", None,              None),   # special case
    ("syn_ratio",        "SYN Flag Cnt",    "Tot Fwd Pkts",    1.0),
]


class FeatureEngineer:
    """Transforms raw CICIDS2018 data into ML features."""

    def __init__(self):
        self.label_col = "Label"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, emit=None) -> pd.DataFrame:
        """Full engineering pipeline. Returns feature DataFrame + binary label."""
        def log(msg):
            logger.info(msg)
            if emit:
                emit(msg)

        log("Feature engineering started ...")

        df = df.copy()

        # 1. Encode labels
        df = self._encode_labels(df, log)

        # 2. Derive rate features
        df = self._derive_features(df, log)

        # 3. Drop non-numeric & metadata columns
        df = self._drop_non_numeric(df, log)

        # 4. Clip outliers
        df = self._clip_outliers(df, log)

        log(f"Feature engineering complete — {df.shape[1]} columns, {len(df):,} rows")
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_labels(self, df: pd.DataFrame, log) -> pd.DataFrame:
        """Binary encode: Benign=0, Attack=1"""
        raw_labels = df[self.label_col].str.strip().str.lower()
        df[LABEL_BINARY] = (~raw_labels.isin(BENIGN_LABELS)).astype(int)

        n_benign = (df[LABEL_BINARY] == 0).sum()
        n_attack = (df[LABEL_BINARY] == 1).sum()
        log(f"  Labels encoded — Benign: {n_benign:,} | Attack: {n_attack:,}")
        return df

    def _derive_features(self, df: pd.DataFrame, log) -> pd.DataFrame:
        """Create rate-based engineered features."""
        created = []

        if "Flow Duration" in df.columns:
            # CICIDS Flow Duration is in microseconds
            duration_sec = (df["Flow Duration"].abs() / 1e6).clip(lower=1e-9)

            df["fwd_packet_rate"] = (
                df.get("Tot Fwd Pkts", 0) / duration_sec
            ).clip(0, 1e6)
            df["bwd_packet_rate"] = (
                df.get("Tot Bwd Pkts", 0) / duration_sec
            ).clip(0, 1e6)
            created += ["fwd_packet_rate", "bwd_packet_rate"]

            if "Fwd Pkts/s" not in df.columns:
                df["Fwd Pkts/s"] = df["fwd_packet_rate"]
                created.append("Fwd Pkts/s")
            if "Bwd Pkts/s" not in df.columns:
                df["Bwd Pkts/s"] = df["bwd_packet_rate"]
                created.append("Bwd Pkts/s")
            if "Flow Pkts/s" not in df.columns:
                total_pkts = df.get("Tot Fwd Pkts", 0) + df.get("Tot Bwd Pkts", 0)
                df["Flow Pkts/s"] = (total_pkts / duration_sec).clip(0, 1e6)
                created.append("Flow Pkts/s")
            if "Fwd Seg Size Min" not in df.columns:
                df["Fwd Seg Size Min"] = 0
                created.append("Fwd Seg Size Min")

        # fwd byte ratio
        fwd_bytes = df.get("TotLen Fwd Pkts", pd.Series(0, index=df.index))
        bwd_bytes = df.get("TotLen Bwd Pkts", pd.Series(0, index=df.index))
        total_bytes = fwd_bytes + bwd_bytes + 1e-9
        df["fwd_byte_ratio"] = (fwd_bytes / total_bytes).clip(0, 1)
        created.append("fwd_byte_ratio")

        # SYN ratio
        syn = df.get("SYN Flag Cnt", pd.Series(0, index=df.index))
        fwd_pkts = df.get("Tot Fwd Pkts", pd.Series(1, index=df.index))
        df["syn_ratio"] = (syn / (fwd_pkts + 1)).clip(0, 1)
        created.append("syn_ratio")

        log(f"  Derived {len(created)} features: {created}")
        return df

    def _drop_non_numeric(self, df: pd.DataFrame, log) -> pd.DataFrame:
        """Keep only numeric columns (+ label_binary). Drop text metadata."""
        keep = [LABEL_BINARY] + [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != LABEL_BINARY
        ]
        dropped = [c for c in df.columns if c not in keep]
        if dropped:
            log(f"  Dropped {len(dropped)} non-numeric columns")
        return df[keep]

    def _clip_outliers(self, df: pd.DataFrame, log) -> pd.DataFrame:
        """Cap feature values at 99.9th percentile to reduce noise."""
        feat_cols = [c for c in df.columns if c != LABEL_BINARY]
        caps = df[feat_cols].quantile(0.999)
        df[feat_cols] = df[feat_cols].clip(upper=caps, axis=1)
        log(f"  Clipped extreme values at 99.9th percentile")
        return df
