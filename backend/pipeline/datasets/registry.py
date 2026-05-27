"""
pipeline/datasets/registry.py
Multi-dataset support with schema normalization.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from config import DATASETS, DATA_DIR

logger = logging.getLogger(__name__)

# Column mappings: dataset column -> CICIDS-style column
UNSW_MAP = {
    "sport": "Src Port",
    "dsport": "Dst Port",
    "spkts": "Tot Fwd Pkts",
    "dpkts": "Tot Bwd Pkts",
    "sbytes": "TotLen Fwd Pkts",
    "dbytes": "TotLen Bwd Pkts",
    "dur": "Flow Duration",
    "proto": "Protocol",
    "label": "Label",
}

CICIDS2018_META = {
    "id": "cicids2018",
    "name": "CICIDS2018",
    "label_col": "Label",
    "benign_values": {"benign"},
}


def list_datasets() -> list[dict]:
    result = []
    for ds_id, path in DATASETS.items():
        result.append({
            "id": ds_id,
            "name": ds_id.replace("_", " ").upper(),
            "path": str(path),
            "ready": Path(path).exists(),
        })
    return result


def get_dataset_path(dataset_id: str) -> Path:
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {list(DATASETS.keys())}")
    return Path(DATASETS[dataset_id])


def _normalize_unsw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    for src, dst in UNSW_MAP.items():
        if src in df.columns:
            df[dst] = df[src]

    if "srcip" in df.columns:
        df["Src IP"] = df["srcip"]
    if "dstip" in df.columns:
        df["Dst IP"] = df["dstip"]

    if "Label" in df.columns:
        df["Label"] = df["Label"].apply(
            lambda x: "BENIGN" if str(x).lower() in ("normal", "benign") else "Attack"
        )

    if "Flow Duration" in df.columns:
        df["Flow Duration"] = df["Flow Duration"] * 1_000_000

    return df


def load_dataset(dataset_id: str, emit=None) -> pd.DataFrame:
    def log(msg):
        logger.info(msg)
        if emit:
            emit(msg)

    path = get_dataset_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' not found at {path}. "
            f"Place the CSV file in backend/data/raw/"
        )

    log(f"Loading dataset '{dataset_id}' from {path.name} ...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    log(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if dataset_id == "unsw_nb15":
        df = _normalize_unsw(df)
        log("  Applied UNSW-NB15 → CICIDS column mapping")
    elif dataset_id == "cicids2018":
        if "Label" not in df.columns:
            raise ValueError("CICIDS2018 requires a 'Label' column")

    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    log(f"Dataset '{dataset_id}' loaded — {len(df):,} samples")
    return df
