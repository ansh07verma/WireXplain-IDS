"""
WireXplain IDS — Global Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
for d in [DATA_DIR / "raw", DATA_DIR / "processed", MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset
RAW_DATA_PATH = DATA_DIR / "raw" / "02-14-2018.csv"
FEATURES_PATH = DATA_DIR / "processed" / "features.csv"
SELECTED_FEATURES_PATH = DATA_DIR / "processed" / "selected_features.csv"
FILTERED_DATA_PATH = DATA_DIR / "processed" / "filtered_data.csv"

# Model paths
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.pkl"
LGB_MODEL_PATH = MODELS_DIR / "lightgbm.pkl"
ISOLATION_MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# Training defaults
TOP_N_FEATURES = 15
CONTAMINATION = 0.05
N_ESTIMATORS = 100
TEST_SIZE = 0.2

# API Keys (from .env)
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY", "")
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")

# ML model selection: rf | xgb | lgb
MODEL_TYPE = os.getenv("MODEL_TYPE", "rf").lower()

# SIEM export
SYSLOG_HOST = os.getenv("SYSLOG_HOST", "")
SYSLOG_PORT = int(os.getenv("SYSLOG_PORT", "514"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# Intel cache DB
INTEL_CACHE_PATH = DATA_DIR / "intel_cache.db"

# Dataset registry paths
DATASETS = {
    "cicids2018": DATA_DIR / "raw" / "02-14-2018.csv",
    "unsw_nb15": DATA_DIR / "raw" / "UNSW-NB15_1.csv",
}
DEFAULT_DATASET = os.getenv("DEFAULT_DATASET", "cicids2018")

# Alerts DB
ALERTS_DB_PATH = DATA_DIR / "alerts.db"

# CORS origins (frontend dev server)
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
