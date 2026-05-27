<div align="center">

# 🛡️ WireXplain IDS

### Explainable Real-Time Hybrid Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A production-grade intrusion detection system that captures live network traffic, detects attacks using multiple AI models, and explains *why* each decision was made.

![WireXplain Dashboard](https://img.shields.io/badge/Status-Active-brightgreen)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Dataset Setup](#-dataset-setup)
- [How to Use](#-how-to-use)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Detection Methodology](#-detection-methodology)
- [Docker Deployment](#-docker-deployment)
- [Results](#-results)

---

## 🔍 Overview

WireXplain IDS is a **hybrid intrusion detection system** that goes beyond traditional IDS tools by combining three detection mechanisms and providing human-readable AI explanations for every alert.

**The problem with most IDS tools:**
- High false positives with no explanation
- ML-only or signature-only — not both
- Can't process live traffic, only offline CSVs
- Black-box decisions that analysts can't act on

**WireXplain solves this by:**
- Capturing and processing **live network packets** in real time
- Running **three detection engines simultaneously** (ML + anomaly + signatures)
- Using **SHAP** to explain every model decision in plain English
- Providing a **SIEM-style dashboard** with alert lifecycle management

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Live Packet Capture** | Real-time sniffing on any network interface using Scapy |
| **PCAP Replay** | Test with known attack captures without live traffic |
| **Hybrid Detection** | ML classifier + IsolationForest anomaly + JSON signature rules |
| **Multi-Model ML** | Train and compare RandomForest, XGBoost, and LightGBM |
| **SHAP Explainability** | Per-flow feature contributions with natural language reasoning |
| **Threat Intelligence** | AbuseIPDB + VirusTotal IP reputation lookups with caching |
| **SIEM-style Alerts** | Severity scoring, alert lifecycle (ack/close/false positive), syslog export |
| **Multi-Dataset** | Train on CICIDS2018 or UNSW-NB15 from the UI |
| **Real-Time Dashboard** | SSE-powered live event stream, packet rate chart, alert stats |
| **Docker Deployment** | Full stack containerised with docker-compose |

---

## 🏗️ Architecture

```
Network Interface / PCAP File
         │
         ▼
  ┌─────────────────┐
  │  CaptureService  │  ← Scapy packet sniffer (background thread)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │LiveFlowExtractor │  ← Bidirectional 5-tuple flow builder
  └────────┬────────┘
           │  (flush every 3s)
           ▼
  ┌─────────────────┐
  │ Feature Engineer │  ← Derives rates, flags, IAT stats (CICIDS-compatible)
  └────────┬────────┘
           │
           ▼
  ┌────────────────────────────────────┐
  │          HybridDetector            │
  │  ┌──────────┐ ┌─────┐ ┌────────┐  │
  │  │Signatures│ │ ML  │ │AnomalyD│  │  ← Priority: sig > ML > anomaly
  │  └──────────┘ └─────┘ └────────┘  │
  └────────┬───────────────────────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
AlertManager  ThreatIntel    ← SQLite + AbuseIPDB/VirusTotal cache
     │
     ▼
 SSE Stream  →  React Dashboard
     │
     ▼
SHAP Explainer (on demand per alert)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **ML Models** | scikit-learn (RandomForest, IsolationForest), XGBoost, LightGBM |
| **Explainability** | SHAP (TreeExplainer) |
| **Packet Capture** | Scapy, Npcap (Windows) |
| **Frontend** | React 18, Vite 5, Recharts |
| **Database** | SQLite (alerts + intel cache) |
| **Intelligence** | AbuseIPDB API, VirusTotal API |
| **Deployment** | Docker, docker-compose, Nginx |
| **Dataset** | CICIDS2018, UNSW-NB15 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Npcap](https://npcap.com/) (Windows — for live capture, install with "WinPcap API-compatible mode")
- Run backend **as Administrator** for live packet sniffing

### 1. Clone

```bash
git clone https://github.com/ansh07verma/wirexplain-ide.git
cd wirexplain-ide
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # Add API keys (optional)
uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## 📦 Dataset Setup

WireXplain supports two datasets out of the box. Place CSV files in `backend/data/raw/`:

| Dataset | File name | Download |
|---------|-----------|----------|
| **CICIDS2018** (recommended) | `02-14-2018.csv` | [Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv) |
| **UNSW-NB15** | `UNSW-NB15_1.csv` | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

After placing the CSV, go to the **Pipeline** page and click **Run Pipeline** to train all models.

---

## 📖 How to Use

### 1. Train Models
- Go to **Pipeline** page → configure settings → click **Run Pipeline**
- Watch live training logs stream in real time
- All four models (RF, XGB, LGB, IsolationForest) train in one run

### 2. Live Capture
- Go to **Live Capture** page
- Select your active network interface (the one with your LAN/Wi-Fi IP)
- Click **Start Capture** and browse normally — flows appear within seconds
- Or click **Replay PCAP** to test with a known attack file

### 3. CSV Detection
- Go to **Predict** page
- Upload a CICIDS-format CSV file
- Hybrid detection runs and returns per-flow results with status, confidence, and source

### 4. Explain an Alert
- Go to **Explain** page
- Upload the same CSV — choose global (overall importance) or local (per-row SHAP)
- See which features drove the decision and a natural-language explanation

### 5. Alert Management
- Go to **Alerts** page — auto-refreshes every 15 seconds
- Filter by severity (critical / high / medium / info)
- Click **ack**, **fp** (false positive), or **close** on each alert

### 6. Settings
- Go to **Settings** page
- Add AbuseIPDB and VirusTotal API keys for threat intelligence
- Switch active ML model (RF / XGB / LGB) without retraining
- Configure syslog or webhook for SIEM export

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Backend health check |
| `GET` | `/api/pipeline/status` | Model and dataset status |
| `POST` | `/api/pipeline/run` | Start training run |
| `GET` | `/api/pipeline/stream/{id}` | SSE live training logs |
| `GET` | `/api/pipeline/metrics` | Last training metrics |
| `POST` | `/api/detect/csv` | Batch hybrid detection on CSV |
| `POST` | `/api/capture/start` | Start live packet capture |
| `POST` | `/api/capture/stop` | Stop capture |
| `POST` | `/api/capture/replay` | Replay a PCAP file |
| `GET` | `/api/capture/stream` | SSE live flow events |
| `GET` | `/api/capture/interfaces` | List network interfaces |
| `POST` | `/api/explain/global` | SHAP global feature importance |
| `POST` | `/api/explain/local` | SHAP local per-row explanation |
| `GET` | `/api/alerts` | List alerts (filterable) |
| `PATCH` | `/api/alerts/{id}` | Update alert lifecycle state |
| `GET` | `/api/alerts/stats` | Alert summary statistics |
| `GET` | `/api/rules/` | Get signature rules |
| `PUT` | `/api/rules/` | Update signature rules |
| `GET` | `/api/settings` | Get current settings |
| `PUT` | `/api/settings` | Update API keys / model config |

Full interactive docs at **http://localhost:8000/docs**

---

## 📁 Project Structure

```
wirexplain-ide/
│
├── backend/                    # FastAPI Python backend
│   ├── main.py                 # App entry point, router registration
│   ├── config.py               # Global config, env vars, paths
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   ├── api/                    # REST + SSE route handlers
│   │   ├── pipeline_routes.py  # Training pipeline + SSE log stream
│   │   ├── capture_routes.py   # Live capture + PCAP replay
│   │   ├── detection_routes.py # CSV batch detection
│   │   ├── explain_routes.py   # SHAP explanations
│   │   ├── alert_routes.py     # Alert CRUD + lifecycle
│   │   ├── rule_routes.py      # Signature rule management
│   │   └── settings_routes.py  # Runtime configuration
│   │
│   ├── capture/                # Packet capture & flow extraction
│   │   ├── capture_service.py  # Background sniff/replay service
│   │   ├── flow_extractor.py   # Bidirectional flow builder
│   │   └── live_flow_mapper.py # Maps live features to CICIDS schema
│   │
│   ├── detection/              # Detection engines
│   │   ├── hybrid.py           # Fusion logic (sig > ML > anomaly)
│   │   ├── ml_detector.py      # RF / XGB / LGB wrapper
│   │   ├── anomaly_detector.py # IsolationForest wrapper
│   │   └── signature_detector.py # JSON rule engine + heuristics
│   │
│   ├── pipeline/               # ML training pipeline
│   │   ├── train.py            # Train RF, XGB, LGB, IsolationForest
│   │   ├── feature_engineering.py
│   │   ├── feature_selection.py # Mutual Information feature ranking
│   │   ├── evaluate.py
│   │   └── datasets/
│   │       └── registry.py     # Multi-dataset loader (CICIDS, UNSW)
│   │
│   ├── explainability/
│   │   └── shap_explainer.py   # SHAP TreeExplainer, natural language
│   │
│   ├── intelligence/           # Threat intelligence
│   │   ├── enricher.py
│   │   ├── abuseipdb.py
│   │   ├── virustotal.py
│   │   └── intel_cache.py      # SQLite TTL cache
│   │
│   ├── alerting/
│   │   └── alert_manager.py    # SQLite alerts + syslog/webhook export
│   │
│   └── config/
│       └── rules.json          # Signature detection rules
│
├── frontend/                   # React 18 + Vite SPA
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx   # System overview + live alert stats
│   │   │   ├── Pipeline.jsx    # Training UI with live SSE logs
│   │   │   ├── LiveCapture.jsx # Real-time capture + event stream
│   │   │   ├── Predict.jsx     # CSV upload batch detection
│   │   │   ├── Explain.jsx     # SHAP visualization
│   │   │   ├── Alerts.jsx      # SIEM-style alert management
│   │   │   └── Settings.jsx    # API keys + model config
│   │   ├── components/         # Navbar, MetricCard, StatusBadge, etc.
│   │   ├── hooks/useApi.js     # Health + pipeline status polling
│   │   └── api.js              # All backend API calls
│   ├── Dockerfile
│   └── nginx.conf
│
└── docker-compose.yml
```

---

## 🔬 Detection Methodology

### Fusion Priority
```
Signature match  →  status: "attack"   (highest confidence)
ML prediction    →  status: "attack"   (classifier fired)
Anomaly score    →  status: "anomaly"  (unusual flow pattern)
All pass         →  status: "normal"
```

### Signature Rules (`backend/config/rules.json`)
- Rule-based JSON conditions (port matching, rate thresholds, flag counts)
- Built-in heuristics: port scan detection (>50 unique dst ports per src IP), DoS detection (>500 flows per src IP)

### ML Models
- **RandomForest** — ensemble of decision trees, good baseline
- **XGBoost** — gradient boosting, often highest accuracy
- **LightGBM** — fast gradient boosting, best for large datasets
- All trained on top-15 features selected by Mutual Information from the full CICIDS2018 feature set

### Anomaly Detection
- **IsolationForest** — unsupervised; flags flows that are statistically unusual compared to the training distribution
- Runs independently of the ML classifier — catches zero-day-style patterns

### Explainability (SHAP)
- TreeExplainer on the trained classifier
- **Global**: mean absolute SHAP values across the dataset — shows which features matter most overall
- **Local**: per-flow SHAP values — shows exactly why this specific flow was flagged

---

## 🐳 Docker Deployment

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000/docs

> **Note for live capture in Docker:** Linux hosts can use `network_mode: host` in `docker-compose.yml`. On Windows, run the backend natively and only containerise the frontend, or use PCAP replay mode.

---

## 📊 Results

Trained on CICIDS2018 (14 February 2018 capture — 822,947 flows):

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| RandomForest | 100.00% | 100.00% | 1.000 |
| XGBoost | 100.00% | 100.00% | 1.000 |
| LightGBM | 100.00% | 100.00% | 1.000 |

> **Note on accuracy:** CICIDS2018 has well-separated attack patterns and known data quality characteristics (near-duplicate rows, clear label separation) that result in very high scores on test splits derived from the same capture. Real-world performance on unseen live traffic will be lower, which is why the IsolationForest anomaly layer and threshold tuning are important for production use.

**Features selected by Mutual Information (top 15):**

`Init Fwd Win Byts` · `Fwd Seg Size Min` · `Dst Port` · `Init Bwd Win Byts` · `Flow Duration` · `Flow IAT Mean` · `bwd_packet_rate` · `Bwd Pkts/s` · `Bwd Header Len` · `Fwd Pkts/s` · `Fwd Header Len` · `Flow Pkts/s` · `fwd_packet_rate` · `Flow IAT Max` · `Fwd Pkt Len Max`

---

## 🔧 Configuration

All runtime settings are in `backend/.env` (copy from `.env.example`):

```env
# Threat intelligence (optional)
ABUSEIPDB_API_KEY=your_key_here
VIRUSTOTAL_API_KEY=your_key_here

# ML model selection: rf | xgb | lgb
MODEL_TYPE=rf

# Default training dataset: cicids2018 | unsw_nb15
DEFAULT_DATASET=cicids2018

# SIEM export (optional)
SYSLOG_HOST=
WEBHOOK_URL=
```

---

## 👤 Author

**Ansh Verma** — Final Year ECE Student  
GitHub: [@ansh07verma](https://github.com/ansh07verma)

---

## 📄 License

This project is licensed under the MIT License.
