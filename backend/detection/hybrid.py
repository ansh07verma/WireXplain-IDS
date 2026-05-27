"""
detection/hybrid.py
Shared hybrid detection logic: ML + anomaly + signatures with fusion policy.
"""
import logging
import pandas as pd
from pathlib import Path

from pipeline.feature_engineering import FeatureEngineer
from pipeline.feature_selection import FeatureSelector
from detection.ml_detector import MLDetector
from detection.anomaly_detector import AnomalyDetector
from detection.signature_detector import SignatureDetector
from alerting.alert_manager import AlertManager
from config import MODELS_DIR, MODEL_TYPE

logger = logging.getLogger(__name__)

_ml_detector = None
_iso_detector = None
_feature_selector = None
_feature_engineer = None
_signature_detector = None
_alert_manager = None

SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def get_detectors():
    global _ml_detector, _iso_detector, _feature_selector, _feature_engineer
    global _signature_detector, _alert_manager

    if not _ml_detector or _ml_detector.model_type != MODEL_TYPE:
        _ml_detector = MLDetector(model_type=MODEL_TYPE)
    if not _iso_detector:
        _iso_detector = AnomalyDetector()
    if not _signature_detector:
        _signature_detector = SignatureDetector()
    if not _alert_manager:
        _alert_manager = AlertManager()
    if not _feature_selector:
        selector_path = MODELS_DIR / "feature_selector.json"
        if not selector_path.exists():
            raise FileNotFoundError("Feature selector not trained. Run the pipeline first.")
        _feature_selector = FeatureSelector.load(selector_path)
    if not _feature_engineer:
        _feature_engineer = FeatureEngineer()

    return (
        _ml_detector,
        _iso_detector,
        _signature_detector,
        _feature_selector,
        _feature_engineer,
        _alert_manager,
    )


def _meta_cols(df: pd.DataFrame) -> list[str]:
    preferred = ["Src IP", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Src Port"]
    cols = [c for c in preferred if c in df.columns]
    return cols if cols else list(df.columns[:3])


def _fuse_row(
    is_attack: bool,
    is_anomaly: bool,
    sig_match: dict,
    ml_confidence: float,
) -> dict:
    """Priority: signature > ML attack > anomaly > normal."""
    sig_hit = bool(sig_match.get("signature_match"))

    if sig_hit:
        status = "attack"
        source = "signature"
        severity = sig_match.get("severity") or "high"
    elif is_attack:
        status = "attack"
        source = "ml"
        severity = "high" if ml_confidence >= 0.8 else "medium"
    elif is_anomaly:
        status = "anomaly"
        source = "anomaly"
        severity = "medium"
    else:
        status = "normal"
        source = "none"
        severity = "info"

    return {
        "status": status,
        "detection_source": source,
        "severity": severity,
    }


def detect_flows(
    df: pd.DataFrame,
    log_alerts: bool = True,
    enrich_intel: bool = True,
) -> dict:
    """
    Run hybrid detection on a flow DataFrame (CICIDS-format or mapped live flows).
    Returns summary dict with per-row results.
    """
    if df.empty:
        return {
            "total_flows": 0,
            "threats_detected": 0,
            "anomalies_detected": 0,
            "signature_hits": 0,
            "results": [],
        }

    ml, iso, sig, selector, engineer, alert_mgr = get_detectors()

    meta_cols = _meta_cols(df)
    original_meta = df[meta_cols].copy()

    df_work = df.copy()
    if "Label" not in df_work.columns:
        df_work["Label"] = "BENIGN"

    # Signatures need full pre-engineering columns (ports, SYN counts, etc.)
    sig_df = df_work.copy()

    df_eng = engineer.run(df_work.copy())

    # Ensure all trained features exist (live capture may omit some CICIDS columns)
    for col in selector.selected_features_:
        if col not in df_eng.columns:
            df_eng[col] = 0.0

    df_sel = selector.transform(df_eng)
    if "label_binary" in df_sel.columns:
        df_sel = df_sel.drop(columns=["label_binary"])

    ml_results = ml.predict(df_sel)
    iso_results = iso.predict(df_sel)
    sig_results = sig.predict(sig_df)

    intel_enricher = None
    if enrich_intel:
        try:
            from intelligence.enricher import IntelEnricher
            intel_enricher = IntelEnricher()
        except Exception as e:
            logger.debug(f"Intel enricher unavailable: {e}")

    results = []
    for i in range(len(df)):
        row_meta = original_meta.iloc[i].to_dict()
        if "Timestamp" in row_meta and pd.notna(row_meta.get("Timestamp")):
            row_meta["Timestamp"] = str(row_meta["Timestamp"])

        is_attack = bool(ml_results["labels"][i] == 1)
        is_anomaly = bool(iso_results["is_anomaly"][i])
        sig_match = sig_results[i]
        confidence = float(ml_results["probabilities"][i])

        fused = _fuse_row(is_attack, is_anomaly, sig_match, confidence)

        intel_data = {}
        if intel_enricher and fused["status"] in ("attack", "anomaly"):
            src = row_meta.get("Src IP", "")
            intel_data = intel_enricher.enrich_ip(str(src)) if src else {}
            if intel_data.get("abuse_score", 0) >= 75:
                fused["severity"] = "critical"

        result = {
            "id": i,
            "metadata": row_meta,
            "status": fused["status"],
            "detection_source": fused["detection_source"],
            "severity": fused["severity"],
            "ml_confidence": round(confidence, 4),
            "anomaly_score": round(iso_results["anomaly_scores"][i], 4),
            "signature_match": sig_match["signature_match"],
            "rule_name": sig_match["rule_name"],
            "intel": intel_data,
        }
        results.append(result)

        if log_alerts and fused["status"] in ("attack", "anomaly"):
            meta = {**row_meta, "intel": intel_data, "detection_source": fused["detection_source"]}
            alert_mgr.log_alert(
                flow_meta=meta,
                status=fused["status"],
                confidence=confidence,
                rule_name=sig_match["rule_name"],
                severity=fused["severity"],
                detection_source=fused["detection_source"],
            )

    return {
        "total_flows": len(results),
        "threats_detected": sum(1 for r in results if r["status"] == "attack"),
        "anomalies_detected": sum(1 for r in results if r["status"] == "anomaly"),
        "signature_hits": sum(1 for r in results if r["signature_match"]),
        "results": results,
    }
