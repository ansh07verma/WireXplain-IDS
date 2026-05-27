import pandas as pd
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SignatureDetector:
    def __init__(self):
        self.rules = []
        self._load_rules()

    def _load_rules(self):
        rules_path = Path(__file__).parent.parent / "config" / "rules.json"
        if rules_path.exists():
            try:
                self.rules = json.loads(rules_path.read_text())
                logger.info(f"Loaded {len(self.rules)} signature rules.")
            except Exception as e:
                logger.error(f"Failed to load rules.json: {e}")
        else:
            logger.warning("No rules.json found.")

    def _evaluate_rule(self, row, rule, cols_map):
        # A rule matches only if ALL conditions match (AND logic)
        for cond in rule.get("conditions", []):
            field_name = cond.get("field").lower().strip()
            col = cols_map.get(field_name)
            if not col:
                return False # Field not present in dataset
            
            val = row.get(col)
            if pd.isna(val):
                return False
                
            op = cond.get("operator")
            target = cond.get("value")
            
            # Try to convert to float for numeric comparison
            try:
                val_num = float(val)
                target_num = float(target)
                is_num = True
            except:
                is_num = False
                val = str(val).lower()
                target = str(target).lower()

            if is_num:
                if op == "==" and not (val_num == target_num): return False
                elif op == "!=" and not (val_num != target_num): return False
                elif op == ">" and not (val_num > target_num): return False
                elif op == "<" and not (val_num < target_num): return False
                elif op == ">=" and not (val_num >= target_num): return False
                elif op == "<=" and not (val_num <= target_num): return False
            else:
                if op == "==" and not (val == target): return False
                elif op == "!=" and not (val != target): return False
                else: return False # >, < not well supported for strings in this engine
            
        return True

    def predict(self, df_raw: pd.DataFrame) -> list[dict]:
        results = [{"signature_match": False, "rule_name": None, "severity": None} for _ in range(len(df_raw))]
        
        if df_raw.empty:
            return results

        cols = {c.lower().strip(): c for c in df_raw.columns}
        src_ip_col = cols.get("src ip")
        dst_port_col = cols.get("dst port")
        
        # 1. Flow-level Aggregation Heuristics
        scanning_ips = []
        dos_ips = []
        if src_ip_col and dst_port_col:
            port_counts = df_raw.groupby(src_ip_col)[dst_port_col].nunique()
            scanning_ips = port_counts[port_counts > 50].index.tolist()
            
            flow_counts = df_raw[src_ip_col].value_counts()
            dos_ips = flow_counts[flow_counts > 500].index.tolist()

        # 2. Row-level JSON Rules & Aggregation assignment
        for idx, (_, row) in enumerate(df_raw.iterrows()):
            matched = False
            
            # Check JSON rules first
            for rule in self.rules:
                if self._evaluate_rule(row, rule, cols):
                    results[idx] = {
                        "signature_match": True,
                        "rule_name": rule["name"],
                        "severity": rule.get("severity", "medium")
                    }
                    matched = True
                    break
                    
            if matched:
                continue

            # Fallback to Aggregation Heuristics
            if src_ip_col:
                src_ip = row[src_ip_col]
                if src_ip in scanning_ips:
                    results[idx] = {
                        "signature_match": True,
                        "rule_name": "Heuristic: Port Scan",
                        "severity": "high"
                    }
                elif src_ip in dos_ips:
                    results[idx] = {
                        "signature_match": True,
                        "rule_name": "Heuristic: Possible DoS",
                        "severity": "critical"
                    }

        return results
