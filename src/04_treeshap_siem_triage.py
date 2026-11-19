"""
04_treeshap_siem_triage.py
Actionable TreeSHAP Attribution & Automated MITRE ATT&CK SIEM/SOAR Containment Triage.
"""

import os
import json
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

FEATURE_COLUMNS = [
    "http_external_uploads",
    "ext_email_bytes",
    "after_hours_logins",
    "usb_bytes_copied",
    "usb_connect_count",
    "sensitive_file_access",
]


def run_treeshap_siem_triage(df: pd.DataFrame, rules_path: str, random_state: int = 42):
    with open(rules_path, "r", encoding="utf-8") as f:
        containment_config = json.load(f)

    rule_map = {r["feature"]: r for r in containment_config["rules"]}

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    model = LGBMClassifier(
        n_estimators=150, num_leaves=31, learning_rate=0.08,
        random_state=random_state, n_jobs=-1, verbose=-1
    )
    model.fit(X_train_res, y_train_res)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = (
        pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Mean_Abs_SHAP": mean_abs_shap,
            "MITRE_ID": [rule_map[c]["mitre_id"] for c in FEATURE_COLUMNS],
            "Protocol_Layer": [rule_map[c]["network_or_bus_layer"] for c in FEATURE_COLUMNS],
            "Automated_SOAR_Action": [rule_map[c]["soar_action"] for c in FEATURE_COLUMNS],
        })
        .sort_values(by="Mean_Abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )

    # Generate actionable SIEM/SOAR alerts for high-risk test detections
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    flagged_indices = np.where(y_probs >= 0.50)[0][:10]

    actionable_alerts = []
    for idx in flagged_indices:
        sample_shap = shap_values[idx]
        top_feat_idx = int(np.argmax(sample_shap))
        top_feature = FEATURE_COLUMNS[top_feat_idx]
        matched_rule = rule_map[top_feature]

        actionable_alerts.append({
            "event_index": int(idx),
            "exfiltration_risk_score": round(float(y_probs[idx]), 4),
            "primary_egress_driver": top_feature,
            "local_shap_contribution": round(float(sample_shap[top_feat_idx]), 4),
            "mitre_attack_id": matched_rule["mitre_id"],
            "protocol_layer": matched_rule["network_or_bus_layer"],
            "severity": matched_rule["severity"],
            "triggered_soar_playbook": matched_rule["soar_action"],
        })

    return global_importance, actionable_alerts


def main():
    data_path = os.path.join("results", "cert_telemetry_10k.csv")
    rules_path = os.path.join("config", "siem_containment_rules.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Run 01_preprocess_and_feature_engineering.py first.")

    df = pd.read_csv(data_path)
    print("[INFO] Computing TreeSHAP attributions and mapping SIEM/SOAR playbooks...")
    global_imp, alerts = run_treeshap_siem_triage(df, rules_path)

    global_imp.to_csv(os.path.join("results", "treeshap_global_attribution.csv"), index=False)
    with open(os.path.join("results", "shap_siem_triage_alerts.json"), "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)

    print("\n=== Global TreeSHAP Egress Attribution & SIEM Containment Mapping ===")
    print(global_imp[["Feature", "Mean_Abs_SHAP", "MITRE_ID", "Protocol_Layer"]].to_string(index=False))


if __name__ == "__main__":
    main()
