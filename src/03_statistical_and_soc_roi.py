"""
03_statistical_and_soc_roi.py
Non-Parametric Wilcoxon Signed-Rank Hypothesis Testing & SOC Alert Fatigue ROI Modeling.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def compute_wilcoxon_significance(fold_df: pd.DataFrame) -> dict:
    """Compares paired 10-fold F1-score vectors between Proposed LightGBM and XGBoost."""
    lgbm_f1 = fold_df[fold_df["Model"] == "In-Fold SMOTE + LightGBM (Proposed)"]["F1_Score"].values
    xgb_f1 = fold_df[fold_df["Model"] == "In-Fold SMOTE + XGBoost"]["F1_Score"].values
    rf_f1 = fold_df[fold_df["Model"] == "In-Fold SMOTE + Random Forest"]["F1_Score"].values

    stat_xgb, p_xgb = wilcoxon(lgbm_f1, xgb_f1, alternative="two-sided")
    stat_rf, p_rf = wilcoxon(lgbm_f1, rf_f1, alternative="two-sided")

    return {
        "LightGBM_vs_XGBoost": {
            "W_statistic": float(stat_xgb),
            "p_value": float(p_xgb),
            "significant_at_0_05": bool(p_xgb < 0.05),
        },
        "LightGBM_vs_RandomForest": {
            "W_statistic": float(stat_rf),
            "p_value": float(p_rf),
            "significant_at_0_05": bool(p_rf < 0.05),
        },
    }


def compute_soc_alert_fatigue_roi(
    daily_events: int = 100000,
    attack_prevalence: float = 0.05,
    baseline_fpr: float = 0.15,
    proposed_fpr: float = 0.0426,
    triage_minutes_per_alert: float = 5.0,
) -> dict:
    """Quantifies daily false alarm reduction and Tier-1 SOC analyst-hours reclaimed."""
    benign_events = int(daily_events * (1.0 - attack_prevalence))

    baseline_false_alarms = round(benign_events * baseline_fpr)
    proposed_false_alarms = round(benign_events * proposed_fpr)
    alerts_eliminated = baseline_false_alarms - proposed_false_alarms
    noise_reduction_pct = (alerts_eliminated / baseline_false_alarms) * 100.0

    baseline_hours = (baseline_false_alarms * triage_minutes_per_alert) / 60.0
    proposed_hours = (proposed_false_alarms * triage_minutes_per_alert) / 60.0
    hours_saved_daily = baseline_hours - proposed_hours

    return {
        "daily_network_events": daily_events,
        "benign_daily_events": benign_events,
        "uncalibrated_baseline_fpr": baseline_fpr,
        "proposed_lightgbm_fpr": proposed_fpr,
        "baseline_daily_false_alarms": baseline_false_alarms,
        "proposed_daily_false_alarms": proposed_false_alarms,
        "false_alarms_eliminated_per_day": alerts_eliminated,
        "noise_reduction_percentage": round(noise_reduction_pct, 2),
        "baseline_analyst_hours_per_day": round(baseline_hours, 1),
        "proposed_analyst_hours_per_day": round(proposed_hours, 1),
        "analyst_hours_saved_per_day": round(hours_saved_daily, 1),
    }


def main():
    fold_path = os.path.join("results", "cv_fold_metrics.csv")
    if not os.path.exists(fold_path):
        raise FileNotFoundError("Run 02_train_and_evaluate_cv.py first.")

    fold_df = pd.read_csv(fold_path)
    wilcoxon_res = compute_wilcoxon_significance(fold_df)
    soc_roi_res = compute_soc_alert_fatigue_roi()

    report = {
        "wilcoxon_signed_rank_test": wilcoxon_res,
        "soc_operational_throughput_roi": soc_roi_res,
    }

    out_path = os.path.join("results", "soc_roi_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[DONE] Saved statistical significance & SOC ROI report to", out_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
