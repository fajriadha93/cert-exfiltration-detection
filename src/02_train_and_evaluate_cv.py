"""
02_train_and_evaluate_cv.py
Leakage-Free 10-Fold Stratified Cross-Validation Benchmark.

Compares uncalibrated baselines against strict within-fold StandardScaler + SMOTE
pipelines across Random Forest, XGBoost, and LightGBM.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

FEATURE_COLUMNS = [
    "http_external_uploads",
    "ext_email_bytes",
    "after_hours_logins",
    "usb_bytes_copied",
    "usb_connect_count",
    "sensitive_file_access",
]


def get_model_pipelines(random_state: int = 42) -> dict:
    """Returns uncalibrated baseline pipelines and leakage-free in-fold SMOTE pipelines."""
    rf_params = dict(n_estimators=150, max_depth=10, random_state=random_state, n_jobs=-1)
    xgb_params = dict(
        n_estimators=150, max_depth=6, learning_rate=0.08,
        eval_metric="logloss", random_state=random_state, n_jobs=-1
    )
    lgbm_params = dict(
        n_estimators=150, num_leaves=31, learning_rate=0.08,
        random_state=random_state, n_jobs=-1, verbose=-1
    )

    return {
        "Baseline RF (No Resampling)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**rf_params)),
        ]),
        "Baseline XGBoost (No Resampling)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(**xgb_params)),
        ]),
        "Baseline LightGBM (No Resampling)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(**lgbm_params)),
        ]),
        "In-Fold SMOTE + Random Forest": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", RandomForestClassifier(**rf_params)),
        ]),
        "In-Fold SMOTE + XGBoost": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", XGBClassifier(**xgb_params)),
        ]),
        "In-Fold SMOTE + LightGBM (Proposed)": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("clf", LGBMClassifier(**lgbm_params)),
        ]),
    }


def evaluate_leakage_free_cv(df: pd.DataFrame, n_splits: int = 10, random_state: int = 42):
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipelines = get_model_pipelines(random_state=random_state)

    fold_records = []
    summary_records = []

    for model_name, pipeline in pipelines.items():
        precisions, recalls, f1s, aucs, fprs = [], [], [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Fit StandardScaler and SMOTE strictly inside the training fold
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            y_prob = pipeline.predict_proba(X_val)[:, 1]

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            fpr = fp / (fp + tn)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_prob)

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            aucs.append(auc)
            fprs.append(fpr)

            fold_records.append({
                "Model": model_name,
                "Fold": fold_idx,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "ROC_AUC": auc,
                "FPR": fpr,
            })

        summary_records.append({
            "Model Strategy": model_name,
            "Precision": f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
            "Recall": f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
            "F1-Score": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "ROC-AUC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
            "FPR": f"{np.mean(fprs):.4f} ± {np.std(fprs):.4f} ({np.mean(fprs)*100:.2f}%)",
            "Mean_F1": np.mean(f1s),
            "Mean_FPR": np.mean(fprs),
        })

    return pd.DataFrame(fold_records), pd.DataFrame(summary_records)


def main():
    data_path = os.path.join("results", "cert_telemetry_10k.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Run 01_preprocess_and_feature_engineering.py first.")

    df = pd.read_csv(data_path)
    print("[INFO] Running leakage-free 10-fold stratified cross-validation...")
    fold_df, summary_df = evaluate_leakage_free_cv(df)

    fold_df.to_csv(os.path.join("results", "cv_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join("results", "cv_summary_metrics.csv"), index=False)

    print("\n=== 10-Fold Stratified Cross-Validation Summary ===")
    print(summary_df[["Model Strategy", "Precision", "Recall", "F1-Score", "FPR"]].to_string(index=False))


if __name__ == "__main__":
    main()
