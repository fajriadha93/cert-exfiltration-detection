# Multi-Protocol Data Exfiltration Detection & Actionable TreeSHAP Triage

Official code repository for the manuscript:
> **"Suppressing Alert Fatigue in Multi-Protocol Data Exfiltration Detection via Leakage-Free Minority Synthesis and Actionable TreeSHAP Triage"**  
> *Fajri Adha, Bertha S. Djahi, Marianus Yakobus Lili Lejap, and Marlon Jusak Rinaldy Benu*  
> Faculty of Science and Engineering, Universitas Nusa Cendana, Kupang, Indonesia

---

## Overview

Stealthy insider data exfiltration across enterprise network egress channels (HTTP/S web proxies, SMTP mail gateways, and removable USB storage buses) is notoriously difficult to detect in Security Information and Event Management (SIEM) telemetry due to extreme class imbalance and stochastic log noise. Conventional oversampling pipelines often introduce cross-fold synthetic data leakage and boundary overfitting, inflating False Positive Rates (FPR) and causing severe alert fatigue in Security Operations Centers (SOC).

This repository implements an end-to-end, **leakage-free exfiltration triage pipeline** featuring:
1. **Multi-Protocol Telemetry Feature Engineering:** Aggregates network and endpoint audit logs mapped to MITRE ATT&CK enterprise techniques (`T1041`, `T1567`, `T1078`, `T1052`, `T1091`, `T1005`).
2. **Leakage-Free 10-Fold Stratified Evaluation:** Enforces strict within-fold `StandardScaler` fitting and `SMOTE` minority interpolation exclusively on training partitions while evaluating on native imbalanced validation folds.
3. **Histogram-Regularized Leaf-Wise Classification (LightGBM):** Suppresses synthetic boundary drift under 3.5% stochastic telemetry noise, achieving an **F1-Score of 0.5449 ± 0.0336** and compressing operational **FPR to 4.26%** (Wilcoxon Signed-Rank $p = 0.00195$).
4. **Actionable TreeSHAP-to-SIEM Containment Mapping:** Translates quantitative SHAP egress attributions directly into deterministic MITRE ATT&CK-aligned SIEM/SOAR containment playbooks.

---

## Repository Structure

```text
cert-exfiltration-detection/
├── README.md                                      # Project documentation & reproduction guide
├── LICENSE                                        # MIT License
├── requirements.txt                               # Python dependencies
├── .gitignore                                     # Git ignore rules
├── config/
│   └── siem_containment_rules.json                # MITRE ATT&CK & TreeSHAP-to-SIEM triage rules
├── data/
│   └── README_DATA.md                             # Instructions for CMU CERT r4.2 dataset (DOI)
├── results/                                       # Generated metrics, statistical tests & SHAP outputs
└── src/
    ├── 01_preprocess_and_feature_engineering.py   # CERT r4.2 log aggregation & benchmark builder
    ├── 02_train_and_evaluate_cv.py                # Leakage-free 10-fold CV (RF, XGBoost, LightGBM)
    ├── 03_statistical_and_soc_roi.py              # Wilcoxon Signed-Rank test & SOC alert fatigue ROI
    ├── 04_treeshap_siem_triage.py                 # TreeSHAP attribution & automated SOAR rule trigger
    └── run_pipeline.py                            # End-to-end execution script
```

---

## Dataset Attribution

The datasets analyzed in this study are derived from the publicly available **Carnegie Mellon University (CMU) CERT Insider Threat Test Dataset (r4.2)**:
- **Official DOI:** [https://doi.org/10.1184/R1/12841247.v1](https://doi.org/10.1184/R1/12841247.v1)
- **Citation:** Lindauer, B. (2020). *Insider Threat Test Dataset* [Data set]. Carnegie Mellon University Software Engineering Institute.

See `data/README_DATA.md` for instructions on placing the raw CERT r4.2 CSV files (`logon.csv`, `device.csv`, `email.csv`, `file.csv`, `http.csv`) or running the pipeline with the built-in 10,000-sample benchmark generator (9,187 benign records, 813 exfiltration instances, 3.5% stochastic telemetry noise).

---

## Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/fajriadha93/cert-exfiltration-detection.git
cd cert-exfiltration-detection
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Complete Pipeline
```bash
python src/run_pipeline.py
```

Running `src/run_pipeline.py` sequentially executes:
1. Telemetry preprocessing & feature extraction (`results/cert_telemetry_10k.csv`)
2. Leakage-free 10-fold cross-validation comparing uncalibrated baselines and in-fold SMOTE ensembles (`results/cv_summary_metrics.csv`)
3. Non-parametric Wilcoxon Signed-Rank hypothesis testing & SOC analyst-hour ROI modeling (`results/soc_roi_report.json`)
4. TreeSHAP global attribution and automated SIEM/SOAR containment alert generation (`results/shap_siem_triage_alerts.json`)

---

## Citation

If you use this code or experimental framework in your research, please cite:

```bibtex
@article{adha2026suppressing,
  title={Suppressing Alert Fatigue in Multi-Protocol Data Exfiltration Detection via Leakage-Free Minority Synthesis and Actionable TreeSHAP Triage},
  author={Adha, Fajri and Djahi, Bertha S. and Lejap, Marianus Yakobus Lili and Benu, Marlon Jusak Rinaldy},
  year={2026},
  institution={Universitas Nusa Cendana}
}
```

## Contact
- **Corresponding Author:** Fajri Adha (`fajri.adha@staf.undana.ac.id`) — ORCID: [0009-0004-7642-9222](https://orcid.org/0009-0004-7642-9222)
