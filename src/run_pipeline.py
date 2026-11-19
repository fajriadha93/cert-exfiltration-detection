"""
run_pipeline.py
Master execution script reproducing all experiments, statistical tests, SOC ROI,
and TreeSHAP-to-SIEM containment mappings.
"""

import subprocess
import sys

SCRIPTS = [
    "src/01_preprocess_and_feature_engineering.py",
    "src/02_train_and_evaluate_cv.py",
    "src/03_statistical_and_soc_roi.py",
    "src/04_treeshap_siem_triage.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n{'='*70}\nExecuting: {script}\n{'='*70}")
        subprocess.run([sys.executable, script], check=True)
    print("\n[SUCCESS] Full pipeline completed. All outputs saved in results/.")


if __name__ == "__main__":
    main()
