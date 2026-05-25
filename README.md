**Confidence-Aware Self-Healing ML System**
An autonomous machine learning pipeline that detects concept drift and self-corrects — without human intervention.
**Overview**
Production ML models silently degrade when real-world data distributions shift — a phenomenon known as Concept Drift. Most systems require manual human intervention to detect and fix this. This project builds a fully automated 3-tier self-healing pipeline that:

Monitors drift using PSI + KS-Test + AUC tracking
Detects confidence degradation using Expected Calibration Error (ECE)
Classifies severity (LOW / MODERATE / CRITICAL)
Heals itself autonomously — recalibrate, retrain, or fully replace the model
confidence-aware-self-healing-ml/
│
├── 📓 Notebooks
│   ├── 01_data_analysis.ipynb          # EDA, class imbalance, correlation analysis
│   ├── 02_baseline_model.ipynb         # RF + LR + XGBoost, time-based split
│   ├── 03_drift_detection.ipynb        # PSI + KS-Test, 3 drift scenarios
│   ├── 04_confidence_monitoring.ipynb  # ECE + Temperature Scaling, reliability diagrams
│   ├── 05_self_healing_system.ipynb    # 3-tier severity engine + healing pipeline
│   └── 06_evaluation_comparison.ipynb # 5-panel dashboard: Baseline vs Drifted vs Healed
│
├── outputs/
│   ├── baseline_metrics.json           # AUC, PR-AUC, F1 saved from NB02
│   ├── X_drift_s1/s2/s3.npy           # Drift scenario arrays from NB03
│   ├── confidence_summary.json         # ECE before/after, optimal T from NB04
│   └── healing_results.json            # Per-scenario healing outcomes from NB05
│
├── models/
│   ├── baseline_model.pkl
│   ├── healed_*_recalibrated.pkl       # LOW severity
│   ├── healed_model_incremental.pkl    # MODERATE severity
│   └── healed_model_full_replace.pkl   # CRITICAL severity
│
├── requirements.txt
└── README.md

