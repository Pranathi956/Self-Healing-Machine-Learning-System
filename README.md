**Confidence-Aware Self-Healing ML System**
An autonomous machine learning pipeline that detects concept drift and self-corrects — without human intervention.
**Overview**
Production ML models silently degrade when real-world data distributions shift — a phenomenon known as Concept Drift. Most systems require manual human intervention to detect and fix this. This project builds a fully automated 3-tier self-healing pipeline that:

Monitors drift using PSI + KS-Test + AUC tracking
Detects confidence degradation using Expected Calibration Error (ECE)
Classifies severity (LOW / MODERATE / CRITICAL)
Heals itself autonomously — recalibrate, retrain, or fully replace the model

![Result](results/ss1.png)
