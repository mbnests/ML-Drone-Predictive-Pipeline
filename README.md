# Autonomous Drone Predictive Failure Pipeline

## Overview
This repository contains an end-to-end Machine Learning classification pipeline built with **Python**, **Scikit-Learn**, and **XGBoost**. It simulates autonomous drone operational telemetry (wind speed, motor RPM, battery temperature, and vibration levels) to predict in-flight component failure risks before they occur.

To mirror real-world production engineering, this project implements strict data separation, automated scaling, and exhaustive hyperparameter optimization to prevent overfitting and data leakage.

## Core Engineering Pillars Demonstrated
1. **Automated Feature Standardization (`StandardScaler`):** Normalizes multi-scale telemetry inputs so that high-magnitude features do not distort the gradient updates of tree-based models.
2. **Scikit-Learn Pipelines:** Encapsulates preprocessing and modeling into a single cohesive object, guaranteeing that data scaling is computed strictly within cross-validation folds to avoid data leakage.
3. **Exhaustive Hyperparameter Tuning (`GridSearchCV`):** Automatically searches through a multi-dimensional grid of estimator counts, depths, and learning rates to identify the optimal model configuration.
4. **5-Fold Cross-Validation:** Robustly validates model performance across multiple data subsets to prove generalizability on unseen data.
5. **Ensemble Modeling (XGBoost):** Leverages gradient-boosted decision trees for high-precision tabular classification.

## Technology Stack
* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Wrangling:** Pandas, NumPy

## Execution
Run the complete training, tuning, and evaluation pipeline from your terminal:
```bash
python drone_predictive_pipeline.py
