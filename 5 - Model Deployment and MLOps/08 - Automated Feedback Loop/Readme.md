# Continuous Learning MLOps Feedback Loop

## 1. Project Overview

In production, machine learning models degrade over time due to shifts in environmental data distributions. This project demonstrates a closed-loop system that simulates a live production data stream, evaluates real-time model performance, detects statistical data drift, and automatically triggers an orchestration pipeline to retrain and update the model without human intervention.

### Key Features

- **Dynamic Stream Simulation:** Simulates steady-state production data versus sudden environmental shifts.
- **Statistical Monitoring:** Uses the Kolmogorov-Smirnov (KS) test to evaluate feature distribution drift.
- **Automated Loop Closure:** Triggers the retraining pipeline when performance drops or feature drift is detected.

---

## 2. Component Architecture and System Nuances

The project is structured modularly to separate configuration, pipeline orchestration, and runtime monitoring.

```text
       +------------------+
       |     main.py      | <--- Simulates production stream (Normal & Drifted)
       +--------+---------+
                |
        (Passes live records)
                v
       +------------------+
       |    monitor.py    | <--- Performs performance evaluation & KS test
       +--------+---------+
                |
     (If alert is triggered)
                v
       +------------------+
       |   pipeline.py    | <--- Trains challenger, evaluates sensitivity/recall
       +--------+---------+
                |
     (If approved -> overwrites)
                v
       +------------------+
       |   artifacts/     | <--- Model binaries (.joblib) & reference baseline (.csv)
       +------------------+
```

## 3. Components Detailed

- **config.py**: Centralizes operational thresholds such as p-value tolerance and minimum recall targets.
- **pipeline.py**: Handles feature tracking, model training, validation scoring, and deployment promotion logic.
- **monitor.py**: Computes production performance continuously and performs two checks:
  - **Performance Degradation:** Detects when recall falls below the defined threshold.
  - **Statistical Feature Drift:** Runs a two-sample KS test for each feature against the saved reference data.
- **main.py**: Acts as the orchestration engine, managing initialization, normal traffic, drift injection, and loop closure.

## 4. Toy Architecture vs. Enterprise Production MLOps

While this repository demonstrates the logic of automated retraining, a production-grade system scales these ideas into distributed infrastructure.

- **Data & Feature Storage**
  - Toy approach: Flat CSV files stored locally in the artifacts folder.
  - Enterprise approach: Centralized feature stores such as Feast or Hopsworks.

- **Compute & Pipeline Orchestration**
  - Toy approach: Sequential Python script execution through main.py.
  - Enterprise approach: Production orchestrators such as Prefect or Apache Airflow.

- **Experiment & Artifact Tracking**
  - Toy approach: Local .joblib files are overwritten on disk.
  - Enterprise approach: Model registries such as MLflow or Weights & Biases track versions and experiments.

- **Deployment & Serving**
  - Toy approach: Models are loaded directly from local serialized binaries.
  - Enterprise approach: Microservices and containerized deployment pipelines are used.

- **Deployment Strategies**
  - Toy approach: Direct replacement of the current local model file.
  - Enterprise approach: Canary or shadow deployments validate challenger models before promotion.

- **Observability & Logging**
  - Toy approach: Console prints and local metric calculations.
  - Enterprise approach: Distributed dashboards and drift monitoring tools such as Prometheus, Grafana, Evidently AI, and Whylogs.

## 5. Full End-to-End MLOps Feedback Loop

1. **Data Ingestion & Feature Engineering**
   - Raw sources are ingested and prepared for training and inference.
2. **Continuous Training (CT) Pipeline**
   - The system triggers retraining, tunes hyperparameters, and validates model performance.
3. **Model Governance & Deployment**
   - The model is approved, versioned, and promoted for serving.
4. **Observability & Closed-Loop Monitoring**
   - Live inference is monitored for drift and degradation, which triggers the feedback loop again.

```text
+-------------------------------------------------------------------------+
|                                                                         |
| 1. DATA INGESTION & FEATURE ENGINEERING                                 |
|    Raw Sources -> Streaming/Batch Ingestion -> Enterprise Store        |
|                                                                         |
| 2. CONTINUOUS TRAINING (CT) PIPELINE                                    |
|    Trigger -> Hyperparameter Tuning -> Validation (Recall/Curves)     |
|                                                                         |
| 3. MODEL GOVERNANCE & DEPLOYMENT                                        |
|    MLflow Registry -> CI/CD Gate -> Shadow Mode Serving                 |
|                                                                         |
| 4. OBSERVABILITY & SYSTEM CLOSED-LOOP                                  |
|    Live Inference -> Drift Monitoring -> Automated Webhook Alert        |
|                                                                         |
+-----------------------------> Webhook triggers retraining loop ---------+
```                                       