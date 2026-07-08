# MLOps in 1 Month — Practical + Interview-Ready Roadmap

**Goal:** Go from "I know ML" to "I can build, deploy, monitor, and retrain an ML system end-to-end" in 4 weeks, with the theory depth needed for Data Scientist / MLE interviews.

**How to use this:** Each week has a theme, daily topics, resource links, and a hands-on task. By the end of Week 4 you'll have one working end-to-end project (data → training → tracking → API → Docker → CI/CD → monitoring) — this is also your portfolio/interview talking point.

---

## WEEK 1 — Foundations: ML Lifecycle, Data Handling & Experiment Tracking

**Theme:** Understand the full MLOps lifecycle and get your data + tracking foundations right — this is where most interview questions start ("walk me through your ML pipeline").

| Day | Topic | Resource | Practical Task |
|---|---|---|---|
| 1 | MLOps lifecycle & phases (exploration → PoC → automation) | [ml-ops.org](https://ml-ops.org/) · [MLOps Principles](https://ml-ops.org/content/mlops-principles) | Sketch the end-to-end architecture diagram (data → feature → train → deploy → monitor) for a project you know |
| 2 | Designing ML systems (production thinking, not notebook thinking) | [Chip Huyen – Designing ML Systems (GitHub)](https://github.com/chiphuyen/dmls-book/tree/main) · [Stanford CS329S syllabus](https://stanford-cs329s.github.io/syllabus.html) | Read Ch.1-2, note down "training-serving skew" and "data leakage" concepts |
| 3 | Git for ML projects + data/storage formats | GitHub docs (built-in knowledge) | Set up a repo with proper structure: `/data`, `/src`, `/models`, `/notebooks`, `.gitignore` for large files |
| 4 | Data leakage (train-test leakage, time-based splits, label leakage) — **very common interview topic** | [Understanding Data Drift & Model Drift](https://www.datacamp.com/tutorial/understanding-data-drift-model-drift) | Take a dataset, deliberately create a leaky split vs a clean time-based split, compare metrics |
| 5 | Data versioning with DVC | [DVC docs](https://dvc.org) · [Intro to DVC article](https://codecut.ai/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-2/) | Version a dataset with DVC, push to a remote (Google Drive/S3), track a change |
| 6 | Experiment tracking — MLflow | [MLflow docs](https://mlflow.org/#features) · [MLflow + DagsHub tutorial](https://youtu.be/6ngxBkx05Fs) | Train 3 versions of a model, log params/metrics/artifacts in MLflow, compare runs in the UI |
| 7 | Experiment tracking — Weights & Biases (know both; W&B is asked in interviews for advanced tracking) | [W&B docs](https://wandb.ai/home) | Repeat Day 6's experiment logging in W&B, compare UX vs MLflow |

**Week 1 Interview concepts to nail:** train/test/validation leakage, why time-based splits matter for time-series/production data, what experiment tracking solves (reproducibility), MLflow components (Tracking, Projects, Models, Registry).

---

## WEEK 2 — Feature Engineering, Pipelines, Model Validation & Containerization

**Theme:** Move from a single script to a proper pipeline, add a feature store, validate models like a production engineer, and containerize everything.

| Day | Topic | Resource | Practical Task |
|---|---|---|---|
| 8 | ML pipelines (data ingestion → feature engineering → training → evaluation as separate stages) | [End-to-End ML Overview video](https://www.youtube.com/watch?v=gc1DGuzJPDA) | Refactor Week 1's script into modular pipeline stages (functions/classes) |
| 9 | Pipeline orchestration — Kedro (standardizes project structure) | Kedro docs | Rebuild the pipeline using Kedro's node/pipeline structure |
| 10 | Feature stores (offline vs online, training-serving skew, feature freshness) | [Feature Store site](https://www.featurestore.org/) · [Feast tutorial video](https://www.youtube.com/watch?v=iZ8R_EUf_pM) | Set up Feast locally, define one feature view, fetch offline + online features |
| 11 | Model validation & testing (unit tests, integration tests for ML code) | Pytest docs · [PyCaret](https://pycaret.org) | Write pytest unit tests for your feature transformation functions + a data schema validation test |
| 12 | AutoML (know it exists, when to use it, its limits) | [PyCaret](https://pycaret.org) | Run PyCaret's `compare_models()` on your dataset, compare against your manual model |
| 13 | Docker fundamentals for ML (images, multi-stage builds, GPU containers) | [Docker Tutorial for Beginners 2024](https://www.knowledgehut.com/blog/devops/docker-for-beginners) | Containerize your training pipeline: write a Dockerfile, build, run it |
| 14 | Docker for serving + review/catch-up day | same as above | Containerize a simple inference script; review Week 1-2 gaps |

**Week 2 Interview concepts to nail:** why feature stores exist (training-serving skew), online vs offline store latency tradeoffs, difference between model-as-file / model-as-service / model+processing coupling, why you containerize ML workloads (reproducibility, dependency isolation).

---

## WEEK 3 — Deployment, CI/CD/CT & Serving

**Theme:** This is the "MLOps" core that interviewers probe hardest — how a model actually gets from your laptop to production safely and repeatably.

| Day | Topic | Resource | Practical Task |
|---|---|---|---|
| 15 | Model packaging & serving with FastAPI | [Build & Deploy ML Churn model with FastAPI, MLflow, Docker, AWS](https://www.youtube.com/watch?v=luJ64trcCwc) | Wrap your trained model in a FastAPI `/predict` endpoint |
| 16 | MLflow Model Registry (staging → production, model versioning/lifecycle) | MLflow docs (Model Registry section) | Register your model in MLflow Registry, transition it staging → production |
| 17 | BentoML (alternative serving framework, commonly asked in interviews) | BentoML docs | Package the same model with BentoML, compare DX vs FastAPI+MLflow |
| 18 | CI concepts: code checks, data validation checks, pipeline unit tests, model regression tests | [GitHub Actions for ML Beginners (KDnuggets)](https://www.kdnuggets.com/github-actions-for-machine-learning-beginners) | Write a GitHub Actions workflow that runs pytest + linting on every push |
| 19 | CD concepts: model validation gates, automated deployment, A/B testing, rollback strategies | [MLOps on GitHub – Deploy & Automate ML Workflow with GitHub Actions](https://www.youtube.com/watch?v=u_rCPdZY2g4) | Extend the GitHub Actions workflow to build & push a Docker image on merge to main |
| 20 | CT (Continuous Training) + champion-challenger setups | [MLOps Pipeline using Apache Airflow](https://amanxai.com/2025/01/20/mlops-pipeline-using-apache-airflow/) | Design (diagram) a retraining trigger: new data arrives → pipeline retrains → validates against current "champion" model → promotes if better |
| 21 | Kubernetes basics (just enough for interviews: pods, deployments, auto-scaling, why K8s for ML serving) | [24 Most Popular AWS Services 2025](https://www.youtube.com/watch?v=G-4o0dclZeQ) (for cloud context) | Not mandatory to fully deploy — read + write a sample `deployment.yaml` for your containerized model |

**Week 3 Interview concepts to nail:** CI vs CD vs CT (this is asked constantly), blue-green / canary / shadow deployment, model registry stages, rollback strategy, why champion-challenger matters for safe retraining.

---

## WEEK 4 — Monitoring, Drift Detection, Retraining & Capstone

**Theme:** Close the loop — a model in production is not "done," it needs to be watched and refreshed. Then integrate everything into one capstone.

| Day | Topic | Resource | Practical Task |
|---|---|---|---|
| 22 | Types of drift: data drift, concept drift, model drift (univariate vs multivariate) | [What is data drift in ML (Evidently AI)](https://www.evidentlyai.com/ml-in-production/data-drift) · [Understanding Data Drift and Model Drift](https://www.datacamp.com/tutorial/understanding-data-drift-model-drift) | Simulate drift: shift your test data distribution, measure the metric degradation |
| 23 | Drift detection tools — Evidently AI | Evidently AI docs | Run an Evidently AI report comparing "reference" vs "current" data for your project |
| 24 | Alternative drift libraries + NannyML | [Frouros GitHub](https://github.com/IFCA-Advanced-Computing/frouros) · [NannyML: How to detect and resolve drift](https://www.youtube.com/watch?v=zkWDb2URdIQ) | Skim Frouros repo examples; compare its API to Evidently |
| 25 | Monitoring infra: Prometheus + Grafana, model fairness/robustness, cost/latency monitoring | Prometheus & Grafana docs | Set up basic Prometheus metrics scraping for your FastAPI app (request count, latency) + a Grafana dashboard |
| 26 | Retraining triggers (time-based, drift-based, performance-based, human-in-the-loop) + Airflow DAGs | Airflow docs | Write a simple Airflow DAG that: checks for drift → if triggered, reruns training pipeline |
| 27 | Bring it together — architecture review | [Figure: End-to-end MLOps architecture](https://ml-ops.org/) (from your notes) | Redraw your own end-to-end architecture diagram including feature store, registry, CI/CD, monitoring, retraining loop |
| 28-30 | **Capstone project**: One dataset → Feature Eng pipeline → Training pipeline (MLflow tracked) → FastAPI serving (Dockerized) → GitHub Actions CI → Evidently monitoring report | Reuse all above | Deploy to a free cloud tier (Render/DigitalOcean/Railway) so you have a live demo link for interviews |

**Week 4 Interview concepts to nail:** difference between data drift and concept drift (this trips people up — concept drift = P(y\|x) changes, data drift = P(x) changes), how you'd design a retraining trigger system, what metrics you'd monitor in production beyond accuracy (latency, cost, fairness).

---

## Extra High-Value Resources (use as needed, not sequentially)

**Courses / structured learning**
- [MLOps Zoomcamp by DataTalks Club](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK) — best free structured course, do this in parallel if you have extra time
- [MLOps Fundamentals (DataCamp)](https://app.datacamp.com/learn/skill-tracks/mlops-fundamentals)
- [Machine Learning in Production (DataCamp)](https://app.datacamp.com/learn/skill-tracks/machine-learning-in-production)

**Reference / cheat sheets**
- [Awesome MLOps (visenger)](https://github.com/visenger/awesome-mlops) and [Awesome MLOps (kelvins)](https://github.com/kelvins/awesome-mlops) — bookmark for tool discovery
- [Deploying ML Models: A Checklist](https://twolodzko.github.io/posts/ml-checklist.html) — great pre-deployment sanity check
- [Chip Huyen's ML Systems Design notes](https://huyenchip.com/machine-learning-systems-design/toc.html)

**Interview prep specifically**
- [Cracking ML Interviews with MLOps (GitHub)](https://github.com/shafaypro/CrackingMachineLearningInterview/tree/master/mlops)
- [Crack ML System Design Interviews Like a Pro](https://freedium.cfd/medium.com/the-algorithmic-minds/crack-ml-system-design-interviews-like-a-pro-part-1-9c1203b2014f)
- [How to Learn ML System Design & Land $300k+ Offers (Marina Wyss)](https://www.youtube.com/watch?v=cx2Jdhz3CnY)

---

## Topics from your notes intentionally de-prioritized (nice-to-know, not core for 1 month)
- Kubeflow, ZenML, Jenkins, DagSter, Metaflow, Prefect — these are alternative tools to what's already scheduled (Airflow/GitHub Actions/MLflow). Skim their docs in Week 4 if time permits, but don't build with them — interviewers care that you understand *why* each pipeline/orchestration layer exists, not that you've used every tool.
- Full production Kubernetes deployment — understand the concepts (Day 21) but a full K8s cluster deployment isn't realistic to master in 30 days alongside everything else.

---

## Weekly Self-Check (use this to know if you're on track)
- **End of Week 1:** Can you explain data leakage and show an MLflow experiment comparison?
- **End of Week 2:** Do you have a modular pipeline + Dockerized training step + a Feast feature view?
- **End of Week 3:** Do you have a live FastAPI endpoint + a passing GitHub Actions CI workflow + a registered model in MLflow?
- **End of Week 4:** Can you show a drift report, explain your retraining trigger design, and walk through your full architecture diagram end-to-end?
