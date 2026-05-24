# Model Deployment & Lifecycle Management — One-Pager

## Model Packaging & Deployment

Models are packaged using **BentoML** (for containerised serving) and **MLflow** (for experiment tracking and model registry). The trained model artifact (`.pkl` or `bentoml` bundle) is versioned and stored in the MLflow registry.

**Preferred deployment strategy:** Containerised REST API via BentoML, deployed on EC2 instance. BentoML provides built-in model serving, OpenTelemetry tracing, and horizontal scaling. For lighter deployments, Docker Compose suffices (see `deployment/docker-compose.yml`).

## Collaboration with DevOps/Engineering

- **Model handoff:** The model is registered in MLflow; the DevOps team pulls the latest production-ready version via the MLflow API.
- **Infrastructure:** Docker Compose files define the serving stack (API, MLflow).
- **CI/CD:** GitHub Actions pipeline runs linting, tests, model validation, and builds the Docker image. On merge to `main`, the image is pushed to ECR/GCR and deployed via ArgoCD or GitHub Actions.

## Model Maintenance & Iteration

- **Monitoring:** Track prediction drift and data drift using **Evidently AI** (or custom metrics with Prometheus). Key metrics: daily PSI (population stability index), feature distributions, prediction confidence.
- **Retraining triggers:** Retrain when (a) drift exceeds a threshold, (b) performance drops below a predefined ROC-AUC floor, or (c) new labelled data becomes available quarterly.
- **CI/CD for models:** The MLflow registry stores model versions; a promotion workflow moves candidates from staging → production after validation tests pass.

## Handling Business Conflicts & Overrides

- **Shadow mode:** New models run in shadow (predictions logged but not used) for one cycle before full rollout.
- **Confidence thresholds:** Low-confidence predictions (< 0.6 probability) are flagged for human review. Human overrides are logged and fed into the training set for the next retraining cycle.
- **Feedback loop:** A labelled dataset of override decisions is maintained; this is used as holdout validation and periodically mixed into retraining data to reduce future conflict.
