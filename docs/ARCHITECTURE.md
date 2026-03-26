# Wine Quality MLE - Architecture

## Overview

End-to-end ML Engineering project that predicts wine quality from chemical properties.
The goal is not the model itself, but the **engineering infrastructure** around it.

## Project Structure

```
wine-quality-mle/
├── .github/workflows/       # CI/CD pipelines (GitHub Actions)
├── data/                    # Raw and processed datasets (tracked by DVC)
├── models/                  # Trained model artifacts (tracked by MLflow)
├── src/
│   ├── train.py             # Training pipeline
│   └── api.py               # FastAPI prediction service
├── tests/                   # Automated tests
├── docs/                    # Documentation
├── pyproject.toml           # Python dependencies
├── Dockerfile               # Container packaging
├── dvc.yaml                 # DVC pipeline definition
└── .gitignore               # Excludes data/ and models/ from git
```

## Tools and Their Roles

### 1. Git + GitHub
- **What it tracks:** Code, configuration, documentation
- **Why:** Version control for everything that is text/source code
- **Rule:** Never commit binary files (data, models) to git

### 2. DVC (Data Version Control)
- **What it tracks:** Datasets (CSV files in `data/`)
- **Why:** Datasets are too large for git. DVC stores them externally (local, S3, GCS)
  and keeps only a small `.dvc` pointer file in git.
- **How it works:**
  1. `dvc init` - initializes DVC in the repo
  2. `dvc add data/wine.csv` - tells DVC to track this file
  3. This creates `data/wine.csv.dvc` (pointer) and adds `data/wine.csv` to `.gitignore`
  4. The pointer file goes into git, the actual data goes to DVC storage
  5. Anyone who clones the repo runs `dvc pull` to get the data
- **Commands:**
  - `dvc add <file>` - start tracking a file
  - `dvc push` - upload data to remote storage
  - `dvc pull` - download data from remote storage
  - `dvc repro` - reproduce the full pipeline

### 3. MLflow
- **What it tracks:** Experiments, model metrics, model versions
- **Why:** When you train 50 models with different parameters, you need to know
  which one was best and why. MLflow logs everything automatically.
- **Components:**
  - **Tracking:** Logs parameters (n_estimators=100), metrics (MAE=0.42, R2=0.54),
    and artifacts (the model file) for each training run
  - **Model Registry:** Stores model versions with stages (Staging, Production, Archived).
    You can promote a model from Staging to Production when it's validated.
  - **UI:** Web dashboard at localhost:5000 to compare experiments visually
- **How it works in our code:**
  ```python
  import mlflow

  with mlflow.start_run():
      # Log what parameters we used
      mlflow.log_param("n_estimators", 100)
      mlflow.log_param("test_size", 0.2)

      # Train model
      model.fit(X_train, y_train)

      # Log how well it performed
      mlflow.log_metric("mae", 0.422)
      mlflow.log_metric("r2", 0.539)

      # Save the model to MLflow (not just a local file)
      mlflow.sklearn.log_model(model, "model")
  ```
- **Model Registry workflow:**
  1. Train model -> automatically logged as a new run
  2. If metrics are good -> register as new model version
  3. Test in staging -> promote to production
  4. API loads the "Production" model from registry instead of a local file

### 4. FastAPI
- **What it does:** Serves the trained model as an HTTP API
- **Why:** Makes the model usable by any application (web, mobile, other services)
- **Endpoints:**
  - `GET /health` - is the service alive?
  - `POST /predict` - send wine features, get quality prediction
  - `GET /docs` - interactive API documentation (Swagger UI)

### 5. Docker (next step)
- **What it does:** Packages the API + model into a container
- **Why:** "Works on my machine" is not acceptable. Docker ensures the same
  environment everywhere (your laptop, CI/CD, production server).

### 6. GitHub Actions (next step)
- **What it does:** Automated CI/CD pipeline
- **Why:** Every push to the repo automatically runs tests, builds Docker image,
  and can deploy to production. No manual steps.
- **Pipeline:**
  1. Push code -> triggers workflow
  2. Run linting (code quality)
  3. Run tests (pytest)
  4. Build Docker image
  5. Deploy (if on main branch)

### 7. Monitoring (next step)
- **What it does:** Tracks model performance in production
- **Why:** Models degrade over time (data drift). You need to know when to retrain.
- **Tools:** Prometheus (metrics collection), Grafana (dashboards), Evidently (drift detection)

## Data Flow

```
[Raw Data] --DVC--> [Feature Engineering] --MLflow--> [Trained Model]
                                                           |
                                                      [Model Registry]
                                                           |
                                                   [FastAPI Service]
                                                           |
                                                     [Docker Container]
                                                           |
                                                  [Production Server]
                                                           |
                                                    [Monitoring]
```

## Implementation Roadmap

| Step | What | Tools | Status |
|------|------|-------|--------|
| 1 | Training pipeline + API | scikit-learn, FastAPI | Done |
| 2 | Git setup + first commit | Git, GitHub | In progress |
| 3 | Data versioning | DVC | Planned |
| 4 | Experiment tracking + model registry | MLflow | Planned |
| 5 | Automated tests | pytest | Planned |
| 6 | Containerization | Docker | Planned |
| 7 | CI/CD pipeline | GitHub Actions | Planned |
| 8 | Monitoring | Prometheus, Grafana | Planned |
