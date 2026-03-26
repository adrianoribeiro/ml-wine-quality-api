# Wine Quality MLE

End-to-end ML Engineering project that predicts wine quality from chemical properties.
The goal is not the model itself, but the **engineering infrastructure** around it.

## Project Structure

```
wine-quality-mle/
├── .github/workflows/       # CI/CD pipelines (GitHub Actions)
├── data/                    # Raw and processed datasets (tracked by DVC)
├── models/                  # Trained model artifacts (tracked by MLflow)
├── src/
│   ├── train.py             # Training pipeline with MLflow tracking
│   └── api.py               # FastAPI prediction service
├── tests/                   # Automated tests
├── docs/                    # Additional documentation
├── pyproject.toml           # Python dependencies
├── Dockerfile               # Container packaging
└── .gitignore               # Excludes data/ and models/ from git
```

## Quick Start

```bash
# Install dependencies
pip install pandas scikit-learn fastapi uvicorn joblib mlflow dvc

# Train the model (downloads data, trains, logs to MLflow)
python src/train.py

# Serve the API
uvicorn src.api:app --port 8000

# Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0.0, "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0, "total_sulfur_dioxide": 34.0, "density": 0.9978, "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4}'

# View experiment history
mlflow ui
```

## Tools and Their Roles

| Tool | What it tracks | Why |
|------|---------------|-----|
| **Git + GitHub** | Code, configuration, docs | Version control for source code |
| **DVC** | Datasets | Data files are too large for git. DVC keeps a pointer in git and stores the actual file externally |
| **MLflow** | Experiments, metrics, models | Compare training runs, manage model versions, promote to production |
| **FastAPI** | - | Serves the model as an HTTP API so any application can use it |
| **Docker** | - | Packages everything into a container that runs the same way everywhere |
| **GitHub Actions** | - | Automates testing, building, and deployment on every push |

## Data Versioning (DVC)

> **Note:** This is a learning/demo project. In a production environment, DVC would be configured
> with a remote storage backend (Amazon S3, Google Cloud Storage, Azure Blob, etc.) so that
> team members can share datasets via `dvc push` and `dvc pull`.
>
> In this project, the dataset is small (84KB) and publicly available, so we include it in the
> `data/` directory for convenience. DVC is configured to demonstrate the versioning workflow
> (pointer files, `.gitignore` integration, hash-based tracking) without requiring cloud infrastructure.
>
> See [docs/DVC_SETUP.md](docs/DVC_SETUP.md) for details on how DVC is configured and how to
> set up a remote storage backend for production use.

## Experiment Tracking (MLflow)

Each training run is automatically logged to MLflow with:
- **Parameters:** model type, n_estimators, test_size, random_state
- **Metrics:** MAE (Mean Absolute Error), R2 score
- **Artifacts:** the trained model file

Run `mlflow ui` and open http://localhost:5000 to compare experiments.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict wine quality from chemical features |
| GET | `/docs` | Interactive API documentation (Swagger UI) |

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
| 2 | Data versioning | DVC | Done |
| 3 | Experiment tracking | MLflow | Done |
| 4 | Git setup + first commit | Git, GitHub | In progress |
| 5 | Automated tests | pytest | Planned |
| 6 | Containerization | Docker | Planned |
| 7 | CI/CD pipeline | GitHub Actions | Planned |
| 8 | Monitoring | Prometheus, Grafana | Planned |
