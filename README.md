# Wine Quality Prediction API

End-to-end ML Engineering project that predicts red wine quality from chemical properties.
The focus is not the model itself, but the **engineering infrastructure** around it.

**Live demo:** [ml-wine-quality-api-production.up.railway.app](https://ml-wine-quality-api-production.up.railway.app/)

## Project Structure

```
wine-quality-mle/
├── .github/workflows/ci.yml  # CI/CD pipeline (GitHub Actions)
├── data/                     # Dataset (tracked by DVC)
├── models/                   # Trained model artifacts
├── src/
│   ├── train.py              # Training pipeline with MLflow tracking
│   ├── api.py                # FastAPI prediction service
│   └── monitoring.py         # Drift detection and metrics
├── static/
│   └── index.html            # Web interface
├── tests/
│   ├── test_api.py           # API tests (endpoints, validation, drift)
│   └── test_model.py         # Model tests (quality thresholds, data integrity)
├── docs/
│   └── ARCHITECTURE.md       # Detailed architecture documentation
├── Dockerfile                # Container packaging
├── pyproject.toml            # Python dependencies
└── .gitignore
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adrianoribeiro/ml-wine-quality-api.git
cd ml-wine-quality-api

# Install dependencies
pip install pandas scikit-learn fastapi uvicorn joblib numpy mlflow

# Download dataset
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" -o data/winequality-red.csv

# Train the model
python src/train.py

# Start the API
uvicorn src.api:app --port 8000
```

Open [localhost:8000](http://localhost:8000) for the web interface or [localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

## Running with Docker

```bash
docker build -t wine-quality-api .
docker run -p 8000:8000 wine-quality-api
```

## Tech Stack

| Tool | Role |
|------|------|
| **scikit-learn** | Model training (RandomForestRegressor) |
| **FastAPI** | REST API serving predictions |
| **MLflow** | Experiment tracking, parameter/metric logging, model registry |
| **DVC** | Dataset versioning |
| **Docker** | Containerization for reproducible deployments |
| **GitHub Actions** | CI/CD pipeline (test + Docker build on every push) |
| **pytest** | Automated testing (15 tests: API + model quality) |
| **Railway** | Cloud deployment |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| GET | `/metrics` | Prediction statistics (count, avg, min, max) |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| POST | `/predict` | Predict wine quality from chemical features |

### Predict Example

```bash
curl -X POST https://ml-wine-quality-api-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol": 12.0, "pH": 3.2, "sulphates": 0.8}'
```

Response:
```json
{
  "quality": 5.82,
  "drift": {
    "drift_detected": false,
    "alerts_count": 0,
    "alerts": []
  },
  "input_data": { ... }
}
```

## Monitoring and Drift Detection

Every prediction is checked against the training data distribution. The API raises alerts when input values fall outside expected ranges:

- **Warning:** value is more than 3 standard deviations from the training mean
- **Critical:** value is outside the training data range entirely

This helps detect **data drift** — when production data starts looking different from training data, indicating the model may need retraining.

## Data Versioning (DVC)

> **Note:** This is a portfolio/demo project. In a production environment, DVC would be configured
> with a remote storage backend (Amazon S3, Google Cloud Storage, Azure Blob) so that
> team members can share datasets via `dvc push` and `dvc pull`.
>
> In this project, the dataset is small (84KB) and publicly available from the
> [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality), so it is
> downloaded during the Docker build. DVC is configured to demonstrate the versioning workflow
> (pointer files, hash-based tracking) without requiring cloud infrastructure.

## Experiment Tracking (MLflow)

Each training run is logged to MLflow with:
- **Parameters:** model type, n_estimators, test_size, random_state
- **Metrics:** MAE (Mean Absolute Error), R2 score
- **Artifacts:** trained model file

Run `mlflow ui` locally to compare experiments at [localhost:5000](http://localhost:5000).

## CI/CD Pipeline

Every push to `main` triggers the GitHub Actions pipeline:

```
Push → Install dependencies → Download data → Train model → Run tests (15) → Build Docker image
```

If any test fails, the pipeline stops. Tests cover both software quality (API responses, validation) and model quality (MAE thresholds, prediction ranges).

## Model Performance

| Metric | Value |
|--------|-------|
| MAE | 0.422 |
| R2 | 0.539 |
| Algorithm | RandomForestRegressor (n_estimators=100) |
| Dataset | UCI Wine Quality - Red (1599 samples, 11 features) |
