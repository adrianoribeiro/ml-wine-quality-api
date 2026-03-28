"""Train a wine quality prediction model with optional MLflow tracking."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


DATA_PATH = Path(__file__).parent.parent / "data" / "winequality-red.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "model.joblib"
EXPERIMENT_NAME = "wine-quality"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=";")


def train(n_estimators: int = 100, test_size: float = 0.2):
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow.start_run()

    print("1. Loading data...")
    df = load_data()
    print(f"   Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"\n2. Split: {len(X_train)} train, {len(X_test)} test")

    if MLFLOW_AVAILABLE:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "RandomForestRegressor")

    print(f"\n3. Training RandomForest (n_estimators={n_estimators})...")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if MLFLOW_AVAILABLE:
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

    # Always save a local copy for the API
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n4. Results:")
    print(f"   MAE: {mae:.3f}")
    print(f"   R2:  {r2:.3f}")
    print(f"   Model saved to: {MODEL_PATH}")
    if MLFLOW_AVAILABLE:
        mlflow.end_run()
        print(f"   Run logged to MLflow (experiment: {EXPERIMENT_NAME})")

    return model


if __name__ == "__main__":
    train()
