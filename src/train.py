"""Train a wine quality prediction model."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "model.joblib"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, sep=";")
    return df


def train():
    print("1. Loading data...")
    df = load_data()
    print(f"   Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    print(f"   Features: {list(X.columns)}")
    print(f"   Target: quality (range {y.min()} to {y.max()})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n2. Split: {len(X_train)} train, {len(X_test)} test")

    print("\n3. Training RandomForest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n4. Results:")
    print(f"   MAE: {mae:.3f}")
    print(f"   R2:  {r2:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n5. Model saved to: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train()
