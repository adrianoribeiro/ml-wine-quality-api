"""Tests for model training and quality."""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.train import load_data, train


def test_load_data_returns_dataframe():
    """load_data should return a DataFrame with expected shape."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "quality" in df.columns


def test_load_data_has_expected_columns():
    """Dataset should have the 12 expected columns."""
    df = load_data()
    expected = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
        "density", "pH", "sulphates", "alcohol", "quality",
    ]
    assert list(df.columns) == expected


def test_train_returns_model():
    """train() should return a fitted model."""
    model = train(n_estimators=10)  # fewer trees for speed
    assert model is not None
    assert hasattr(model, "predict")


def test_model_mae_below_threshold():
    """Model MAE should be below 0.6 (minimum acceptable quality)."""
    df = load_data()
    X = df.drop("quality", axis=1)
    y = df["quality"]

    model = train(n_estimators=10)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    assert mae < 0.6, f"MAE {mae:.3f} is above threshold 0.6"


def test_model_predictions_in_valid_range():
    """All predictions should be within wine quality range."""
    df = load_data()
    X = df.drop("quality", axis=1)

    model = train(n_estimators=10)
    predictions = model.predict(X)
    assert all(1 <= p <= 10 for p in predictions)
