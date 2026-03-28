"""Tests for the FastAPI prediction service."""
from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)

SAMPLE_WINE = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


def test_health():
    """Health endpoint should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_200():
    """Predict endpoint should return 200 for valid input."""
    response = client.post("/predict", json=SAMPLE_WINE)
    assert response.status_code == 200


def test_predict_returns_expected_fields():
    """Response should contain quality, drift and input_data fields."""
    response = client.post("/predict", json=SAMPLE_WINE)
    data = response.json()
    assert "quality" in data
    assert "drift" in data
    assert "input_data" in data


def test_predict_quality_is_float():
    """Quality should be a number."""
    response = client.post("/predict", json=SAMPLE_WINE)
    quality = response.json()["quality"]
    assert isinstance(quality, float)


def test_predict_quality_in_valid_range():
    """Quality should be within a reasonable wine score range (1-10)."""
    response = client.post("/predict", json=SAMPLE_WINE)
    quality = response.json()["quality"]
    assert 1 <= quality <= 10


def test_predict_with_default_values():
    """Predict should work with empty body (uses defaults)."""
    response = client.post("/predict", json={})
    assert response.status_code == 200
    assert 1 <= response.json()["quality"] <= 10


def test_predict_rejects_invalid_type():
    """Predict should return 422 for invalid input types."""
    response = client.post("/predict", json={"alcohol": "not_a_number"})
    assert response.status_code == 422


def test_predict_no_drift_for_normal_input():
    """Normal input should not trigger drift alerts."""
    response = client.post("/predict", json=SAMPLE_WINE)
    drift = response.json()["drift"]
    assert drift["drift_detected"] is False


def test_predict_drift_for_extreme_input():
    """Extreme values should trigger drift alerts."""
    extreme_wine = SAMPLE_WINE.copy()
    extreme_wine["alcohol"] = 99.0
    response = client.post("/predict", json=extreme_wine)
    drift = response.json()["drift"]
    assert drift["drift_detected"] is True
    assert drift["alerts_count"] > 0


def test_metrics_endpoint():
    """Metrics endpoint should return prediction statistics."""
    # Make a prediction first
    client.post("/predict", json=SAMPLE_WINE)
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert data["total_predictions"] > 0
