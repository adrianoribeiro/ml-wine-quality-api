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


def test_predict_returns_quality_field():
    """Response should contain a quality field."""
    response = client.post("/predict", json=SAMPLE_WINE)
    data = response.json()
    assert "quality" in data
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
