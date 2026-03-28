"""FastAPI app for wine quality prediction."""
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from src.monitoring import log_prediction, get_metrics, check_drift


MODEL_PATH = Path(__file__).parent.parent / "models" / "model.joblib"
STATIC_PATH = Path(__file__).parent.parent / "static"

app = FastAPI(title="Wine Quality API", version="0.1.0")

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_PATH / "index.html")


class WineFeatures(BaseModel):
    fixed_acidity: float = 7.4
    volatile_acidity: float = 0.7
    citric_acid: float = 0.0
    residual_sugar: float = 1.9
    chlorides: float = 0.076
    free_sulfur_dioxide: float = 11.0
    total_sulfur_dioxide: float = 34.0
    density: float = 0.9978
    pH: float = 3.51
    sulphates: float = 0.56
    alcohol: float = 9.4


class Prediction(BaseModel):
    quality: float
    drift: dict
    input_data: dict


FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.post("/predict", response_model=Prediction)
def predict(wine: WineFeatures):
    model = joblib.load(MODEL_PATH)
    values = [
        wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid,
        wine.residual_sugar, wine.chlorides, wine.free_sulfur_dioxide,
        wine.total_sulfur_dioxide, wine.density, wine.pH,
        wine.sulphates, wine.alcohol,
    ]
    df = pd.DataFrame([values], columns=FEATURE_NAMES)
    quality = round(float(model.predict(df)[0]), 2)

    input_data = wine.model_dump()
    drift = check_drift(input_data)
    log_prediction(input_data, quality)

    return Prediction(
        quality=quality,
        drift=drift,
        input_data=input_data,
    )
