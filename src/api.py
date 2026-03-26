"""FastAPI app for wine quality prediction."""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path


MODEL_PATH = Path(__file__).parent.parent / "models" / "model.joblib"

app = FastAPI(title="Wine Quality API", version="0.1.0")


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
    input_data: dict


FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol",
]


@app.get("/health")
def health():
    return {"status": "ok"}


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
    quality = model.predict(df)[0]
    return Prediction(
        quality=round(float(quality), 2),
        input_data=wine.model_dump(),
    )
