"""Basic monitoring for predictions and data drift detection."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd


logger = logging.getLogger("wine-quality")
LOG_PATH = Path(__file__).parent.parent / "logs"
TRAIN_DATA_PATH = Path(__file__).parent.parent / "data" / "winequality-red.csv"

# Keep last 1000 predictions in memory
prediction_log: deque = deque(maxlen=1000)

# Training data stats (loaded once)
_train_stats: dict | None = None


def get_train_stats() -> dict:
    """Load training data statistics for drift comparison."""
    global _train_stats
    if _train_stats is None:
        df = pd.read_csv(TRAIN_DATA_PATH, sep=";")
        features = df.drop("quality", axis=1)
        _train_stats = {
            "mean": features.mean().to_dict(),
            "std": features.std().to_dict(),
            "min": features.min().to_dict(),
            "max": features.max().to_dict(),
        }
    return _train_stats


def log_prediction(input_data: dict, quality: float):
    """Log a prediction for monitoring."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": quality,
    }
    prediction_log.append(entry)
    logger.info(f"prediction={quality:.2f} alcohol={input_data.get('alcohol')}")


def get_metrics() -> dict:
    """Return current monitoring metrics."""
    if not prediction_log:
        return {
            "total_predictions": 0,
            "message": "No predictions yet",
        }

    predictions = [p["prediction"] for p in prediction_log]

    return {
        "total_predictions": len(prediction_log),
        "avg_prediction": round(np.mean(predictions), 3),
        "min_prediction": round(min(predictions), 3),
        "max_prediction": round(max(predictions), 3),
        "std_prediction": round(np.std(predictions), 3),
        "last_prediction_at": prediction_log[-1]["timestamp"],
    }


def check_drift(input_data: dict) -> dict:
    """Compare input data against training data distribution."""
    stats = get_train_stats()
    alerts = []

    feature_mapping = {
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
    }

    for api_name, value in input_data.items():
        train_name = feature_mapping.get(api_name, api_name)
        if train_name not in stats["mean"]:
            continue

        mean = stats["mean"][train_name]
        std = stats["std"][train_name]
        min_val = stats["min"][train_name]
        max_val = stats["max"][train_name]

        # Alert if value is more than 3 standard deviations from mean
        if std > 0 and abs(value - mean) > 3 * std:
            alerts.append({
                "feature": api_name,
                "value": value,
                "expected_range": f"{mean - 3*std:.2f} to {mean + 3*std:.2f}",
                "severity": "warning",
                "message": f"Value {value} is outside 3 standard deviations from training mean ({mean:.2f})",
            })

        # Alert if value is outside training data range
        if value < min_val or value > max_val:
            alerts.append({
                "feature": api_name,
                "value": value,
                "training_range": f"{min_val:.2f} to {max_val:.2f}",
                "severity": "critical",
                "message": f"Value {value} is outside training data range [{min_val:.2f}, {max_val:.2f}]",
            })

    return {
        "drift_detected": len(alerts) > 0,
        "alerts_count": len(alerts),
        "alerts": alerts,
    }
