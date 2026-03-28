FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir pandas scikit-learn fastapi uvicorn joblib numpy

COPY src/ src/

RUN mkdir -p data && \
    curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" -o data/winequality-red.csv

RUN python src/train.py

EXPOSE 8000

CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
