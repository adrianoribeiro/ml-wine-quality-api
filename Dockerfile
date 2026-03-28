FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir pandas scikit-learn fastapi uvicorn joblib

RUN pip install --no-cache-dir numpy

COPY src/ src/
COPY data/ data/

RUN python src/train.py

EXPOSE 8000

CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
