import joblib
import numpy as np
import logging
import time
from typing import Dict

from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Prometheus Metrics
# -------------------------------------------------
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions',
    ['risk_level']
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Prediction confidence scores',
    ['risk_level']
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether the model is loaded (1) or not (0)'
)

# -------------------------------------------------
# App Initialization
# -------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps API for predicting heart disease risk",
    version="1.0.0"
)

# Load trained pipeline
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = joblib.load("model.joblib")
    MODEL_LOADED.set(1)
    logger.info("Model loaded successfully")
except Exception as e:
    MODEL_LOADED.set(0)
    logger.error(f"Failed to load model: {e}")
    raise

# -------------------------------------------------
# Request Monitoring Middleware
# -------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Update Prometheus metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)

    logger.info(
        f"Request: {request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Time: {process_time:.4f}s"
    )
    return response

# -------------------------------------------------
# Input Schema
# -------------------------------------------------
class Patient(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(data: Patient):
    X = np.array([list(data.dict().values())])

    prob = model.predict_proba(X)[0][1]
    risk = int(prob > 0.5)
    risk_level = "high" if risk == 1 else "low"

    # Update Prometheus metrics
    PREDICTION_COUNT.labels(risk_level=risk_level).inc()
    PREDICTION_CONFIDENCE.labels(risk_level=risk_level).observe(prob)

    logger.info(
        f"Prediction | Risk={risk} | Confidence={prob:.4f}"
    )

    return {
        "risk": risk,
        "confidence": float(prob)
    }

# -------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED._value.get() == 1.0}

# -------------------------------------------------
# Metrics Endpoint (Prometheus)
# -------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)