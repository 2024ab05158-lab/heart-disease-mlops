import joblib
import numpy as np
import logging
import time

from fastapi import FastAPI, Request
from pydantic import BaseModel

# -------------------------------------------------
# Logging Configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -------------------------------------------------
# App Initialization
# -------------------------------------------------
app = FastAPI()

# Load trained pipeline
model = joblib.load("model.joblib")
logger.info("Model loaded successfully")

# -------------------------------------------------
# Request Monitoring Middleware
# -------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

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
    return {"status": "ok"}