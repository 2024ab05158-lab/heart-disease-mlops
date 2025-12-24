import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load trained pipeline
model = joblib.load("model.joblib")

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

@app.post("/predict")
def predict(data: Patient):
    X = np.array([list(data.dict().values())])
    prob = model.predict_proba(X)[0][1]

    return {
        "risk": int(prob > 0.5),
        "confidence": float(prob)
    }
