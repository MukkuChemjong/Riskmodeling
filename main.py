from fastapi import FastAPI
import joblib
import numpy as np
import os

import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model  = joblib.load(os.path.join(BASE_DIR, "models", "bankruptcy_model_v2.1.0.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler_v2.1.0.joblib"))

@app.post("/predict")
def predict(features: dict):
    X = np.array([list(features.values())])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[:, 1][0]
    return {"bankruptcy_probability": round(float(prob), 4)}

@app.get("/health")
def health():
    return {"status": "ok"}