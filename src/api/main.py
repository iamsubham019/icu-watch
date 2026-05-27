"""
ICU Deterioration Prediction API
Authors: Swarnali Ghosh & Subham Pal
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import VitalsInput, PredictionOutput, AlertSeverity
import uvicorn
import random

app = FastAPI(
    title="ICU Deterioration Prediction API",
    description="Predicts patient deterioration 6 hours in advance using ICU vitals",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mock patient database ─────────────────────────────────────
MOCK_PATIENTS = {
    "P001": {
        "name": "Patient 001",
        "age": 67,
        "vitals": {
            "heart_rate": 112,
            "systolic_bp": 88,
            "diastolic_bp": 58,
            "spo2": 93,
            "respiratory_rate": 24,
            "temperature": 38.9,
            "gcs": 13
        }
    },
    "P002": {
        "name": "Patient 002",
        "age": 54,
        "vitals": {
            "heart_rate": 78,
            "systolic_bp": 122,
            "diastolic_bp": 80,
            "spo2": 98,
            "respiratory_rate": 16,
            "temperature": 37.1,
            "gcs": 15
        }
    },
    "P003": {
        "name": "Patient 003",
        "age": 71,
        "vitals": {
            "heart_rate": 98,
            "systolic_bp": 100,
            "diastolic_bp": 65,
            "spo2": 95,
            "respiratory_rate": 21,
            "temperature": 38.2,
            "gcs": 14
        }
    },
}


# ── Health Check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# ── Get all patients ──────────────────────────────────────────
@app.get("/patients")
async def get_patients():
    return {"patients": list(MOCK_PATIENTS.keys())}


# ── Get patient vitals ────────────────────────────────────────
@app.get("/patient/{patient_id}/vitals")
async def get_vitals(patient_id: str):
    if patient_id not in MOCK_PATIENTS:
        raise HTTPException(status_code=404, detail="Patient not found")
    return MOCK_PATIENTS[patient_id]


# ── Predict deterioration ─────────────────────────────────────
@app.get("/patient/{patient_id}/predict")
async def predict(patient_id: str):
    if patient_id not in MOCK_PATIENTS:
        raise HTTPException(status_code=404, detail="Patient not found")

    vitals = MOCK_PATIENTS[patient_id]["vitals"]

    # Mock prediction logic based on vitals
    risk_score = 0
    if vitals["heart_rate"] > 100:       risk_score += 0.2
    if vitals["systolic_bp"] < 100:      risk_score += 0.25
    if vitals["spo2"] < 95:              risk_score += 0.2
    if vitals["respiratory_rate"] > 20:  risk_score += 0.2
    if vitals["temperature"] > 38.5:     risk_score += 0.15

    risk_score = min(round(risk_score + random.uniform(-0.05, 0.05), 2), 1.0)

    if risk_score >= 0.75:
        severity = AlertSeverity.CRITICAL
    elif risk_score >= 0.45:
        severity = AlertSeverity.WATCH
    else:
        severity = AlertSeverity.STABLE

    return PredictionOutput(
        patient_id=patient_id,
        deterioration_probability=risk_score,
        severity=severity,
        confidence=round(random.uniform(0.75, 0.95), 2),
        top_contributing_vitals=["respiratory_rate", "systolic_bp", "spo2"],
        prediction_horizon_hours=6
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)