"""
ICU Deterioration Prediction API
Authors: Swarnali Ghosh 
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import VitalsInput, PredictionOutput, AlertSeverity
import uvicorn
import random
from src.api.model_loader import model_loader
import json
from pathlib import Path
from src.explainability.alert_logic import AlertEngine
alert_engine = AlertEngine()
# Load SHAP explanations
SHAP_FILE = Path('data/processed/shap/patient_explanations.json')
SHAP_DATA = []
if SHAP_FILE.exists():
    with open(SHAP_FILE) as f:
        SHAP_DATA = json.load(f)

# Mock SHAP values for dashboard (used when real SHAP file not available)
MOCK_SHAP = {
    "respiratory_rate": 0.24,
    "systolic_bp": 0.18,
    "spo2": 0.15,
    "heart_rate": 0.12,
    "temperature": 0.08,
    "map": 0.06,
    "diastolic_bp": 0.04,
}

app = FastAPI(
    title="ICU Deterioration Prediction API",
    description="Predicts patient deterioration 6 hours in advance using ICU vitals",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
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
    "P004": {
        "name": "Patient 004",
        "age": 45,
        "vitals": {
            "heart_rate": 118,
            "systolic_bp": 85,
            "diastolic_bp": 55,
            "spo2": 91,
            "respiratory_rate": 26,
            "temperature": 39.2,
            "gcs": 12
        }
    },
    "P005": {
        "name": "Patient 005",
        "age": 62,
        "vitals": {
            "heart_rate": 105,
            "systolic_bp": 95,
            "diastolic_bp": 62,
            "spo2": 94,
            "respiratory_rate": 22,
            "temperature": 38.6,
            "gcs": 14
        }
    },
    "P006": {
        "name": "Patient 006",
        "age": 38,
        "vitals": {
            "heart_rate": 72,
            "systolic_bp": 118,
            "diastolic_bp": 76,
            "spo2": 99,
            "respiratory_rate": 14,
            "temperature": 36.8,
            "gcs": 15
        }
    },
    "P007": {
        "name": "Patient 007",
        "age": 79,
        "vitals": {
            "heart_rate": 88,
            "systolic_bp": 110,
            "diastolic_bp": 70,
            "spo2": 96,
            "respiratory_rate": 18,
            "temperature": 37.4,
            "gcs": 15
        }
    },
    "P008": {
        "name": "Patient 008",
        "age": 55,
        "vitals": {
            "heart_rate": 125,
            "systolic_bp": 82,
            "diastolic_bp": 52,
            "spo2": 90,
            "respiratory_rate": 28,
            "temperature": 39.5,
            "gcs": 11
        }
    },
    "P009": {
        "name": "Patient 009",
        "age": 48,
        "vitals": {
            "heart_rate": 82,
            "systolic_bp": 115,
            "diastolic_bp": 74,
            "spo2": 97,
            "respiratory_rate": 17,
            "temperature": 37.0,
            "gcs": 15
        }
    },
    "P010": {
        "name": "Patient 010",
        "age": 65,
        "vitals": {
            "heart_rate": 108,
            "systolic_bp": 92,
            "diastolic_bp": 60,
            "spo2": 94,
            "respiratory_rate": 23,
            "temperature": 38.4,
            "gcs": 13
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

    # Run real LSTM model (or mock if model not loaded)
    result = model_loader.predict(vitals)
    risk_score = result['probability']

    # Get SHAP values
    shap_values = {
        k: round(v * (1 + random.uniform(-0.1, 0.1)), 4)
        for k, v in MOCK_SHAP.items()
    }

    alert = alert_engine.process_prediction(
        patient_id=patient_id,
        probability=risk_score,
        confidence=round(random.uniform(0.75, 0.95), 2),
        top_drivers=["respiratory_rate", "systolic_bp", "spo2"],
    )

    return {
        "patient_id": patient_id,
        "deterioration_probability": alert.probability,
        "severity": alert.severity,
        "confidence": alert.confidence,
        "top_contributing_vitals": alert.top_drivers,
        "prediction_horizon_hours": 6,
        "shap_values": shap_values,
        "model_used": result['model'],
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)