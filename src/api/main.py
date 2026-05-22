"""
ICU Deterioration Prediction API
Author: Swarnali Ghosh
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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


# ── Health Check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# ── Placeholder: Predict endpoint (Swarnali to implement) ─────
@app.post("/predict")
async def predict(patient_id: str):
    # TODO: integrate Subham's model here
    raise HTTPException(status_code=501, detail="Model integration pending")


# ── Placeholder: Patient vitals endpoint ──────────────────────
@app.get("/patient/{patient_id}/vitals")
async def get_vitals(patient_id: str):
    # TODO: query MIMIC-III PostgreSQL
    raise HTTPException(status_code=501, detail="DB integration pending")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
