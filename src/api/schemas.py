"""
API Schemas — Pydantic models for request/response validation
Author: Swarnali Ghosh
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WATCH = "watch"
    STABLE = "stable"


class VitalsInput(BaseModel):
    patient_id: str
    heart_rate: float = Field(..., ge=0, le=300, description="Heart rate in bpm")
    systolic_bp: float = Field(..., ge=0, le=300, description="Systolic BP in mmHg")
    diastolic_bp: float = Field(..., ge=0, le=200, description="Diastolic BP in mmHg")
    spo2: float = Field(..., ge=0, le=100, description="SpO2 percentage")
    respiratory_rate: float = Field(..., ge=0, le=100, description="Breaths per minute")
    temperature: float = Field(..., ge=25, le=45, description="Temperature in Celsius")
    gcs: Optional[int] = Field(None, ge=3, le=15, description="Glasgow Coma Scale")


class PredictionOutput(BaseModel):
    patient_id: str
    deterioration_probability: float = Field(..., ge=0, le=1)
    severity: AlertSeverity
    confidence: float = Field(..., ge=0, le=1)
    top_contributing_vitals: list[str]
    prediction_horizon_hours: int = 6


class SHAPExplanation(BaseModel):
    patient_id: str
    shap_values: dict[str, float]
    base_value: float
    prediction: float
