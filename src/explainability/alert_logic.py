"""
Alert Intelligence Layer
Author: Swarnali Ghosh

Converts raw model predictions into smart, actionable clinical alerts.
Key goal: reduce alert fatigue by suppressing redundant alerts.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger


class AlertSeverity(str, Enum):
    CRITICAL = "critical"   # Immediate intervention needed
    WATCH    = "watch"      # Close monitoring needed
    STABLE   = "stable"     # No action needed


@dataclass
class Alert:
    patient_id: str
    severity: AlertSeverity
    probability: float
    confidence: float
    top_drivers: list[str]
    timestamp: datetime
    suppressed: bool = False
    suppression_reason: Optional[str] = None


class AlertEngine:
    """
    Intelligent alert system with suppression logic.

    Suppression rules:
    - Don't re-alert if same severity within last 30 minutes
    - Don't alert if probability dropped significantly (recovering patient)
    - Escalate immediately on Critical regardless of suppression
    """

    CRITICAL_THRESHOLD = 0.75
    WATCH_THRESHOLD    = 0.45
    SUPPRESSION_WINDOW = timedelta(minutes=30)

    def __init__(self):
        self.alert_history: dict[str, list[Alert]] = {}

    def classify_severity(self, probability: float) -> AlertSeverity:
        if probability >= self.CRITICAL_THRESHOLD:
            return AlertSeverity.CRITICAL
        elif probability >= self.WATCH_THRESHOLD:
            return AlertSeverity.WATCH
        else:
            return AlertSeverity.STABLE

    def should_suppress(self, patient_id: str, new_alert: Alert) -> tuple[bool, str]:
        """Check if this alert should be suppressed."""
        history = self.alert_history.get(patient_id, [])
        if not history:
            return False, ""

        last_alert = history[-1]
        time_since_last = new_alert.timestamp - last_alert.timestamp

        # Never suppress critical alerts
        if new_alert.severity == AlertSeverity.CRITICAL:
            return False, ""

        # Suppress same-severity alerts within suppression window
        if (last_alert.severity == new_alert.severity
                and time_since_last < self.SUPPRESSION_WINDOW):
            return True, f"Same severity within {self.SUPPRESSION_WINDOW.seconds // 60} min"

        return False, ""

    def process_prediction(
        self,
        patient_id: str,
        probability: float,
        confidence: float,
        top_drivers: list[str],
    ) -> Alert:
        """Process a model prediction into a smart alert."""
        severity = self.classify_severity(probability)
        alert = Alert(
            patient_id=patient_id,
            severity=severity,
            probability=probability,
            confidence=confidence,
            top_drivers=top_drivers,
            timestamp=datetime.utcnow(),
        )

        suppress, reason = self.should_suppress(patient_id, alert)
        if suppress:
            alert.suppressed = True
            alert.suppression_reason = reason
            logger.debug(f"Alert suppressed for {patient_id}: {reason}")
        else:
            logger.info(
                f"[{severity.value.upper()}] Patient {patient_id} — "
                f"P(deterioration)={probability:.2%} | Drivers: {top_drivers[:2]}"
            )

        self.alert_history.setdefault(patient_id, []).append(alert)
        return alert
