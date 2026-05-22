"""
SHAP Explainability Engine
Author: Subham Pal

Generates per-patient, per-prediction SHAP explanations.
Tells clinicians exactly which vital signs drove each alert.
"""

import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path


VITAL_NAMES = [
    "heart_rate", "systolic_bp", "diastolic_bp",
    "spo2", "respiratory_rate", "temperature", "gcs_total"
]


class SHAPExplainer:
    """
    Wraps a trained LSTM model with SHAP DeepExplainer.
    Produces per-timestep feature importance for each prediction.
    """

    def __init__(self, model, background_data: torch.Tensor):
        self.model = model
        self.model.eval()
        self.explainer = shap.DeepExplainer(model, background_data)
        logger.info("SHAP explainer initialized")

    def explain(self, X: torch.Tensor) -> dict:
        """
        Generate SHAP values for a batch of patients.
        Returns mean absolute SHAP value per vital sign.
        """
        shap_values = self.explainer.shap_values(X)
        # Average over timesteps → shape: (batch, n_vitals)
        mean_shap = np.abs(shap_values[0]).mean(axis=1)

        results = []
        for i in range(len(mean_shap)):
            patient_shap = {
                vital: float(mean_shap[i][j])
                for j, vital in enumerate(VITAL_NAMES)
            }
            sorted_shap = dict(
                sorted(patient_shap.items(), key=lambda x: x[1], reverse=True)
            )
            results.append({
                "shap_values": sorted_shap,
                "top_vital": list(sorted_shap.keys())[0],
                "top_3_vitals": list(sorted_shap.keys())[:3],
            })

        return results

    def plot_waterfall(self, patient_shap: dict, patient_id: str, save_dir: Path):
        """Generate waterfall plot for a single patient."""
        vitals = list(patient_shap["shap_values"].keys())
        values = list(patient_shap["shap_values"].values())

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]
        ax.barh(vitals, values, color=colors)
        ax.set_xlabel("SHAP Value (impact on deterioration probability)")
        ax.set_title(f"Patient {patient_id} — Deterioration Drivers")
        ax.axvline(x=0, color="black", linewidth=0.8)

        save_path = save_dir / f"shap_{patient_id}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"SHAP plot saved: {save_path}")
        return str(save_path)
