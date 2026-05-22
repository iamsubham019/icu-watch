"""
Baseline Model — NEWS2 (National Early Warning Score 2)
Author: Subham Pal

Replicates the clinical standard that hospitals currently use.
This is the benchmark our LSTM model must beat.

NEWS2 scoring: https://www.rcp.ac.uk/improving-care/resources/national-early-warning-score-news2/
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger


def compute_news2_score(row: pd.Series) -> int:
    """
    Compute NEWS2 score for a single patient observation.
    Each vital contributes 0–3 points based on deviation from normal range.
    """
    score = 0

    # Respiratory Rate
    rr = row.get("respiratory_rate", np.nan)
    if not np.isnan(rr):
        if rr <= 8:              score += 3
        elif 9 <= rr <= 11:      score += 1
        elif 12 <= rr <= 20:     score += 0
        elif 21 <= rr <= 24:     score += 2
        elif rr >= 25:           score += 3

    # SpO2
    spo2 = row.get("spo2", np.nan)
    if not np.isnan(spo2):
        if spo2 <= 91:           score += 3
        elif 92 <= spo2 <= 93:   score += 2
        elif 94 <= spo2 <= 95:   score += 1
        elif spo2 >= 96:         score += 0

    # Systolic BP
    sbp = row.get("systolic_bp", np.nan)
    if not np.isnan(sbp):
        if sbp <= 90:            score += 3
        elif 91 <= sbp <= 100:   score += 2
        elif 101 <= sbp <= 110:  score += 1
        elif 111 <= sbp <= 219:  score += 0
        elif sbp >= 220:         score += 3

    # Heart Rate
    hr = row.get("heart_rate", np.nan)
    if not np.isnan(hr):
        if hr <= 40:             score += 3
        elif 41 <= hr <= 50:     score += 1
        elif 51 <= hr <= 90:     score += 0
        elif 91 <= hr <= 110:    score += 1
        elif 111 <= hr <= 130:   score += 2
        elif hr >= 131:          score += 3

    # Temperature
    temp = row.get("temperature", np.nan)
    if not np.isnan(temp):
        if temp <= 35.0:         score += 3
        elif 35.1 <= temp <= 36.0: score += 1
        elif 36.1 <= temp <= 38.0: score += 0
        elif 38.1 <= temp <= 39.0: score += 1
        elif temp >= 39.1:       score += 2

    # GCS (consciousness)
    gcs = row.get("gcs_total", np.nan)
    if not np.isnan(gcs):
        if gcs < 15:             score += 3

    return score


def classify_news2(score: int) -> str:
    """Map NEWS2 score to clinical alert category."""
    if score >= 7:    return "critical"
    elif score >= 5:  return "watch"
    else:             return "stable"


def evaluate_baseline(df: pd.DataFrame) -> dict:
    """
    Compute NEWS2 scores and evaluate against ground truth labels.
    Returns evaluation metrics.
    """
    logger.info("Computing NEWS2 baseline scores...")
    df["news2_score"] = df.apply(compute_news2_score, axis=1)
    df["news2_severity"] = df["news2_score"].apply(classify_news2)

    # Normalize score to [0,1] for AUROC comparison
    max_score = 20
    df["news2_prob"] = df["news2_score"] / max_score

    auroc = roc_auc_score(df["label"], df["news2_prob"])
    auprc = average_precision_score(df["label"], df["news2_prob"])

    metrics = {
        "model": "NEWS2 Baseline",
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
    }

    logger.info(f"NEWS2 Baseline — AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    logger.info("Our LSTM must beat this score to justify the AI approach.")
    return metrics


if __name__ == "__main__":
    from pathlib import Path
    df = pd.read_parquet(Path("data/processed/vitals_processed.parquet"))
    metrics = evaluate_baseline(df)
    print(metrics)
