"""
Data Preprocessing Pipeline
Author: Subham Pal

Cleans and prepares MIMIC-III vitals for model training.
Handles: missing values, outlier removal, time windowing, label assignment.
"""

import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PREDICTION_HORIZON_HOURS = 6
WINDOW_SIZE_HOURS = 12
RESAMPLE_FREQ = "1H"

# Clinical valid ranges for outlier removal
VITAL_RANGES = {
    "heart_rate":       (0, 300),
    "systolic_bp":      (0, 300),
    "diastolic_bp":     (0, 200),
    "spo2":             (50, 100),
    "respiratory_rate": (0, 100),
    "temperature":      (25, 45),
    "gcs_total":        (3, 15),
}


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physiologically impossible values."""
    for vital, (low, high) in VITAL_RANGES.items():
        if vital in df.columns:
            before = len(df)
            df = df[(df[vital] >= low) & (df[vital] <= high)]
            removed = before - len(df)
            if removed:
                logger.warning(f"Removed {removed} outliers from {vital}")
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample irregular vital sign timestamps to regular 1-hour intervals.
    Uses forward-fill with a 2-hour limit for missing values.
    """
    df["charttime"] = pd.to_datetime(df["charttime"])
    df = df.set_index("charttime")

    resampled = (
        df.groupby("icustay_id")
        .resample(RESAMPLE_FREQ)
        .mean()
        .ffill(limit=2)        # Forward fill max 2 hours
    )

    logger.info(f"Resampled to hourly — {len(resampled):,} records")
    return resampled.reset_index()


def create_rolling_features(df: pd.DataFrame, vitals: list) -> pd.DataFrame:
    """
    Create rolling window features (mean, std, trend) over past N hours.
    These capture temporal deterioration patterns.
    """
    for vital in vitals:
        if vital not in df.columns:
            continue
        df[f"{vital}_mean_4h"] = df.groupby("icustay_id")[vital].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
        df[f"{vital}_std_4h"] = df.groupby("icustay_id")[vital].transform(
            lambda x: x.rolling(4, min_periods=1).std()
        )
        df[f"{vital}_trend_4h"] = df.groupby("icustay_id")[vital].transform(
            lambda x: x.diff(4)
        )

    logger.info("Rolling features created")
    return df


def assign_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    Assign binary deterioration labels.
    Label = 1 if deterioration occurs within next PREDICTION_HORIZON_HOURS.
    """
    df = df.merge(labels[["icustay_id", "deathtime", "deteriorated"]], on="icustay_id", how="left")

    df["deathtime"] = pd.to_datetime(df["deathtime"])
    df["charttime"] = pd.to_datetime(df["charttime"])

    df["label"] = (
        df["deteriorated"] &
        (df["deathtime"] - df["charttime"]).dt.total_seconds().between(
            0, PREDICTION_HORIZON_HOURS * 3600
        )
    ).astype(int)

    pos_rate = df["label"].mean() * 100
    logger.info(f"Label positive rate: {pos_rate:.2f}%")
    return df


if __name__ == "__main__":
    vitals = pd.read_parquet(PROCESSED_DIR / "vitals_raw.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels_raw.parquet")

    vitals = remove_outliers(vitals)
    vitals = resample_to_hourly(vitals)
    vitals = create_rolling_features(vitals, list(VITAL_RANGES.keys()))
    vitals = assign_labels(vitals, labels)

    vitals.to_parquet(PROCESSED_DIR / "vitals_processed.parquet", index=False)
    logger.success("Preprocessing complete.")
