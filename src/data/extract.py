"""
MIMIC-III Data Extraction
Author: Subham Pal

Extracts ICU vitals and deterioration labels from MIMIC-III PostgreSQL database.
Target vitals: HR, BP (sys/dia), SpO2, RR, Temperature, GCS
Target label: deterioration event within next 6 hours
"""

import pandas as pd
import sqlalchemy
from loguru import logger
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PREDICTION_HORIZON_HOURS = 6

VITAL_ITEMIDS = {
    "heart_rate":       [211, 220045],
    "systolic_bp":      [51, 442, 455, 6701, 220179, 220050],
    "diastolic_bp":     [8368, 8440, 8441, 8555, 220180, 220051],
    "spo2":             [646, 220277],
    "respiratory_rate": [615, 618, 220210, 224690],
    "temperature":      [223761, 678],
    "gcs_total":        [198, 226755],
}


def get_db_engine(database_url: str) -> sqlalchemy.Engine:
    return sqlalchemy.create_engine(database_url)


def extract_vitals(engine: sqlalchemy.Engine) -> pd.DataFrame:
    """
    Extract all ICU vitals for adult patients from chartevents.
    Returns a long-format dataframe.
    """
    all_itemids = [id for ids in VITAL_ITEMIDS.values() for id in ids]
    itemids_str = ", ".join(map(str, all_itemids))

    query = f"""
        SELECT 
            ce.subject_id,
            ce.hadm_id,
            ce.icustay_id,
            ce.charttime,
            ce.itemid,
            ce.valuenum
        FROM mimiciii.chartevents ce
        INNER JOIN mimiciii.icustays icu ON ce.icustay_id = icu.icustay_id
        INNER JOIN mimiciii.patients p ON ce.subject_id = p.subject_id
        WHERE ce.itemid IN ({itemids_str})
          AND ce.valuenum IS NOT NULL
          AND ce.valuenum > 0
          AND (DATE_PART('year', icu.intime) - DATE_PART('year', p.dob)) >= 18
        ORDER BY ce.subject_id, ce.icustay_id, ce.charttime
    """

    logger.info("Extracting vitals from MIMIC-III...")
    df = pd.read_sql(query, engine)
    logger.info(f"Extracted {len(df):,} vital sign records")
    return df


def extract_deterioration_labels(engine: sqlalchemy.Engine) -> pd.DataFrame:
    """
    Define deterioration events: death in ICU or unplanned ICU transfer.
    Returns patient-level deterioration timestamps.
    """
    query = """
        SELECT 
            icu.subject_id,
            icu.hadm_id,
            icu.icustay_id,
            icu.intime,
            icu.outtime,
            adm.deathtime,
            CASE 
                WHEN adm.hospital_expire_flag = 1 THEN 1
                ELSE 0
            END AS deteriorated
        FROM mimiciii.icustays icu
        INNER JOIN mimiciii.admissions adm 
            ON icu.hadm_id = adm.hadm_id
    """

    logger.info("Extracting deterioration labels...")
    df = pd.read_sql(query, engine)
    logger.info(f"Extracted {len(df):,} ICU stay records")
    return df


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    engine = get_db_engine(os.getenv("DATABASE_URL"))
    vitals = extract_vitals(engine)
    labels = extract_deterioration_labels(engine)

    vitals.to_parquet(PROCESSED_DIR / "vitals_raw.parquet", index=False)
    labels.to_parquet(PROCESSED_DIR / "labels_raw.parquet", index=False)
    logger.success("Extraction complete.")
