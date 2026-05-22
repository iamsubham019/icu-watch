# 🏥 ICU Patient Deterioration Prediction

> An AI-powered early warning system that predicts patient clinical deterioration **6 hours in advance** using multivariate ICU vitals — with explainable, confidence-calibrated alerts designed to reduce alarm fatigue in critical care units.

---

## 🎯 Problem Statement

Hospitals already have monitoring systems. They fail because they generate hundreds of meaningless alerts — and nurses tune them out. The real problem isn't detection. It's **intelligent, calibrated, trustworthy alerting.**

This system addresses that gap by combining deep learning on time-series vitals with SHAP-based explainability, giving clinicians not just a warning — but a *reason*.

---

## 👥 Team

| Name | Role |
|------|------|
| **Subham Pal** | ML Core, Feature Engineering, Explainability (SHAP) |
| **Swarnali Ghosh** | Backend (FastAPI), Alert Logic, React Dashboard |

---

## 🧠 What It Does

- Ingests multivariate ICU vitals: **HR, BP, SpO2, Respiratory Rate, Temperature, GCS**
- Predicts probability of deterioration event within the next **6 hours**
- Beats the clinical baseline (**NEWS2 score**) on AUROC
- Explains every prediction with **SHAP values** — per patient, per vital sign
- Classifies alert severity: 🔴 Critical / 🟡 Watch / 🟢 Stable
- Suppresses redundant alerts — directly solving the alert fatigue problem

---

## 🏗️ Architecture

```
MIMIC-III Data
     │
     ▼
Data Pipeline (extract.py → preprocess.py → feature_store.py)
     │
     ▼
ML Models (Baseline NEWS2 → LSTM → Temporal Fusion Transformer)
     │
     ▼
Explainability Engine (SHAP per prediction)
     │
     ▼
FastAPI Backend (REST endpoints + alert logic)
     │
     ▼
React Dashboard (real-time vitals + predictions + SHAP charts)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Data | MIMIC-III (PhysioNet), PostgreSQL |
| ML | PyTorch, scikit-learn, SHAP |
| Backend | FastAPI, Pydantic, SQLAlchemy |
| Frontend | React, Recharts, TailwindCSS |
| Infrastructure | Docker, Docker Compose |
| Experiment Tracking | MLflow |

---

## 📂 Project Structure

```
icu-deterioration-ai/
├── data/
│   ├── raw/                  # MIMIC-III raw extracts (gitignored)
│   ├── processed/            # Cleaned, feature-engineered data
│   └── samples/              # Small anonymized samples for testing
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_deep_model.ipynb
├── src/
│   ├── data/                 # ETL pipeline
│   ├── models/               # ML models
│   ├── explainability/       # SHAP engine + alert logic
│   └── api/                  # FastAPI backend
├── dashboard/                # React frontend
├── research/                 # Papers, notes, experiments
├── tests/
├── docs/
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker + Docker Compose
- MIMIC-III access credentials (see below)

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/icu-deterioration-ai.git
cd icu-deterioration-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start full stack
docker-compose up --build
```

### MIMIC-III Data Access

This project uses the **MIMIC-III Clinical Database** from PhysioNet.

1. Complete CITI training at [physionet.org](https://physionet.org/content/mimiciii/)
2. Submit credentialing request (approval: 3–7 days)
3. Download and place data in `data/raw/`

> ⚠️ MIMIC-III data is **never committed to this repository**. The `data/raw/` directory is gitignored.

---

## 📊 Evaluation Metrics

- **AUROC** — primary metric for deterioration prediction
- **Precision-Recall AUC** — accounts for class imbalance
- **Calibration Curve** — ensures confidence scores are trustworthy
- **Alert Precision** — how many alerts were clinically meaningful

---

## 🗓️ Development Timeline

| Week | Subham | Swarnali |
|------|--------|----------|
| 1 | MIMIC extraction + EDA | Repo setup + FastAPI skeleton |
| 2 | Feature engineering pipeline | Patient schema + API endpoints |
| 3 | Baseline NEWS2 model | Alert logic layer |
| 4 | LSTM model trained + evaluated | Dashboard vitals display |
| 5 | SHAP engine complete | SHAP visualization on dashboard |
| 6 | Model API integration | Docker + full stack running |
| 7 | Research notes documented | README + docs complete |
| 8 | **Full system demo** | **Full system demo** |

---

## 📄 Research

All research papers, experiment logs, and notes are stored in `/research`.

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## ⚠️ Disclaimer

This system is a **research prototype** and is **not intended for clinical use**. All predictions must be validated by qualified medical professionals before any real-world application.
