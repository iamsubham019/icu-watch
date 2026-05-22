<div align="center">

<img src="https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />

<br/><br/>

# 🏥 ICU-Watch

### AI-Powered ICU Patient Deterioration Prediction

*Predicting clinical deterioration **6 hours in advance** — giving clinicians time to act, not just react.*

<br/>

[📖 Documentation](#-getting-started) • [🏗️ Architecture](#-system-architecture) • [📊 Results](#-evaluation-metrics) • [🤝 Contributing](#-team)

</div>

---

## 📌 The Problem

> **Hospitals already have monitoring systems. They fail.**

Every ICU nurse manages 2–4 critically ill patients simultaneously. Existing systems generate **hundreds of meaningless alerts per shift** — and nurses, overwhelmed, begin to tune them out. This is called **alarm fatigue**, and it kills people.

The real problem isn't *detection*. It's **intelligent, calibrated, trustworthy alerting.**

When a patient's condition starts to deteriorate, the window to intervene is narrow. Sepsis caught 6 hours early has a survival rate dramatically higher than sepsis caught at crisis point. Current rule-based systems miss the subtle, multivariate patterns that precede deterioration.

**ICU-Watch** bridges that gap.

---

## 💡 Our Solution

ICU-Watch is an end-to-end AI system that:

- 📈 **Ingests** continuous multivariate ICU vitals in real-time
- 🧠 **Predicts** deterioration probability 6 hours before it happens
- 🔍 **Explains** every prediction with SHAP values — *which vital sign drove this alert, and why*
- 🔔 **Alerts** intelligently — suppressing redundant noise, escalating what truly matters
- 📊 **Visualizes** patient trajectories on a clean clinical dashboard

Unlike rule-based systems, ICU-Watch learns complex temporal patterns across multiple vital signs simultaneously — the kind of subtle cascade that precedes sepsis, cardiac events, or respiratory failure.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MIMIC-III Database                       │
│         53,423 ICU admissions · 38,597 adult patients           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Pipeline                              │
│   extract.py → preprocess.py → feature_store.py                │
│   • Outlier removal    • Hourly resampling                      │
│   • Rolling features   • Label assignment (6h horizon)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML Models                                 │
│                                                                 │
│   [NEWS2 Baseline] ──→ [Logistic Regression] ──→ [LSTM]        │
│                                                     │           │
│   Bidirectional LSTM + Temporal Attention           │           │
│   Input: (batch, 12h sequence, 7 vitals)            │           │
│   Output: P(deterioration within 6h)                │           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Explainability Engine                         │
│   SHAP DeepExplainer → per-patient, per-vital importance       │
│   "SpO2 drop + rising HR drove this Critical alert"            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Alert Intelligence                           │
│   🔴 Critical (P ≥ 0.75) · 🟡 Watch (P ≥ 0.45) · 🟢 Stable   │
│   Smart suppression — no redundant alerts within 30 min        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────┬──────────────────────────────────────┐
│     FastAPI Backend      │         React Dashboard              │
│   REST endpoints         │   Real-time vitals charts            │
│   Pydantic validation    │   Prediction timeline                │
│   PostgreSQL queries     │   SHAP waterfall per patient         │
└──────────────────────────┴──────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data** | MIMIC-III, PostgreSQL | Clinical database, 53K+ ICU admissions |
| **ML Core** | PyTorch 2.3, scikit-learn | LSTM model, baseline models |
| **Explainability** | SHAP | Per-prediction vital sign importance |
| **Experiment Tracking** | MLflow | Model versioning, metric logging |
| **Backend** | FastAPI, Pydantic, SQLAlchemy | REST API, data validation |
| **Frontend** | React 18, Recharts, TailwindCSS | Clinical dashboard |
| **Infrastructure** | Docker, Docker Compose | Containerized full stack |
| **Testing** | pytest, pytest-asyncio | Unit + integration tests |

</div>

---

## 📊 Evaluation Metrics

We evaluate against **NEWS2** — the clinical standard currently used in hospitals. Beating it is the core contribution.

| Metric | Description | Target |
|--------|------------|--------|
| **AUROC** | Primary discrimination metric | > NEWS2 baseline |
| **PR-AUC** | Handles severe class imbalance | > NEWS2 baseline |
| **Calibration** | Are confidence scores trustworthy? | ECE < 0.05 |
| **Alert Precision** | % of alerts that were clinically meaningful | > 70% |
| **Alert Fatigue Reduction** | Redundant alerts suppressed | > 40% reduction |

> Results will be updated here as experiments complete.

| Model | AUROC | PR-AUC | Notes |
|-------|-------|--------|-------|
| NEWS2 Baseline | `TBD` | `TBD` | Clinical standard |
| Logistic Regression | `TBD` | `TBD` | Simple ML baseline |
| LSTM v1 | `TBD` | `TBD` | First deep model |
| LSTM + Attention | `TBD` | `TBD` | Current best |

---

## 📂 Project Structure

```
icu-watch/
│
├── 📁 data/
│   ├── raw/                    # MIMIC-III raw extracts (gitignored)
│   ├── processed/              # Cleaned, feature-engineered data
│   └── samples/                # Anonymized samples for testing
│
├── 📓 notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb # NEWS2 replication
│   └── 04_deep_model.ipynb     # LSTM training
│
├── 📁 src/
│   ├── data/
│   │   ├── extract.py          # MIMIC-III SQL extraction
│   │   ├── preprocess.py       # Cleaning, resampling, labeling
│   │   └── feature_store.py    # Feature engineering
│   ├── models/
│   │   ├── baseline.py         # NEWS2 clinical score
│   │   ├── lstm_model.py       # Bidirectional LSTM + attention
│   │   └── tft_model.py        # Temporal Fusion Transformer (v2)
│   ├── explainability/
│   │   ├── shap_engine.py      # SHAP DeepExplainer wrapper
│   │   └── alert_logic.py      # Smart alert suppression
│   └── api/
│       ├── main.py             # FastAPI application
│       └── schemas.py          # Pydantic models
│
├── 📁 dashboard/               # React frontend
│   └── src/
│       ├── components/         # Reusable UI components
│       └── pages/              # Dashboard pages
│
├── 📁 research/
│   ├── papers/                 # Reference papers
│   ├── notes/                  # Experiment logs
│   └── experiments/            # MLflow artifacts
│
├── 📁 tests/                   # Unit + integration tests
├── 📁 docs/                    # Extended documentation
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── CONTRIBUTING.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python `3.10+`
- Node.js `18+`
- Docker + Docker Compose
- MIMIC-III access credentials

### 1. Clone the repository

```bash
git clone https://github.com/iamsubham019/icu-watch.git
cd icu-watch
```

### 2. Set up environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your database credentials
```

### 4. Start the full stack

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| React Dashboard | http://localhost:3000 |
| FastAPI Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow Tracking | http://localhost:5000 |

---

## 🗄️ MIMIC-III Data Access

This project uses the **MIMIC-III Clinical Database** — a freely available, de-identified health dataset of ICU patients.

**Steps to get access:**

1. Register at [physionet.org](https://physionet.org)
2. Complete the **CITI "Data or Specimens Only Research"** training (~2 hours)
3. Submit credentialing request — approval takes **3–7 days**
4. Download and place data under `data/raw/`

> ⚠️ **MIMIC-III data is never committed to this repository.** The `data/raw/` directory is in `.gitignore`. Both team members must obtain independent credentials.

---

## 🗓️ Development Roadmap

```
Week 1  ████████░░░░░░░░  Data extraction + EDA
Week 2  ░░░░████████░░░░  Feature engineering + API skeleton
Week 3  ░░░░░░░░████████  Baseline NEWS2 + Alert logic
Week 4  ░░░░░░░░░░░░████  LSTM training + Dashboard vitals
Week 5  ░░░░░░░░░░░░░░░█  SHAP engine + Visualization
Week 6  ░░░░░░░░░░░░░░░░  Full stack integration + Docker
Week 7  ░░░░░░░░░░░░░░░░  Research docs + Testing
Week 8  ░░░░░░░░░░░░░░░░  🚀 Full system demo
```

| Week | Subham Pal (ML) | Swarnali Ghosh (Backend/Frontend) |
|------|----------------|----------------------------------|
| 1 | MIMIC extraction + EDA notebook | Repo setup + FastAPI skeleton |
| 2 | Feature engineering pipeline | Patient schema + API endpoints |
| 3 | Baseline NEWS2 model | Alert intelligence layer |
| 4 | LSTM model trained + evaluated | Dashboard vitals display |
| 5 | SHAP engine complete | SHAP visualization on dashboard |
| 6 | Model → API integration | Docker + full stack running |
| 7 | Research notes + experiment log | README + docs complete |
| 8 | **Full system demo** | **Full system demo** |

---

## 👥 Team

<div align="center">

| | Name | Role | GitHub |
|-|------|------|--------|
| 🧠 | **Subham Pal** | ML Core · Feature Engineering · SHAP Explainability | [@iamsubham019](https://github.com/iamsubham019) |
| ⚙️ | **Swarnali Ghosh** | Backend · Alert Logic · React Dashboard | [@swarnali2005](https://github.com/swarnali2005) |

</div>

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching strategy, commit format, and file ownership.

**Branch structure:**
```
main              ← stable, always working
dev               ← integration branch
subham/ml-core    ← ML features
swarnali/backend  ← Backend + frontend features
```

---

## 📄 Research References

Key papers informing this work:

- Churpek et al. — *Predicting Clinical Deterioration in the Hospital*, NEJM 2016
- Tang et al. — *DeepSOFA: A Continuous Acuity Score for ICU Patients*
- Lim et al. — *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*
- Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions* (SHAP)

Full reading list in [`research/notes/research_log.md`](research/notes/research_log.md)

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## ⚠️ Disclaimer

ICU-Watch is a **research prototype** and is **not intended for clinical use**. All model outputs must be validated by qualified medical professionals before any real-world application. Patient data used in development is de-identified and used strictly in accordance with PhysioNet's data use agreement.

---

<div align="center">

*Built with purpose — because in critical care, every minute matters.*

⭐ Star this repo if you find it useful

</div>
