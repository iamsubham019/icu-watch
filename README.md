<div align="center">

<img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge" />
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

[🌐 Live Dashboard](https://icu-watch.vercel.app) • [⚙️ API Docs](https://icu-watch.onrender.com/docs) • [📖 Documentation](#-getting-started) • [🏗️ Architecture](#-system-architecture)

</div>

---

## 🌐 Live Demo

| Service | URL |
|---------|-----|
| 🖥️ **Dashboard** | https://icu-watch.vercel.app |
| ⚙️ **API Docs** | https://icu-watch.onrender.com/docs |
| 📦 **GitHub** | https://github.com/iamsubham019/icu-watch |

> ⚠️ The free backend may take 30–50 seconds to wake up on first visit. Click **🔄 Refresh** on the dashboard after it loads.

---

## 📌 The Problem

> **Hospitals already have monitoring systems. They fail.**

Every ICU nurse manages 2–4 critically ill patients simultaneously. Existing systems generate **hundreds of meaningless alerts per shift** — and nurses, overwhelmed, begin to tune them out. This is called **alarm fatigue**, and it kills people.

The real problem isn't *detection*. It's **intelligent, calibrated, trustworthy alerting.**

When a patient's condition starts to deteriorate, the window to intervene is narrow. Sepsis caught 6 hours early has a dramatically higher survival rate than sepsis caught at crisis point. Current rule-based systems miss the subtle, multivariate patterns that precede deterioration.

**ICU-Watch bridges that gap.**

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
│                  PhysioNet Challenge 2019 Dataset               │
│            40,336 ICU patients · 1.1M training sequences        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Pipeline                              │
│   extract.py → preprocess.py → feature_store.py                │
│   • Outlier removal    • Hourly resampling                      │
│   • 104 engineered features   • 6h ahead labels                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML Models                                 │
│                                                                 │
│   [NEWS2 Baseline] ──→ [Logistic Regression] ──→ [LSTM]        │
│                                                     │           │
│   Bidirectional LSTM + Temporal Attention           │           │
│   Input: (batch, 12h sequence, 104 features)        │           │
│   Output: P(deterioration within 6h)                │           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Explainability Engine                         │
│   Integrated Gradients → per-patient, per-vital importance     │
│   "Resp instability + BUN + Hct drove this Critical alert"     │
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
│   Alert engine           │   SHAP risk driver visualization     │
│   Model integration      │   Patient detail modal               │
│   Render deployment      │   Vercel deployment                  │
└──────────────────────────┴──────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data** | PhysioNet Challenge 2019 | 40,336 ICU patients |
| **ML Core** | PyTorch 2.3, scikit-learn | BiLSTM + Attention model |
| **Explainability** | Integrated Gradients | Per-prediction vital importance |
| **Experiment Tracking** | MLflow | Model versioning, metric logging |
| **Backend** | FastAPI, Pydantic | REST API, alert engine |
| **Frontend** | React 18, TailwindCSS | Clinical dashboard |
| **Deployment** | Render + Vercel | Free tier, globally accessible |

</div>

---

## 📊 Model Results

Our BiLSTM beats every baseline including the clinical standard used in hospitals:

| Model | AUROC | PR-AUC | Notes |
|-------|-------|--------|-------|
| NEWS2 (Clinical Standard) | `0.6095` | `0.0281` | What hospitals currently use |
| Logistic Regression | `0.7179` | `0.0732` | Simple ML baseline |
| Random Forest | `0.7402` | `0.0772` | Best non-temporal model |
| **BiLSTM + Attention** | **`0.7471`** | **`0.0803`** | ✅ **Our model — beats all** |

**+13.76 AUROC points over the clinical standard (NEWS2)**

### Top SHAP Drivers (Global)
1. **BUN** (Blood Urea Nitrogen) — kidney stress, early sepsis signal
2. **Hct** (Hematocrit) — fluid shifts, septic shock indicator
3. **Resp_std4h** — respiratory instability over 4h — the temporal signal rule-based systems miss

---

## 📂 Project Structure

```
icu-watch/
│
├── 📓 notebooks/
│   ├── 01_eda.ipynb                  # 40,336 patients analyzed
│   ├── 02_feature_engineering.ipynb  # 104 features, 1.1M sequences
│   ├── 03_baseline_model.ipynb       # NEWS2, LR, RF baselines
│   ├── 04_deep_model.ipynb           # BiLSTM training + evaluation
│   └── 05_shap_explainability.ipynb  # SHAP analysis
│
├── 📁 src/
│   ├── data/
│   │   ├── extract.py                # Data extraction
│   │   ├── preprocess.py             # Cleaning, resampling, labeling
│   │   └── feature_store.py          # Feature engineering
│   ├── models/
│   │   ├── baseline.py               # NEWS2 clinical score
│   │   └── lstm_model.py             # BiLSTM + Attention
│   ├── explainability/
│   │   ├── shap_engine.py            # SHAP explanations
│   │   └── alert_logic.py            # Smart alert suppression
│   └── api/
│       ├── main.py                   # FastAPI application
│       ├── model_loader.py           # LSTM model integration
│       └── schemas.py                # Pydantic models
│
├── 📁 dashboard/                     # React frontend
│   └── src/
│       └── App.js                    # Full dashboard application
│
├── 📁 research/
│   └── notes/                        # EDA plots, model charts, SHAP visualizations
│
├── render.yaml                       # Render deployment config
├── docker-compose.yml                # Local full stack
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python `3.10+`
- Node.js `18+`
- Docker + Docker Compose (optional)

### 1. Clone the repository

```bash
git clone https://github.com/iamsubham019/icu-watch.git
cd icu-watch
```

### 2. Set up Python environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 3. Run the backend

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

### 4. Run the dashboard

```bash
cd dashboard
npm install
npm start
```

| Service | Local URL |
|---------|-----------|
| React Dashboard | http://localhost:3000 |
| FastAPI Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## 🗓️ Development Roadmap

| Week | Subham Pal (ML) | Swarnali Ghosh (Backend/Frontend) |
|------|----------------|----------------------------------|
| 1 | EDA — 40,336 patients | Repo setup + FastAPI skeleton |
| 2 | Feature engineering — 104 features | Patient schema + API endpoints |
| 3 | Baseline models (NEWS2, LR, RF) | Alert intelligence layer |
| 4 | BiLSTM trained — AUROC 0.7471 ✅ | Dashboard vitals display |
| 5 | SHAP explainability complete | SHAP visualization on dashboard |
| 6 | Real model integration | Docker + full stack running |
| 7 | Research docs complete | README + deployment |
| 8 | **🚀 Deployed Live** | **🚀 Deployed Live** |

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

---

## 📄 Research References

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

ICU-Watch is a **research prototype** and is **not intended for clinical use**. All model outputs must be validated by qualified medical professionals before any real-world application.

---

<div align="center">

*Built with purpose — because in critical care, every minute matters.*

⭐ Star this repo if you find it useful

**[🌐 View Live Demo](https://icu-watch.vercel.app)**

</div>
