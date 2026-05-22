# Research Log — ICU Patient Deterioration Prediction

## Key Papers to Read

### Foundational
- [ ] "A Comparison of Early Warning Scores for the Early Identification of Sepsis" — Churpek et al.
- [ ] "Multivariable logistic regression analysis for predicting clinical deterioration" — Escobar et al.
- [ ] "Deep Learning for Real-Time Atrial Fibrillation Detection" — Hannun et al. (architecture reference)

### Directly Relevant
- [ ] "Predicting Clinical Deterioration in the Hospital" — Churpek et al., NEJM 2016
- [ ] "DeepSOFA: A Continuous Acuity Score for ICU Patients" — Tang et al.
- [ ] "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" — Lim et al.

### Explainability
- [ ] "A Unified Approach to Interpreting Model Predictions" — Lundberg & Lee (SHAP paper)
- [ ] "Explainability in Clinical Decision Support" — Tonekaboni et al.

---

## Dataset Notes

### MIMIC-III
- 53,423 distinct hospital admissions
- 38,597 adult patients
- Access: PhysioNet credentialing required
- Key tables: `chartevents`, `labevents`, `icustays`, `admissions`, `patients`

### PhysioNet Challenge 2019
- Publicly available without credentialing
- Specifically designed for sepsis prediction
- Good for initial prototyping before MIMIC access is approved

---

## Experiment Log

| Date | Experiment | AUROC | Notes |
|------|-----------|-------|-------|
| - | NEWS2 Baseline | TBD | Clinical standard to beat |
| - | Logistic Regression | TBD | Simple ML baseline |
| - | LSTM v1 | TBD | First deep model |
| - | LSTM + Attention | TBD | With attention mechanism |

---

## Architecture Decisions

### Why LSTM over Transformer?
- ICU data sequences are relatively short (hours, not thousands of tokens)
- LSTM has lower computational cost — fits RTX 4050
- Bidirectional LSTM with attention is well-validated in clinical literature
- TFT (Temporal Fusion Transformer) as stretch goal

### Why NEWS2 as baseline?
- It's the actual clinical standard used in hospitals
- Beating it is a meaningful, real-world contribution
- It establishes that our AI approach adds genuine value

---

## Open Questions
- [ ] How to handle missing vitals (imputation strategy)?
- [ ] What prediction horizon gives the most clinical value — 4h, 6h, 12h?
- [ ] How to calibrate confidence scores for clinical trust?
