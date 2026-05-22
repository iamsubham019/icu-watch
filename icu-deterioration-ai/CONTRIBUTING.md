# Contributing Guide

## Team
- **Subham Pal** — ML Core, Feature Engineering, SHAP Explainability
- **Swarnali Ghosh** — Backend (FastAPI), Alert Logic, React Dashboard

---

## Branching Strategy

```
main              ← stable, always working — never push directly
dev               ← integration branch — merge here first
subham/ml-core    ← Subham's primary branch
swarnali/backend  ← Swarnali's primary branch
```

**Rule:** Always open a PR into `dev`. Never push directly to `main`.

---

## Commit Message Format

```
[scope] short description

Examples:
[data] add MIMIC vitals extraction query
[model] implement bidirectional LSTM with attention
[api] add /predict endpoint with schema validation
[dashboard] add patient vitals chart component
[shap] implement waterfall plot generation
[docs] update research log with experiment results
```

---

## File Ownership

| Path | Owner |
|------|-------|
| `src/data/` | Subham |
| `src/models/` | Subham |
| `src/explainability/shap_engine.py` | Subham |
| `src/explainability/alert_logic.py` | Swarnali |
| `src/api/` | Swarnali |
| `dashboard/` | Swarnali |
| `notebooks/` | Both |
| `research/` | Both |
| `tests/` | Both |

---

## Weekly Sync Checklist
- [ ] Both branches merged to `dev`
- [ ] No broken imports
- [ ] New features have at least one test
- [ ] Research log updated with experiment results
