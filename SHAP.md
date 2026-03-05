# SHAP Usage

This document covers SHAP value generation for KoGES, ANAS, and SNUH.

## Entry Points
- Script: `src/Modeling/main.py` (task: `shap`)
- Helper: `scripts/run_modeling.sh`

## Inputs
Place these under `data/`:
- `Train.csv`
- `Validation.csv`
- `Test.csv`
- `Only_clinical.csv`
- `sample_area_mapping.csv` (used to derive ANAS output; columns: `dist_id`, `area`)

Model checkpoints required:
- `models/Pretrain/best_model_<epoch>.h5`
- `models/Transfer_learning/best_model_elu_*.h5`

## Run
```bash
scripts/run_modeling.sh shap
```

## Outputs
KoGES SHAP (direct):
- `outputs/SHAP_total_koges.csv`
- `outputs/SHAP_total_koges_sample.csv`

ANAS SHAP (derived from KoGES + merge file):
- `outputs/SHAP_total_ANAS.csv`

SNUH SHAP (direct):
- `outputs/SHAP_total_SNUH.csv`
- `outputs/SHAP_total_SNUH_sample.csv`

## Behavior
- If KoGES or SNUH SHAP outputs already exist, their computation is skipped.
- If the ANAS merge file is missing, ANAS output is skipped.
- Plotting is removed; only SHAP values are generated.
