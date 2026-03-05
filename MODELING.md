# Modeling Usage

This document covers **Pretrain** and **Train/Transfer Learning**.

## Entry Points
- Script: `src/Modeling/main.py`
- Helper: `scripts/run_modeling.sh`

## Inputs
Place these under `data/`:
- `Only_clinical.csv`
- `Train.csv`
- `Validation.csv`
- `Test.csv`

## Run
```bash
scripts/run_modeling.sh pretrain
scripts/run_modeling.sh train
```

## Outputs
- Pretrain checkpoints:
  - `models/Pretrain/best_model_<epoch>.h5`
- Transfer learning checkpoints and metrics:
  - `models/Transfer_learning/best_model_elu_*.h5`
  - `models/Transfer_learning/validation_performance.csv`
  - `models/Transfer_learning/test_performance.csv`

## Notes
- The latest pretrain checkpoint is selected automatically (largest `<epoch>`).
- Inputs are expected to be clean CSVs with the columns used in the original notebooks.
