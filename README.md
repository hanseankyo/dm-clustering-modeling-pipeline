# Clustering & Modeling Code (GitHub)

This repository contains **modeling (Pretrain/Train)**, **SHAP analysis**, and **clustering** code used in the manuscript.

## Project Layout
- `src/Modeling/` : Pretrain/Train/SHAP scripts
- `src/Clustering/` : cohort-specific clustering scripts
- `data/` : input data (`data/real_data/` is ignored; lightweight synthetic samples can be tracked)
- `outputs/` : generated outputs (CSV, etc.)
- `models/` : model checkpoints
- `scripts/` : helper shell scripts

## Quick Start
```bash
conda env create -f environment.yml
conda activate clustering_code_py311
```

Alternative (pip only):
```bash
pip install -r requirements.txt
```

## Modeling (Pretrain / Train)
### Inputs
- `data/Only_clinical.csv`
- `data/Train.csv`
- `data/Validation.csv`
- `data/Test.csv`

### Run
```bash
scripts/run_modeling.sh pretrain
scripts/run_modeling.sh train
```

### Outputs
- `models/Pretrain/`
- `models/Transfer_learning/`

**Note**: The latest pretrain checkpoint is selected automatically (`best_model_<num>.h5` with the largest `<num>`).

## SHAP
### Inputs
- `data/Train.csv`
- `data/Validation.csv`
- `data/Test.csv`
- `data/Only_clinical.csv`
- `data/sample_area_mapping.csv` (for ANAS merge; uses `dist_id`, `area`)
- Model checkpoints in `models/Pretrain/` and `models/Transfer_learning/`

### Run
```bash
scripts/run_modeling.sh shap
```

### Outputs
- KoGES
  - `outputs/SHAP_total_koges.csv`
  - `outputs/SHAP_total_koges_sample.csv`
- ANAS (conditional)
  - `outputs/SHAP_total_ANAS.csv`
- SNUH
  - `outputs/SHAP_total_SNUH.csv`
  - `outputs/SHAP_total_SNUH_sample.csv`

### Behavior
- If SHAP output files already exist, SHAP computation is **skipped**.
- If the ANAS merge file is missing, ANAS export is **skipped**.
- Plotting is removed; this script only generates SHAP values.

## Clustering
### Inputs (SHAP outputs)
- KoGES: `outputs/SHAP_total_koges.csv`
- ANAS: `outputs/SHAP_total_ANAS.csv`
- SNUH: `outputs/SHAP_total_SNUH.csv`

If the same files exist in `data/`, `data/` is used first.

### Run (all scripts in a cohort)
```bash
scripts/run_clustering_all.sh KoGES
scripts/run_clustering_all.sh ANAS
scripts/run_clustering_all.sh SNUH
```
Default mode is sequential (`CLUSTER_JOBS=1`). To enable parallel runs:
```bash
CLUSTER_JOBS=4 scripts/run_clustering_all.sh KoGES
```

### Run (single script)
```bash
scripts/run_clustering.sh KoGES PCA2_Kclustering.py
```

### Output Structure
```
outputs/Clustering/<COHORT>/<METHOD>/<ALGO>/
```
Example:
- `outputs/Clustering/KoGES/PCA2/Kmeans/`
- `outputs/Clustering/SNUH/UMAP/Hclust/`

## Notes
- `INPUTS.md` lists required input files.
- Large data/model files should not be committed to GitHub.

## Detailed Guides
- `MODELING.md`
- `SHAP.md`
- `CLUSTERING.md`
