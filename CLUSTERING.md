# Clustering Usage

This document covers cohort-specific clustering runs.

## Entry Points
- Scripts: `src/Clustering/<COHORT>/*.py`
- Helpers: `scripts/run_clustering.sh`, `scripts/run_clustering_all.sh`

## Inputs
Clustering consumes SHAP outputs (or the same files placed in `data/`).

Expected SHAP inputs by cohort:
- KoGES: `SHAP_total_koges.csv`
- ANAS: `SHAP_total_ANAS.csv`
- SNUH: `SHAP_total_SNUH.csv`

Resolution order:
- If the file exists in `data/`, it is used.
- Otherwise, the file is loaded from `outputs/`.

## Run
### Run all scripts in a cohort
```bash
scripts/run_clustering_all.sh KoGES
scripts/run_clustering_all.sh ANAS
scripts/run_clustering_all.sh SNUH
```
Default mode is sequential (`CLUSTER_JOBS=1`). To run in parallel:
```bash
CLUSTER_JOBS=4 scripts/run_clustering_all.sh KoGES
```

### Run a single script
```bash
scripts/run_clustering.sh KoGES PCA2_Kclustering.py
```

## Outputs
Results are written to:
```
outputs/Clustering/<COHORT>/<METHOD>/<ALGO>/
```
Examples:
- `outputs/Clustering/KoGES/PCA2/Kmeans/`
- `outputs/Clustering/SNUH/UMAP/Hclust/`

## Notes
- Parallel runs can be CPU/memory intensive.
- Each script writes its own CSV outputs under the cohort/method/algo directory.
