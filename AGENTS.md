# Repository Guidelines

## Project Structure & Module Organization
- `Clustering/` contains dataset-specific clustering scripts and notebooks, grouped by cohort (`ANAS`, `KoGES`, `SNUH`).
- Top-level notebooks (`Pre_training_code.ipynb`, `Main_training_code.ipynb`, `SHAP.ipynb`) capture the main training, transfer learning, and SHAP workflows.
- Data files are stored as CSVs at the repo root (for example `ANAS_cluster_data.csv`, `SHAP_total_SNUH.csv`).
- Model artifacts appear under `Model/` (for example `Pretrain/`, `Transfer_learning/`).

## Build, Test, and Development Commands
This project is script- and notebook-driven. There is no build system in the repo.
- Run a clustering script: `python Clustering/ANAS/PCA2_Kclustering.py`
- Run notebooks locally: `jupyter lab` or `jupyter notebook`
- Scripts read CSV inputs relative to their directory and write outputs under `Clustering/<cohort>/Output/...`.

## Coding Style & Naming Conventions
- Python scripts use 4-space indentation and a linear, notebook-exported style with minimal functions.
- Keep file naming consistent with the existing pattern: `<method>_<K|H>clustering.py` (for example `PCA3_Kclustering.py`, `UMAP_Hclustering.py`).
- Prefer descriptive variable names already used in scripts (for example `clustering_lst`, `cluster_range`).
- No formatter or linter is configured; match existing style unless you introduce a tool intentionally.

## Testing Guidelines
- There is no automated test suite in this repository.
- When adding new logic, validate via a notebook or by re-running the affected script and inspecting generated CSV outputs.

## Commit & Pull Request Guidelines
- This directory is not currently a Git repository, so commit conventions cannot be inferred.
- If you initialize Git, use short imperative subjects (for example `Add PCA3 KMeans clustering for KoGES`) and describe dataset or variant in the body.
- Pull requests should summarize the dataset, method, and expected output files; include before/after metrics or screenshots if plots change.

## Data & Output Handling
- Scripts read local CSVs; keep paths relative so they run from their own directory.
- Output files can be large. Avoid committing generated outputs unless required for reproducibility.
