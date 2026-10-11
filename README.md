# Automotive Fleet Profiling & Efficiency Clustering (Auto MPG)

M.Tech Machine Learning Lab (MACSE504) — Question 4 dashboard by
**Prakhar Agrawal (26MML0003)**. UCI Auto MPG dataset.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

No code edits are needed: load the dataset from the **Dataset** page
(upload `.csv` / `.data` / `.txt`, or use the project `auto-mpg.data`).

## Deploy (Streamlit Community Cloud)

1. Fork / use this repo (`main` branch, `app.py` at the root — already set up).
2. Open [share.streamlit.io](https://share.streamlit.io) → **Deploy an app**
   → **From existing repo** → select this repo, branch `main`, main file `app.py`.
3. Click **Deploy**. The app builds from `requirements.txt` (Python 3.12 pinned
   via `.python-version`) with no manual steps.

The deployed instance works out of the box: **Dataset → Use Project Dataset**
loads the bundled `auto-mpg.data`. **Demo Mode** (sidebar) adds a guided
nine-step walkthrough; it never runs or changes any computation.

## Features

- Dataset import (UCI whitespace `.data` with `?` horsepower, or CSV) with
  structured validation: what failed, why it matters, and the next step
- Preprocessing: cylinder-group median imputation (affected rows shown),
  Z-score standardization with before/after statistics
- EDA workspace: distributions with live statistics, correlation heatmap,
  relationship explorer with an optional least-squares trendline
- Clustering Lab: K-Means (validated random_state=42, n_init=10; adjustable for
  sensitivity checks, flagged when non-default) and Ward hierarchical, K = 2–6,
  elbow + silhouette with a computed "why this K" reading, full-linkage Ward
  dendrogram whose cut line matches the `fcluster(maxclust)` assignments
- Cluster Quality: silhouette (higher better) / Davies–Bouldin (lower better),
  computed leaders, no composite score
- PCA explorer (visualization only) with projected centroids and loadings
- Fleet archetypes from inverse-transformed centroids, representative vehicles,
  vehicle lookup with standardized distances to every centroid
- Engineering insights, methodology traceability, 8 CSV exports from live state
- Presentation Mode (projector layout) and optional viva explanations

## Project structure

| Path | Role |
| ---- | ---- |
| `app.py` | Shell: navigation, status bar, modes, page routing |
| `src/auto_mpg_utils.py` | Validated science (unchanged by the UI) |
| `src/state.py` | Session state, dataset validation, central invalidation, pipeline actions |
| `src/views/*.py` | One module per page |
| `src/ui_components.py`, `src/chart_factory.py`, `src/theme.py` | Design system: components, Plotly theme, tokens and the validated cluster palette |

## Tests

```bash
python run_tests.py
```

Runs the science audit (reference values), the UI walkthrough, spot checks,
interaction scenarios and the redesign regression suite (state invalidation,
validation, dendrogram/PCA geometry, every page in every state).

## Assignment Mapping

| Interface Feature  | Assignment Requirement |
| ------------------ | ---------------------- |
| Dataset uploader   | Interface requirement  |
| Preprocessing page | Part A                 |
| EDA                | Part A                 |
| K selector         | Part B                 |
| K-Means            | Part B                 |
| Ward               | Part B                 |
| Elbow              | Part B                 |
| Silhouette         | Part B/C               |
| Dendrogram         | Part B                 |
| Davies-Bouldin     | Part C                 |
| PCA                | Part C                 |
| Archetypes         | Part C                 |

## Methodology

Cylinder-group median imputation → Z-score standardization of
`mpg, displacement, horsepower, weight, acceleration` → K-Means
(`random_state=42`, `n_init=10`) / Ward linkage on the standardized
matrix → Elbow + Silhouette over K=2..6 (engineering selection K=3,
silhouette optimum K=2) → PCA visualization → centroid archetypes.
Shared logic lives in `src/auto_mpg_utils.py` and mirrors the validated
Assessment notebooks (which are not modified).

Dataset source: **UCI Auto MPG Dataset**
(https://archive.ics.uci.edu/ml/datasets/auto+mpg).
