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

## Features

- Dataset upload (UCI raw `?` horsepower supported) + inspection
- Preprocessing: cylinder-group median imputation, Z-score standardization
- EDA: distributions, correlation heatmap, pairwise explorer
- K selection: Elbow + Silhouette for K = 2..6 (animated)
- Clustering: K-Means (random_state / n_init adjustable) + Ward hierarchical
- Ward dendrogram with cut indicator + `fcluster` assignments
- Evaluation: Silhouette (higher better) / Davies-Bouldin (lower better)
- PCA 2D explorer (visualization only), animated method transitions
- Fleet archetypes: inverse-transformed centroids, data-driven names,
  representative vehicles, vehicle lookup, engineering inference
- Results & Export: 8 CSV downloads generated from live state

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
