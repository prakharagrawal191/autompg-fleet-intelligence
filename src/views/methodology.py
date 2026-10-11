"""Methodology - how each number is produced and where each assignment part lives."""
from importlib.metadata import version

import pandas as pd
import streamlit as st

from src import state as S
from src import ui_components as UI
from src.auto_mpg_utils import (
    CONTINUOUS_COLS, DEFAULT_N_INIT, DEFAULT_RANDOM_STATE, K_RANGE, optimal_k_from_silhouette,
)


def _qa(s):
    """Viva answers: method facts are fixed; anything about results is read from the session."""
    kr, pca = s.get("krange"), s.get("pca")
    if kr is not None:
        best_k, best_v = optimal_k_from_silhouette(kr)
        k_answer = f"In this session the K-Means sweep peaks at K={best_k} (silhouette {best_v:.4f}). "
    else:
        k_answer = "Compute the K=2–6 sweep in the Clustering Lab to see where silhouette peaks. "
    if S.is_project_data(s):
        k_answer += (f"For the project dataset, the Assessment 10 notebook documents K={S.DOCUMENTED_K} as a joint "
                     "engineering choice (elbow bend plus a more useful fleet segmentation), not as the metric "
                     "optimum.")
    else:
        k_answer += "The selected K is a choice to justify with the elbow, silhouette and the segmentation you need."
    pca_answer = ("PCA projects the five standardized features onto two components for plotting. Clustering always "
                  "runs in the full five-dimensional standardized space.")
    if pca is not None:
        pca_answer += f" Here PC1 + PC2 retain {100 * float(pca['variance_ratio'].sum()):.2f}% of the variance."
    return [
        ("Why cylinder-group median imputation?",
         "Engines with the same cylinder count share a comparable power band, so each missing horsepower takes the "
         "median of its cylinder group. A global median would pull small engines up and large engines down."),
        ("Why standardize before clustering?",
         "K-Means and Ward use Euclidean distance. Weight (thousands of lb) would otherwise outweigh MPG (tens) "
         "purely because of its units; z = (x − μ) / σ puts all five features on a comparable scale."),
        ("Why compare K=2–6?",
         "The assignment fixes that range. Inertia and silhouette are computed for every K so the choice is "
         "evidence-led rather than assumed."),
        ("Is the selected K the mathematical optimum?", "Not necessarily. " + k_answer),
        ("Why Ward hierarchical clustering?",
         "It is a second, agglomerative view of the same standardized data. Its labels come from "
         "AgglomerativeClustering; the dendrogram labels come from cutting the same linkage with "
         "fcluster(maxclust), and Cluster Quality reports how closely the two agree."),
        ("Why PCA only for visualization?", pca_answer),
        ("How are centroids and archetypes derived?",
         "K-Means centroids are mapped back to original units with scaler.inverse_transform. Archetype names come "
         "from ranking those centroids (weight and horsepower; MPG against weight), so they follow the data."),
    ]


def _requirements(s):
    any_m, km = S.has_model(s), S.has_km(s)
    kr, dend = s.get("krange") is not None, s.get("dendro") is not None
    dl = s.get("dlab") is not None and s.get("dk") == s.get("selected_k")
    rows = [  # (part, requirement as worded in the Assessment 10 notebook, page, done, available)
        ("A", "Missing horsepower detection", "Dataset · Preprocessing", S.has_raw(s), True),
        ("A", "Cylinder-group median imputation", "Preprocessing", S.has_clean(s), S.has_raw(s)),
        ("A", "Distribution analysis", "EDA Explorer", False, S.has_raw(s)),
        ("A", "Correlation analysis", "EDA Explorer", False, S.has_raw(s)),
        ("A", "Z-score standardization", "Preprocessing", S.has_std(s), S.has_clean(s)),
        ("B", "K-Means", "Clustering Lab", km, S.has_std(s)),
        ("B", "Ward hierarchical clustering", "Clustering Lab", S.has_ward(s), S.has_std(s)),
        ("B", "Elbow K=2-6", "Clustering Lab · Choosing K", kr, S.has_std(s)),
        ("B", "Silhouette K=2-6", "Clustering Lab · Choosing K", kr, S.has_std(s)),
        ("B", "K selection", "Clustering Lab · Choosing K", kr, S.has_std(s)),
        ("B", "Ward dendrogram", "Clustering Lab · Ward dendrogram", dend, S.has_std(s)),
        ("B", "Cluster assignment extraction", "Clustering Lab · Ward dendrogram", dl, dend),
        ("C", "Silhouette Score", "Cluster Quality", any_m, S.has_std(s)),
        ("C", "Davies-Bouldin Index", "Cluster Quality", any_m, S.has_std(s)),
        ("C", "PCA 2D projection", "PCA Explorer", s.get("pca") is not None, S.has_std(s)),
        ("C", "Cluster visualization", "Clustering Lab · PCA Explorer", any_m, S.has_std(s)),
        ("C", "Centroid interpretation", "Fleet Archetypes", km and s.get("cent_orig") is not None, S.has_std(s)),
        ("C", "Vehicle archetypes", "Fleet Archetypes", km and bool(s.get("arch")), S.has_std(s)),
        ("C", "Engineering inference", "Insights", False, any_m),
    ]
    return pd.DataFrame([{"Part": p, "Requirement": r, "Where": w,
                          "Status": "✓ Computed this session" if d else ("○ Available" if a else "— Waiting")}
                         for p, r, w, d, a in rows])


def render():
    s = st.session_state
    UI.page_header("Methodology", "How every number in this dashboard is produced, and where each assignment "
                   "requirement lives.", ("Automotive Fleet", "System", "Methodology"))
    UI.section("Pipeline", "Live state of each stage — the same stages as the Overview.")
    UI.pipeline(S.stages(s))
    rs, ni = S.km_params(s)
    km_cfg = f"random_state={rs}, n_init={ni}" + (
        "" if (rs, ni) == (DEFAULT_RANDOM_STATE, DEFAULT_N_INIT) else " — non-default; validated: 42 / 10")
    UI.callout("<b>Data</b> → clean ('?' → missing) → impute horsepower (cylinder-group median) → standardize "
               f"(z-score, 5 features) → cluster (K-Means {UI.esc(km_cfg)}; Ward linkage; K=2–6) → evaluate "
               "(silhouette, Davies–Bouldin, elbow) → PCA (2-D, visualization only) → archetypes (inverse-transformed "
               "K-Means centroids, data-driven names).", title="What the app does", tone="teal", html=True)
    UI.section("Assignment traceability", "Requirements as worded in the Assessment 10 notebook. 'Computed' means "
               "the result exists in this session — a page existing is not enough.")
    req = _requirements(s)
    st.dataframe(req, hide_index=True, width="stretch", height=38 + 35 * len(req))
    st.caption("Distribution, correlation and engineering-inference views are computed on demand when their page "
               "opens, so they show as Available rather than Computed.")
    UI.section("Viva questions", "Concise answers describing the actual implementation.")
    for q, a in _qa(s):
        with st.expander(q):
            st.write(a)
    UI.section("Reproducibility")
    UI.keyvals([("Features", ", ".join(CONTINUOUS_COLS)),
                ("K-Means", f"random_state={rs}, n_init={ni}"
                 + ("" if (rs, ni) == (DEFAULT_RANDOM_STATE, DEFAULT_N_INIT) else " (non-default)")),
                ("Candidate K", f"{min(K_RANGE)}–{max(K_RANGE)}"), ("Ward", "linkage='ward', fcluster(maxclust)"),
                ("Dataset fingerprint", s.get("dataset_id") or "no dataset loaded"),
                ("Libraries", " · ".join(f"{p} {version(p)}" for p in
                                         ("scikit-learn", "scipy", "pandas", "numpy", "streamlit", "plotly")))])
