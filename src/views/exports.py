"""Exports - download the current results; every file reflects the active state."""
import pandas as pd
import streamlit as st

from src import state as S
from src import ui_components as UI


def _card(title, desc, label, frame, fname, missing):
    with st.container(border=True):
        UI.md(f"<div class='ui-step'><span class='ui-step-title'>{UI.esc(title)}</span>"
              f"<span class='ui-section-sub'>{UI.esc(fname)}</span></div>")
        if frame is None:
            st.caption(f"Not available — {missing}")
            return
        st.caption(f"{desc} · {len(frame):,} rows × {frame.shape[1]} columns")
        st.download_button(label, frame.to_csv(index=False), file_name=fname, mime="text/csv", width="stretch",
                           icon=":material/download:", on_click="ignore")


def render():
    s = st.session_state
    UI.page_header("Exports", "Download the current results; every file reflects the active dataset, method and K.",
                   ("Automotive Fleet", "System", "Exports"))
    if not S.has_raw(s):
        UI.empty_state("DATASET NOT READY", "Load or upload an Auto MPG dataset to create exports.",
                       "Dataset → Use Project Dataset")
        st.button("Open Dataset", on_click=S.goto, args=("Dataset",))
        return
    km, w = S.model(S.KMEANS, s), S.model(S.WARD, s)
    k = s["selected_k"]
    src = "project dataset" if S.is_project_data(s) else s["dataset_name"]
    UI.provenance(f"{src} · fingerprint {s['dataset_id']} · K={k}", "Exporting")
    st.caption("Files are generated from session state when you click; the source dataset is never modified.")
    clean = s["df_clean"]
    dend_ok = s.get("dlab") is not None and s.get("dk") == k

    def with_labels(**cols):
        if clean is None:
            return None
        d = clean.copy()
        for name, labels in cols.items():
            d[name] = labels
        return d

    UI.section("Data")
    a, b = st.columns(2)
    with a:
        _card("Cleaned dataset", "Horsepower imputed (cylinder-group median)", "Download Cleaned Dataset", clean,
              "auto_mpg_cleaned.csv", "apply imputation on Preprocessing.")
    with b:
        _card("Standardized dataset", "Five clustering features as z-scores", "Download Standardized Dataset",
              s["df_std"], "auto_mpg_standardized.csv", "standardize features on Preprocessing.")
    UI.section("Clustering")
    a, b, c = st.columns(3)
    with a:
        _card("K-Means results", f"Cleaned data + kmeans_cluster (K={k})", "Download K-Means Results",
              with_labels(kmeans_cluster=km["labels"]) if km else None, "auto_mpg_kmeans_results.csv",
              "fit K-Means in the Clustering Lab.")
    with b:
        _card("Ward results", f"Cleaned data + ward_cluster (K={k})", "Download Ward Results",
              with_labels(ward_cluster=w["labels"]) if w else None, "auto_mpg_ward_results.csv",
              "fit Ward in the Clustering Lab.")
    with c:
        dend = clean[["car_name"]].assign(dendrogram_cluster=s["dlab"]) if dend_ok else None
        _card("Dendrogram assignments", f"fcluster(maxclust) labels at K={k}", "Download Dendrogram Assignments",
              dend, "auto_mpg_dendrogram_clusters.csv", "build the dendrogram and extract assignments.")
    UI.section("Evaluation")
    a, b = st.columns(2)
    with a:
        _card("Centroids", "K-Means centroids in original units (inverse-transformed)", "Download Centroids",
              s["cent_orig"] if km else None, "auto_mpg_centroids.csv", "fit K-Means in the Clustering Lab.")
    with b:
        rows = [(name, r["k"], round(r["silhouette"], 4), round(r["db"], 4))
                for name, r in (("K-Means", km), ("Ward", w)) if r is not None]
        ev = pd.DataFrame(rows, columns=["method", "k", "silhouette", "davies_bouldin"]) if rows else None
        _card("Evaluation metrics", "Silhouette and Davies–Bouldin per fitted method", "Download Evaluation Metrics",
              ev, "auto_mpg_evaluation.csv", "fit a model in the Clustering Lab.")
    UI.section("Complete")
    full = None
    if km or w:
        cols = {}
        if km:
            cols["kmeans_cluster"] = km["labels"]
        if w:
            cols["ward_cluster"] = w["labels"]
        if dend_ok:
            cols["dendrogram_cluster"] = s["dlab"]
        full = with_labels(**cols)
    _card("Complete clustered dataset", "Cleaned data + every available cluster label",
          "Download Complete Clustered Dataset", full, "auto_mpg_complete_clustered.csv",
          "fit a model in the Clustering Lab.")
