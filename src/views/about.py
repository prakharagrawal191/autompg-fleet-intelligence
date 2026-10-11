"""About - project, author and scientific notes."""
import streamlit as st

from src import ui_components as UI


def render():
    UI.page_header("About", "Project, author and the scientific conventions behind every page.",
                   ("Automotive Fleet", "System", "About"))
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        UI.keyvals([("Project", "Automotive Fleet Profiling & Efficiency Clustering"),
                    ("Student", "Prakhar Agrawal"), ("Register number", "26MML0003"),
                    ("Course", "Machine Learning Lab (MACSE504)"), ("Assessment", "10 · Question 4"),
                    ("Dataset", "UCI Auto MPG (auto-mpg.data)"),
                    ("Stack", "Python · pandas · scikit-learn · SciPy · Plotly · Streamlit")])
    with b:
        UI.callout("Cylinder-group median imputation of horsepower; z-score standardization of mpg, displacement, "
                   "horsepower, weight and acceleration; K-Means (validated configuration random_state=42, n_init=10) "
                   "and Ward hierarchical "
                   "clustering over K=2–6; elbow and silhouette analysis; silhouette and Davies–Bouldin evaluation; "
                   "PCA for 2-D visualization; inverse-transformed centroids and data-driven archetypes.",
                   title="Methodology", tone="teal")
        UI.callout("PCA never performs clustering. Silhouette: higher is better; Davies–Bouldin: lower is better; "
                   "inertia is read through the elbow, never minimized. The dendrogram is drawn from the full Ward "
                   "linkage with display truncation only; assignments are extracted from that same linkage with "
                   "fcluster(maxclust).", title="Scientific notes", tone="neutral")
    st.caption("Dataset source: UCI Machine Learning Repository — https://archive.ics.uci.edu/ml/datasets/auto+mpg")
