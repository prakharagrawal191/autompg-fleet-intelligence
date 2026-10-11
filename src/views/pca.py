"""PCA Explorer - from five standardized features to a two-dimensional view."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import chart_factory as CF
from src import state as S
from src import theme as T
from src import ui_components as UI
from src.auto_mpg_utils import CONTINUOUS_COLS


def _loadings(p):
    comp = pd.DataFrame(p["model"].components_[:2], index=["PC1", "PC2"], columns=CONTINUOUS_COLS)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        CF.show(CF.diverging_heatmap(comp, "Loadings", "Loading of each standardized feature on PC1 and PC2",
                                     fmt=".2f", colorbar="loading", height=260),
                alt="Heatmap of PC1 and PC2 loadings for the five standardized features.")
    with right:
        lines = []
        for pc in comp.index:
            top = comp.loc[pc].abs().sort_values(ascending=False).index[:3]
            lines.append(f"<b>{pc}</b>: " + ", ".join(f"{f} ({comp.loc[pc, f]:+.2f})" for f in top))
        UI.callout("<br>".join(lines) + "<br><span class='ui-next'>Largest absolute loadings first. A component's sign "
                   "is arbitrary; features with the same sign move together along it.</span>",
                   title="What the axes mean", tone="teal", html=True)


def render():
    s = st.session_state
    UI.page_header("PCA Explorer", "From five-dimensional feature space to a two-dimensional view.",
                   ("Automotive Fleet", "Modeling", "PCA Explorer"))
    if not S.has_std(s):
        UI.empty_state("PREPROCESSING INCOMPLETE", "PCA projects the standardized five-feature matrix.",
                       "Preprocessing → Apply imputation → Standardize")
        st.button("Open Preprocessing", on_click=S.goto, args=("Preprocessing",))
        return
    p = S.ensure_pca(s)
    v = p["variance_ratio"]
    total = float(np.sum(v))
    UI.kpis([{"label": "PC1", "value": f"{100 * v[0]:.2f}%", "sub": "explained variance"},
             {"label": "PC2", "value": f"{100 * v[1]:.2f}%", "sub": "explained variance"},
             {"label": "Combined", "value": f"{100 * total:.2f}%", "sub": "retained in this 2-D view",
              "tone": "gold"}])
    UI.bars([("PC1", v[0], f"{100 * v[0]:.2f}%", T.TEAL), ("PC2", v[1], f"{100 * v[1]:.2f}%", T.TEAL),
             ("PC3–PC5 (not shown)", 1 - total, f"{100 * (1 - total):.2f}%", T.BORDER_STRONG)],
            label="Share of standardized variance by principal component")

    fitted = S.fitted_methods(s)
    frame = s["df_clean"].assign(PC1=p["coords"][:, 0], PC2=p["coords"][:, 1])
    xt, yt = f"PC1 ({100 * v[0]:.1f}% of variance)", f"PC2 ({100 * v[1]:.1f}% of variance)"
    UI.section("Projection", "Each point is a vehicle placed by its first two principal-component scores.")
    if fitted:
        if s.get("pca_color") not in fitted:
            s["pca_color"] = fitted[0]
        a, b, c, d = st.columns([1.3, 1, 1.2, 1.4], vertical_alignment="bottom")
        method = a.radio("Color by", fitted, horizontal=True, key="pca_color", format_func=S.METHOD_TITLES.get)
        show_c = b.checkbox("Show centroid markers", key="pca_centroids",
                            help="Centers projected with the fitted PCA: K-Means centroids or Ward cluster means.")
        show_l = c.checkbox(f"Show vehicle labels ({len(frame)})", key="pca_labels")
        size = d.slider("Point size", 3, 14, key="pca_size")
        r = S.model(method, s)
        centers = p["model"].transform(S.centers_std(method, s)) if show_c else None
        fig = CF.cluster_scatter(frame, "PC1", "PC2", r["labels"], r["k"], xtitle=xt, ytitle=yt,
                                 title=f"{S.METHOD_TITLES[method]} clusters in 2-D PCA space", subtitle=f"K={r['k']}",
                                 centers=centers, center_label="Centroid" if method == S.KMEANS else "Mean",
                                 size=size, show_text=show_l, height=480)
        alt = f"PCA scatter of {len(frame)} vehicles colored by {S.METHOD_TITLES[method]} cluster (K={r['k']})."
    else:
        fig = go.Figure(go.Scatter(x=frame["PC1"], y=frame["PC2"], mode="markers", name="Vehicles",
                                   hovertext=CF.plain(frame["car_name"]),
                                   hovertemplate="<b>%{hovertext}</b><br>PC1 %{x:.2f} · PC2 %{y:.2f}<extra></extra>",
                                   marker=dict(color=T.TEAL, size=7, opacity=0.8, line=dict(width=0.6,
                                                                                         color=T.BACKGROUND))))
        fig = CF.style(fig, 480, "Vehicles in 2-D PCA space", "No clustering fitted yet", xt, yt, legend=False)
        alt = f"PCA scatter of {len(frame)} vehicles without cluster coloring."
    pad = lambda lo, hi: [lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)]  # noqa: E731
    fig.update_xaxes(range=pad(float(frame["PC1"].min()), float(frame["PC1"].max())))
    fig.update_yaxes(range=pad(float(frame["PC2"].min()), float(frame["PC2"].max())))
    CF.show(fig, alt=alt)
    if not fitted:
        UI.callout("Fit K-Means or Ward to color the projection by cluster.", tone="neutral")
        st.button("Open Clustering Lab", on_click=S.goto, args=("Clustering Lab",))
    st.caption("Axis ranges come from the full projection, so switching methods never rescales the view.")
    _loadings(p)
    UI.takeaway(f"The first two principal components explain {100 * total:.2f}% of standardized variance. PCA is "
                "used here for visualization; clustering remains performed in the five-dimensional standardized "
                "feature space, so clusters that overlap in 2-D may still be separated in 5-D.")
    UI.viva("PCA is fitted on the same standardized matrix the clustering uses. Centroids are not placed by eye: "
            "each center's 5-D standardized vector is transformed with the fitted PCA, the same map applied to "
            "every vehicle.")
