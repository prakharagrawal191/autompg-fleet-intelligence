"""Cluster Quality - compare K-Means and Ward on the same data and K."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import adjusted_rand_score

from src import chart_factory as CF
from src import state as S
from src import theme as T
from src import ui_components as UI

METHOD_COLOR = {S.KMEANS: T.GOLD, S.WARD: T.TEAL}
# Radio values stay "Davies-Bouldin" (stored state); the label shows the en dash used everywhere else.
METRICS = {"Silhouette": ("silhouette", "higher is better"), "Davies-Bouldin": ("db", "lower is better")}
leader = S.metric_leader


def _fmt(r, key, spec=".4f"):
    return "Not fitted" if r is None else format(r[key], spec)


def _sizes(r):
    return "Not fitted" if r is None else " / ".join(str(int(v)) for v in r["counts"])


def _table(km, w):
    rows = []
    for name, key, higher in (("Silhouette (higher is better)", "silhouette", True),
                              ("Davies–Bouldin (lower is better)", "db", False)):
        better, gap = "—", "—"
        if km is not None and w is not None:
            lead = leader(km, w, key, higher)
            better = S.METHOD_TITLES[lead] if lead else "Tie"
            gap = f"{abs(km[key] - w[key]):.4f}"
        rows.append({"Metric": name, "K-Means": _fmt(km, key), "Ward hierarchical": _fmt(w, key),
                     "Better": better, "|Δ|": gap})
    rows.append({"Metric": "Inertia (K-Means only)", "K-Means": _fmt(km, "inertia", ".2f"),
                 "Ward hierarchical": "not defined", "Better": "—", "|Δ|": "—"})
    rows.append({"Metric": "Cluster sizes (by id)", "K-Means": _sizes(km), "Ward hierarchical": _sizes(w),
                 "Better": "—", "|Δ|": "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _verdict(km, w):
    sl, dl = leader(km, w, "silhouette", True), leader(km, w, "db", False)
    name = lambda m: f"<b>{S.METHOD_TITLES[m]}</b>" if m else "<b>tie</b>"  # noqa: E731
    lead = (f"Silhouette leader: {name(sl)} (gap {abs(km['silhouette'] - w['silhouette']):.4f}) · "
            f"Davies–Bouldin leader: {name(dl)} (gap {abs(km['db'] - w['db']):.4f}).<br>")
    if sl is None and dl is None:
        lead += f"Both methods score the same on both metrics at K={km['k']}."
    elif sl == dl or None in (sl, dl):
        best = sl or dl
        lead += (f"{S.METHOD_TITLES[best]} leads on both metrics at K={km['k']} for this dataset." if sl == dl else
                 f"{S.METHOD_TITLES[best]} leads on one metric and ties on the other at K={km['k']}.")
    else:
        lead += "No unique overall winner from these two metrics alone — each method is preferred by one."
    UI.callout(lead, title="Leaders", tone="gold", html=True)


def _metric_chart(fitted):
    m = st.radio("Metric", list(METRICS), horizontal=True, key="eval_metric",
                 format_func=lambda v: v.replace("-", "–"))
    key, direction = METRICS[m]
    names = [S.METHOD_TITLES[x] for x in fitted]
    vals = [S.model(x)[key] for x in fitted]
    fig = go.Figure(go.Bar(x=names, y=vals, marker=dict(color=[METHOD_COLOR[x] for x in fitted]),
                           text=[f"{v:.4f}" for v in vals], textposition="outside", cliponaxis=False,
                           textfont=dict(color=T.TEXT_2), width=0.45,
                           hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    label = m.replace("-", "–")
    fig = CF.style(fig, 300, f"{label} by method", direction.capitalize(), None, label, legend=False)
    fig.update_yaxes(range=[0, max(vals) * 1.3])
    CF.show(fig, alt=f"Bar chart comparing {label} for {', '.join(names)}.")


def render():
    s = st.session_state
    UI.page_header("Cluster Quality", "How coherent and separated are the current clustering solutions?",
                   ("Automotive Fleet", "Modeling", "Cluster Quality"))
    if not S.has_std(s):
        UI.empty_state("PREPROCESSING INCOMPLETE", "Evaluation needs the standardized feature matrix.",
                       "Preprocessing → Apply imputation → Standardize")
        st.button("Open Preprocessing", on_click=S.goto, args=("Preprocessing",))
        return
    fitted = S.fitted_methods(s)
    if not fitted:
        UI.empty_state("NO CLUSTER MODEL", "Run clustering to unlock evaluation.", "Clustering Lab → RUN CLUSTERING")
        st.button("Open Clustering Lab", type="primary", on_click=S.goto, args=("Clustering Lab",))
        return
    km, w = S.model(S.KMEANS, s), S.model(S.WARD, s)
    k = (km or w)["k"]
    src = "project dataset" if S.is_project_data(s) else s["dataset_name"]
    UI.provenance(f"{src} · {len(s['df_std'])} vehicles · K={k} · standardized five-feature matrix", "Evaluating")
    st.caption("Every fitted method uses the selected K (changing K clears all fits), so their metrics are "
               "directly comparable.")
    _table(km, w)
    if km is not None and w is not None:
        _verdict(km, w)
    else:
        missing = S.WARD if km is not None else S.KMEANS
        UI.callout(f"Only {S.METHOD_TITLES[fitted[0]]} is fitted. Fit {S.METHOD_TITLES[missing]} at K={k} in the "
                   "Clustering Lab to compare the two methods.", tone="neutral")
    a, b = st.columns(2, gap="large")
    with a:
        UI.callout("Silhouette (−1 to 1): for each vehicle (b − a) / max(a, b), where a is the mean distance to its "
                   "own cluster and b to the nearest other cluster; averaged. Higher means tighter, better-separated "
                   "clusters.", title="Silhouette ↑", tone="teal")
    with b:
        UI.callout("Davies–Bouldin (≥ 0): the average, over clusters, of the worst ratio of within-cluster scatter "
                   "to between-centroid separation. Lower is better. Both metrics use the standardized features.",
                   title="Davies–Bouldin ↓", tone="teal")
    UI.section("Comparison", "Method colors: K-Means gold, Ward teal.")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        _metric_chart(fitted)
    with right:
        cols = st.columns(len(fitted))
        for col, m in zip(cols, fitted):
            with col:
                CF.show(CF.cluster_sizes(S.model(m, s)["counts"], f"{S.METHOD_TITLES[m]} sizes", f"K={k}"),
                        alt=f"Vehicles per cluster for {S.METHOD_TITLES[m]} at K={k}.")
        st.caption("Each method numbers its clusters independently, so the same number need not mean the same "
                   "vehicles" + (" — compare memberships in the cross-tab below." if len(fitted) > 1 else "."))
    if km is not None and w is not None:
        UI.section("How the two partitions overlap", "Vehicles per K-Means cluster (rows) and Ward cluster "
                   "(columns).")
        ct = pd.crosstab(pd.Series(km["labels"], name="K-Means"), pd.Series(w["labels"], name="Ward"))
        ct.index = [f"K-Means {i}" for i in ct.index]
        ct.columns = [f"Ward {j}" for j in ct.columns]
        st.dataframe(ct, width="stretch")
        st.caption(f"Partition agreement (adjusted Rand index): {adjusted_rand_score(km['labels'], w['labels']):.4f} "
                   "— 1 means identical groupings regardless of numbering.")
    if w is not None and s.get("dlab") is not None and s.get("dk") == w["k"]:
        ari = adjusted_rand_score(w["labels"], s["dlab"])
        UI.callout(f"Dendrogram-cut labels (fcluster, maxclust) vs AgglomerativeClustering Ward labels: adjusted Rand "
                   f"index {ari:.4f}. This is a consistency check between two Ward implementations, not a quality "
                   "metric.", title="Dendrogram-cut consistency", tone="neutral")
    if s.get("krange") is not None:
        with st.expander("Context: K-Means silhouette across K=2–6", key="tech_eval_sweep"):
            CF.show(CF.silhouette_by_k(s["krange"], int(s["selected_k"]),
                                       documented=S.DOCUMENTED_K if S.is_project_data(s) else None, height=300),
                    alt="Bar chart of K-Means silhouette for K=2–6 with the selected K highlighted.")
    UI.viva("Silhouette and Davies–Bouldin reward different structure, so both are reported side by side. No "
            "composite score is formed: weighting them would be an arbitrary choice.")
