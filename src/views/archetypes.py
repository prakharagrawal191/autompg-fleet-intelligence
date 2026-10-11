"""Fleet Archetypes - data-driven profiles from the K-Means centroids."""
import numpy as np
import pandas as pd
import streamlit as st

from src import chart_factory as CF
from src import state as S
from src import ui_components as UI
from src.auto_mpg_utils import CONTINUOUS_COLS, find_representative_vehicles

UNIT = {"mpg": "MPG", "displacement": "cu in", "horsepower": "hp", "weight": "lb", "acceleration": "s"}


def _pct(value, base):
    return None if abs(base) < 1e-9 else 100 * (value - base) / base


def _pct_text(p):
    return "n/a" if p is None else f"{p:+.1f}%"


def _cards(k, co, names, counts, fleet):
    total = int(counts.sum())
    for start in range(0, k, 3):
        cols = st.columns(3)
        for i, col in zip(range(start, min(k, start + 3)), cols):
            row = co.loc[i]
            vs = " · ".join(f"{f} {_pct_text(_pct(row[f], fleet[f]))}" for f in ("mpg", "weight", "horsepower"))
            with col:
                UI.cluster_card(i, names.get(i, f"Cluster {i}"), f"{row['mpg']:.2f}", "MPG",
                                [("Weight", f"{row['weight']:,.0f} lb"), ("Horsepower", f"{row['horsepower']:.1f} hp"),
                                 ("Displacement", f"{row['displacement']:.1f} cu in"),
                                 ("Acceleration", f"{row['acceleration']:.2f} s")],
                                f"{int(counts.get(i, 0))} vehicles · {100 * counts.get(i, 0) / total:.1f}%",
                                f"vs fleet mean: {vs}")


def _centroid_view(cs, co, names):
    UI.section("Centroid comparison", "Same centroids, two representations — the toggle changes presentation only.")
    units = st.radio("Centroid units", ["Standardized (z-score)", "Original units"], horizontal=True,
                     key="arch_units")
    if units.startswith("Standardized"):
        z = cs.copy()
        z.columns = [c.replace(" (z)", "") for c in z.columns]
        z.index = [f"Cluster {i} · {names.get(i, '')}" for i in z.index]
        lim = float(np.ceil(np.abs(z.to_numpy()).max() * 10) / 10)
        CF.show(CF.diverging_heatmap(z, "Centroids in standardized units",
                                     "0 = fleet mean · ±1 = one fleet standard deviation", zmin=-lim, zmax=lim,
                                     fmt=".2f", colorbar="z", height=150 + 46 * len(z)),
                alt="Heatmap of K-Means centroid z-scores by cluster and feature.")
    else:
        t = co.copy()
        t.columns = [f"{c} ({UNIT[c]})" for c in t.columns]
        t.insert(0, "Archetype", [names.get(i, "") for i in t.index])
        t.index = [f"Cluster {i}" for i in t.index]
        st.dataframe(t, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.2f") for c in t.columns if c != "Archetype"})
        st.caption("Original units come from scaler.inverse_transform of the standardized centroids.")


def _compare(k, co, names):
    UI.section("Compare archetypes", "Centroid values side by side; differences are relative to the first pick.")
    sel = [int(i) for i in (st.session_state.get("arch_compare") or [])
           if isinstance(i, (int, np.integer)) and 0 <= int(i) < k]
    st.session_state["arch_compare"] = sel  # prune stale ids before the widget renders
    st.multiselect("Pick two or more archetypes to compare", list(range(k)), key="arch_compare",
                   format_func=lambda i: f"Cluster {i} · {names.get(i, '')}",
                   help="Display only — never touches the model.")
    sel = list(st.session_state.get("arch_compare") or [])
    if len(sel) < 2:
        st.caption("Select at least two archetypes to see the comparison.")
        return
    base = sel[0]
    rows = []
    for f in CONTINUOUS_COLS:
        r = {"Feature": f"{f} ({UNIT[f]})"}
        for i in sel:
            r[f"Cluster {i}"] = co.loc[i, f]
        for i in sel[1:]:
            r[f"Δ Cluster {i} − {base}"] = co.loc[i, f] - co.loc[base, f]
        rows.append(r)
    t = pd.DataFrame(rows)
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%+.2f" if c.startswith("Δ") else "%.2f")
        for c in t.columns if c != "Feature"})


def _representatives(s, km, names):
    reps = find_representative_vehicles(s["df_clean"], S.X(s), km["centers"], 5)
    exact = any(float(r["distance_to_centroid"].min()) == 0.0 for r in reps.values())
    UI.section("Representative vehicles", "The five real vehicles nearest each centroid (Euclidean distance in the "
               "5-D standardized space) — deterministic, never random"
               + (". A distance of 0 means the vehicle is the whole cluster." if exact
                  else ", and none sits exactly on its centroid."))
    tabs = st.tabs([f"Cluster {i} · {names.get(i, '')}" for i in range(km["k"])])
    for i, tab in enumerate(tabs):
        with tab:
            st.dataframe(reps[i], hide_index=True, width="stretch", column_config={
                "car_name": st.column_config.TextColumn("Vehicle"),
                "mpg": st.column_config.NumberColumn("MPG", format="%.1f"),
                "horsepower": st.column_config.NumberColumn("Horsepower (hp)", format="%.0f"),
                "weight": st.column_config.NumberColumn("Weight (lb)", format="%.0f"),
                "displacement": st.column_config.NumberColumn("Displacement (cu in)", format="%.0f"),
                "cylinders": st.column_config.NumberColumn("Cylinders", format="%d"),
                "distance_to_centroid": st.column_config.NumberColumn(
                    "Distance to centroid (standardized units)", format="%.4f")})


def _why(k, co, cs, names, fleet):
    UI.section("Why this cluster?", "Where a centroid sits relative to the fleet mean (baseline: mean of the "
               "cleaned dataset).")
    if st.session_state.get("why") not in range(k):
        st.session_state["why"] = 0
    c = st.selectbox("Explain cluster", list(range(k)), key="why",
                     format_func=lambda i: f"Cluster {i} · {names.get(i, '')}")
    rows = [{"Feature": f"{f} ({UNIT[f]})", "Centroid": co.loc[c, f], "Fleet mean": fleet[f],
             "Difference vs fleet": _pct_text(_pct(co.loc[c, f], fleet[f])), "Centroid z-score": cs.loc[c, f + " (z)"]}
            for f in CONTINUOUS_COLS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Centroid": st.column_config.NumberColumn(format="%.2f"),
        "Fleet mean": st.column_config.NumberColumn(format="%.2f"),
        "Centroid z-score": st.column_config.NumberColumn(format="%+.2f")})
    above = [f for f in CONTINUOUS_COLS if co.loc[c, f] > fleet[f]]
    below = [f for f in CONTINUOUS_COLS if co.loc[c, f] < fleet[f]]
    UI.callout(f"Cluster {c} ({names.get(c, '')}) sits above the fleet mean on {', '.join(above) or 'no feature'} "
               f"and below it on {', '.join(below) or 'no feature'}. This describes the centroid's position; it "
               "does not explain why the vehicles differ.", tone="teal")


def render():
    s = st.session_state
    UI.page_header("Fleet Archetypes", "Data-driven vehicle profiles derived from the K-Means centroids.",
                   ("Automotive Fleet", "Interpretation", "Fleet Archetypes"))
    if not S.has_std(s):
        UI.empty_state("PREPROCESSING INCOMPLETE", "Archetypes come from clustering the standardized features.",
                       "Preprocessing → Apply imputation → Standardize")
        st.button("Open Preprocessing", on_click=S.goto, args=("Preprocessing",))
        return
    km = S.model(S.KMEANS, s)
    if km is None or not s.get("arch") or s.get("cent_orig") is None:
        hint = " Ward is the active method; switch to K-Means in the Clustering Lab." if s["method"] == S.WARD else ""
        UI.empty_state("ARCHETYPES NOT AVAILABLE",
                       "Fit K-Means for the current dataset first — archetypes are derived from its centroids." + hint,
                       "Clustering Lab → Use K-Means → RUN CLUSTERING")
        st.button("Open Clustering Lab", type="primary", on_click=S.goto, args=("Clustering Lab",))
        return
    k, names = km["k"], s["arch"]
    co = s["cent_orig"].set_index("cluster")
    cs = s["cent_std"].set_index("cluster")
    fleet = s["df_clean"][CONTINUOUS_COLS].mean()
    UI.provenance(S.provenance(S.KMEANS, s), "Derived from")
    UI.callout(s["arch_rule"], title="How names are assigned", tone="neutral")
    _cards(k, co, names, km["counts"], fleet)
    _centroid_view(cs, co, names)
    _compare(k, co, names)
    _representatives(s, km, names)
    _why(k, co, cs, names, fleet)
    UI.viva("K-Means centroids live in standardized space; scaler.inverse_transform maps them back to MPG, lb and "
            "hp. Names come from ranking those centroids (weight + horsepower, MPG vs weight), so they change "
            "with the data and with K.")
