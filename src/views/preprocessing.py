"""Preprocessing - make imputation and standardization visible."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import chart_factory as CF
from src import state as S
from src import theme as T
from src import ui_components as UI
from src.auto_mpg_utils import CONTINUOUS_COLS, clean_horsepower, standardized_statistics

UNIT = {"mpg": "MPG", "displacement": "cu in", "horsepower": "hp", "weight": "lb", "acceleration": "s"}


def _impute():
    S.apply_imputation()
    s = st.session_state
    s["flash"] = f"Horsepower imputed · missing {s['miss_before']} → {s['miss_after']}"


def _standardize():
    if S.apply_standardization():
        st.session_state["flash"] = "Five features standardized · fitted models cleared"


def _group_table(raw):
    g = raw.groupby("cylinders")["horsepower"]
    t = pd.DataFrame({"Vehicles": g.size(), "Observed HP": g.count(), "Missing HP": g.size() - g.count(),
                      "Median HP": g.median()}).reset_index().rename(columns={"cylinders": "Cylinders"})
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "Cylinders": st.column_config.NumberColumn(format="%d"),
        "Median HP": st.column_config.NumberColumn("Median HP (group)", format="%.1f")})


def _affected(s, raw):
    mask = raw["horsepower"].isna()
    if not mask.any():
        st.caption("No vehicle in the active dataset is missing horsepower.")
        return
    rows = raw.loc[mask, ["car_name", "cylinders"]].copy()
    rows["Original horsepower"] = "missing"
    if S.has_clean(s):
        rows["Imputed horsepower"] = s["df_clean"].loc[mask, "horsepower"]
        rows["Group median used"] = rows["cylinders"].map(s["medians"])
    else:
        rows["Imputed horsepower"] = "pending"
    st.dataframe(rows.rename(columns={"car_name": "Vehicle", "cylinders": "Cylinders"}), hide_index=True,
                 width="stretch", column_config={
                     "Cylinders": st.column_config.NumberColumn(format="%d"),
                     "Group median used": st.column_config.NumberColumn(format="%.1f")})


def _spread_chart(base):
    sd = base[CONTINUOUS_COLS].std(ddof=0)
    fig = go.Figure(go.Bar(x=sd.values, y=[f"{c} ({UNIT[c]})" for c in sd.index], orientation="h",
                           marker=dict(color=T.TEAL), text=[f"{v:,.2f}" for v in sd.values],
                           textposition="outside", cliponaxis=False, textfont=dict(color=T.TEXT_2),
                           hovertemplate="%{y}: SD %{x:,.2f}<extra></extra>"))
    fig = CF.style(fig, 260, "Spread before scaling", "Population standard deviation in original units",
                   "Standard deviation", None, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, float(sd.max()) * 1.2])
    CF.show(fig, alt="Horizontal bar chart of each feature's standard deviation before standardization.")
    return sd


def render():
    s = st.session_state
    UI.page_header("Preprocessing", "Impute missing horsepower and standardize the five clustering features.",
                   ("Automotive Fleet", "Analysis", "Preprocessing"))
    if not S.has_raw(s):
        UI.empty_state("DATASET NOT READY", "Load or upload an Auto MPG dataset before preprocessing.",
                       "Dataset → Use Project Dataset")
        st.button("Open Dataset", on_click=S.goto, args=("Dataset",))
        return
    raw = clean_horsepower(s["df_raw"])
    mb = int(raw["horsepower"].isna().sum())
    clean, std_ = S.has_clean(s), S.has_std(s)
    src = "the project dataset" if S.is_project_data(s) else s["dataset_name"]

    with st.container(border=True):
        UI.step_header(1, "RAW DATA", "complete", f"{len(raw):,} rows × {raw.shape[1]} fields loaded from {src}.")
    UI.arrow()
    with st.container(border=True):
        UI.step_header(2, "MISSING-VALUE IDENTIFICATION", "complete",
                       f"'?' entries were parsed as missing: {mb} horsepower value(s) need imputation.")
    UI.arrow()
    with st.container(border=True):
        UI.step_header(3, "CYLINDER-GROUP HORSEPOWER MEDIAN IMPUTATION", "complete" if clean else "ready",
                       "Each missing horsepower takes the median horsepower of vehicles with the same cylinder "
                       "count. A global-median fallback applies when a vehicle's cylinder count is missing or its whole "
                       "group has no observed horsepower.")
        a, b = st.columns([1, 1.25], gap="large")
        with a:
            st.button("Apply Cylinder-Group Median Imputation", type="secondary" if clean else "primary",
                      width="stretch", icon=":material/healing:", on_click=_impute,
                      help="Re-applying recomputes the medians and clears every downstream result.")
            UI.kpis([{"label": "Missing HP before", "value": str(mb), "sub": "raw file"},
                     {"label": "Missing HP after", "value": str(s["miss_after"]) if clean else "pending",
                      "sub": ("global-median fallback used" if s.get("fallback_used") else "no fallback needed")
                      if clean else "apply imputation", "tone": "success" if clean else None}])
        with b:
            st.caption("Group medians (computed from the observed raw values)")
            _group_table(raw)
        st.caption("Affected records — imputed values are estimates, not observations.")
        _affected(s, raw)
    UI.arrow()
    with st.container(border=True):
        UI.step_header(4, "STANDARDIZATION", "complete" if std_ else ("ready" if clean else "blocked"),
                       "z = (x − μ) / σ for mpg, displacement, horsepower, weight and acceleration. "
                       "StandardScaler divides by the population standard deviation (ddof = 0).")
        a, b = st.columns([1, 1.3], gap="large")
        with a:
            st.button("Standardize Features", type="primary" if clean and not std_ else "secondary",
                      width="stretch", icon=":material/straighten:", disabled=not clean, on_click=_standardize,
                      help="Re-standardizing clears every fitted model." if clean
                      else "Apply horsepower imputation first.")
            if not clean:
                st.caption("Available after imputation — scaling needs complete horsepower values.")
            base = s["df_clean"] if clean else raw
            sd = base[CONTINUOUS_COLS].std(ddof=0)
            big, small = sd.idxmax(), sd.idxmin()
            st.markdown(f"Raw features live on very different scales: **{big}** varies with SD "
                        f"{sd[big]:,.1f} {UNIT[big]}, **{small}** with SD {sd[small]:,.2f} {UNIT[small]}. "
                        "Unscaled, the largest-magnitude feature would dominate Euclidean distance simply "
                        "because of its units.")
        with b:
            _spread_chart(base)
        before = pd.DataFrame({"Mean": base[CONTINUOUS_COLS].mean(), "Population SD": sd})
        before.index = [f"{c} ({UNIT[c]})" for c in before.index]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.caption("Before — original units" + (" (after imputation)" if clean else " (raw)"))
            st.dataframe(before, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%.3f") for c in before.columns})
        with c2:
            st.caption("After — standardized (mean ≈ 0, population SD = 1)")
            if std_:
                st.dataframe(standardized_statistics(s["df_std"]), width="stretch", column_config={
                    "Mean": st.column_config.NumberColumn(format="%.2e"),
                    "Sample_STD": st.column_config.NumberColumn("Sample SD (ddof=1)", format="%.6f"),
                    "Population_STD": st.column_config.NumberColumn("Population SD (ddof=0)", format="%.6f")})
            else:
                st.caption("Standardize the features to see the scaled statistics.")
    UI.arrow()
    with st.container(border=True):
        UI.step_header(5, "READY FOR CLUSTERING", "complete" if std_ else "blocked",
                       "The standardized five-feature matrix feeds K-Means, Ward, the metrics and PCA."
                       if std_ else "Complete imputation and standardization first.")
        if std_:
            st.button("Open Clustering Lab", type="primary", icon=":material/bubble_chart:",
                      on_click=S.goto, args=("Clustering Lab",))
    UI.viva("Engines with the same cylinder count share a comparable power band, so the cylinder-group median "
            "keeps that structure; a single global median would pull small cars up and big cars down.")
    UI.viva("K-Means and Ward use Euclidean distance. Without z-scores, weight (thousands of lb) would outweigh "
            "MPG (tens) purely because of its units.")
