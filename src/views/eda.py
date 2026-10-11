"""EDA Explorer - distributions, correlations and pairwise relationships."""
from itertools import combinations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

from src import chart_factory as CF
from src import state as S
from src import theme as T
from src import ui_components as UI
from src.auto_mpg_utils import (
    CONTINUOUS_COLS, CORR_COLS, calculate_correlations, correlation_table, strength_label,
)

LABEL = {"mpg": "MPG", "displacement": "Displacement (cu in)", "horsepower": "Horsepower (hp)",
         "weight": "Weight (lb)", "acceleration": "Acceleration (s)"}
SETS = {"Assignment set (weight, horsepower, mpg)": CORR_COLS,
        "All five continuous features": CONTINUOUS_COLS}


def _swap():
    s = st.session_state
    s["rel_x"], s["rel_y"] = s["rel_y"], s["rel_x"]


def _shape(sk):
    a = abs(sk)
    if a < 0.5:
        return "approximately symmetric"
    side = "right-skewed, with a longer upper tail" if sk > 0 else "left-skewed, with a longer lower tail"
    return f"{'moderately' if a < 1 else 'highly'} {side}"


def _distribution(df):
    UI.section("Distribution workspace", "Pick a feature; every control below changes the chart.")
    ctrl, chart, stats = st.columns([1, 2.4, 1.2], gap="medium")
    with ctrl:
        feat = st.selectbox("Feature", CONTINUOUS_COLS, key="eda_feat", format_func=LABEL.get)
        viz = st.selectbox("Visualization", ["Histogram", "Boxplot"], key="eda_viz")
        bins = st.slider("Bins", 10, 50, 24, key="eda_bins", disabled=viz != "Histogram")
        show_mean = st.checkbox("Show mean", value=True, key="eda_mean")
        show_med = st.checkbox("Show median", value=True, key="eda_median")
        show_kde = st.checkbox("Show density (KDE)", value=False, key="eda_kde", disabled=viz != "Histogram")
    x = df[feat].dropna().astype(float)
    mean, med, sk = float(x.mean()), float(x.median()), float(x.skew())
    with chart:
        if viz == "Histogram":
            counts, edges = np.histogram(x, bins=bins)
            fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
                                   marker=dict(color=T.TEAL, line=dict(width=1, color=T.BACKGROUND)),
                                   name="Vehicles", hovertemplate="%{x:.1f}: %{y} vehicles<extra></extra>"))
            if show_kde and x.nunique() > 1:
                grid = np.linspace(x.min(), x.max(), 200)
                fig.add_trace(go.Scatter(x=grid, y=gaussian_kde(x)(grid) * len(x) * (edges[1] - edges[0]),
                                         mode="lines", name="Density (scaled to counts)",
                                         line=dict(color=T.WHITE, width=2), hoverinfo="skip"))
            ylab = "Vehicles"
        else:
            fig = go.Figure(go.Box(y=x, name=LABEL[feat], marker=dict(color=T.TEAL), line=dict(color=T.TEAL),
                                   boxpoints="outliers", hovertemplate="%{y:.1f}<extra></extra>"))
            ylab = LABEL[feat]
        # Reference lines: muted, told apart by dash and label; labels sit on opposite sides so they never collide.
        if show_mean:
            CF.ref_line(fig, **({"x": mean} if viz == "Histogram" else {"y": mean}),
                        text=f"mean {mean:.2f}", color=T.MUTED, dash="dash",
                        position="top right" if mean >= med else "top left")
        if show_med:
            CF.ref_line(fig, **({"x": med} if viz == "Histogram" else {"y": med}),
                        text=f"median {med:.2f}", color=T.MUTED, dash="dot",
                        position="top left" if mean >= med else "top right")
        fig = CF.style(fig, 380, f"{LABEL[feat]} · {viz.lower()}", f"{len(x)} vehicles",
                       LABEL[feat] if viz == "Histogram" else None, ylab, legend=show_kde and viz == "Histogram")
        CF.show(fig, alt=f"{viz} of {LABEL[feat]} for {len(x)} vehicles.")
    with stats:
        UI.keyvals([("Vehicles (n)", f"{len(x)}"), ("Missing", f"{int(df[feat].isna().sum())}"),
                    ("Mean", f"{mean:.2f}"), ("Median", f"{med:.2f}"),
                    ("SD (sample, ddof=1)", f"{x.std():.2f}"), ("Min", f"{x.min():.2f}"),
                    ("Max", f"{x.max():.2f}"), ("Skewness", f"{sk:.3f}")])
    below = 100 * float((x < mean).mean())
    rel = "above" if mean > med else ("below" if mean < med else "equal to")
    UI.callout(f"<b>{UI.esc(LABEL[feat])}</b> is {_shape(sk)} (skewness {sk:.2f}). The mean ({mean:.2f}) is {rel} "
               f"the median ({med:.2f}), and {below:.0f}% of vehicles sit below the mean."
               "<br><span class='ui-next'>Rule used: |skewness| &lt; 0.5 approximately symmetric, 0.5–1 moderately "
               "skewed, &gt; 1 highly skewed (pandas sample skewness).</span>", title="Reading", tone="teal",
               html=True)


def _correlation(df):
    UI.section("Correlation", "Pearson correlation on the active data; association only — never causation.")
    choice = st.radio("Feature set", list(SETS), horizontal=True, key="eda_corr_set")
    cols = SETS[choice]
    corr = calculate_correlations(df) if cols is CORR_COLS else df[cols].corr()
    pairs = [f"{a}|{b}" for a, b in combinations(cols, 2)]
    if st.session_state.get("corr_pair") not in pairs:
        st.session_state["corr_pair"] = pairs[0]
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        CF.show(CF.diverging_heatmap(corr, "Correlation matrix", choice, fmt=".2f", height=360 + 18 * len(cols)),
                alt=f"Annotated correlation heatmap for {', '.join(cols)}.")
    with right:
        pair = st.selectbox("Pair", pairs, key="corr_pair", format_func=lambda p: p.replace("|", " ↔ "))
        a, b = pair.split("|")
        r = float(df[a].corr(df[b]))
        UI.kpis([{"label": "Pearson r", "value": f"{r:.4f}", "sub": f"{a} ↔ {b}"},
                 {"label": "Direction", "value": "Positive" if r > 0 else ("Negative" if r < 0 else "None"),
                  "sub": "sign of r"},
                 {"label": "Strength", "value": strength_label(abs(r)).capitalize(), "sub": "by |r|"}])
        st.caption("Strength thresholds: |r| ≥ 0.7 strong · ≥ 0.4 moderate · ≥ 0.2 weak · otherwise very weak / "
                   "negligible. Correlation describes association and does not establish causation.")
    st.dataframe(correlation_table(df), hide_index=True, width="stretch",
                 column_config={"Correlation": st.column_config.NumberColumn(format="%.4f")})


def _relationships(s, df):
    UI.section("Relationship explorer", "Any two continuous features; hover a point for the vehicle.")
    a, b, c, d = st.columns([1.2, 1.2, 0.8, 1.4], vertical_alignment="bottom")
    xa = a.selectbox("X", CONTINUOUS_COLS, key="rel_x", format_func=LABEL.get)
    ya = b.selectbox("Y", CONTINUOUS_COLS, key="rel_y", format_func=LABEL.get)
    c.button("Swap Axes", width="stretch", icon=":material/swap_horiz:", on_click=_swap)
    fitted = S.fitted_methods(s)
    color_opts = ["None"] + [f"{S.METHOD_TITLES[m]} clusters" for m in fitted]
    color = d.radio("Color by", color_opts, horizontal=True, key="eda_colour")
    trend = st.checkbox("Least-squares trendline", key="eda_trend")
    cols = list(dict.fromkeys(["car_name", "cylinders", "mpg", "horsepower", "weight", "displacement", xa, ya]))
    sub = df[cols].dropna(subset=[xa, ya])
    hover = [("cylinders", "Cylinders", ".0f"), ("mpg", "MPG", ".1f"), ("horsepower", "HP", ".0f"),
             ("weight", "Weight (lb)", ",.0f"), ("displacement", "Displacement", ".0f")]
    if color != "None":
        m = fitted[color_opts.index(color) - 1]
        r = S.model(m, s)
        lab = np.asarray(r["labels"])[df.index.get_indexer(sub.index)]  # labels are row-positional
        fig = CF.cluster_scatter(sub, xa, ya, lab, r["k"], xtitle=LABEL[xa], ytitle=LABEL[ya],
                                 title=f"{LABEL[ya]} vs {LABEL[xa]}", subtitle=f"{S.METHOD_TITLES[m]} · K={r['k']}",
                                 hover=hover, height=430)
    else:
        data = sub[[h[0] for h in hover]].to_numpy()
        tpl = ("<b>%{hovertext}</b><br>" + f"{LABEL[xa]} %{{x:.2f}} · {LABEL[ya]} %{{y:.2f}}<br>"
               + "<br>".join(f"{lab_} %{{customdata[{j}]:{fmt}}}" for j, (_, lab_, fmt) in enumerate(hover))
               + "<extra></extra>")
        fig = go.Figure(go.Scatter(x=sub[xa], y=sub[ya], mode="markers", name="Vehicles",
                                   hovertext=CF.plain(sub["car_name"]), customdata=data, hovertemplate=tpl,
                                   marker=dict(color=T.TEAL, size=8, opacity=0.8,
                                               line=dict(width=0.6, color=T.BACKGROUND))))
        fig = CF.style(fig, 430, f"{LABEL[ya]} vs {LABEL[xa]}", f"{len(sub)} vehicles", LABEL[xa], LABEL[ya],
                       legend=trend)
    if trend and sub[xa].nunique() > 1:
        slope, icpt = np.polyfit(sub[xa], sub[ya], 1)
        r_xy = float(np.corrcoef(sub[xa], sub[ya])[0, 1])
        xs = np.array([sub[xa].min(), sub[xa].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", line=dict(color=T.GOLD, width=2),
                                 name=f"Least squares: y = {slope:.4g}·x + {icpt:.4g} (r = {r_xy:.3f})",
                                 hoverinfo="skip"))
        fig.update_layout(showlegend=True)
    CF.show(fig, alt=f"Scatter of {LABEL[ya]} against {LABEL[xa]} for {len(sub)} vehicles.")
    if trend:
        st.caption("The trendline summarises a linear association in this sample; it is not a causal model.")


def render():
    s = st.session_state
    UI.page_header("EDA Explorer", "Explore distributions, correlations and pairwise relationships in the "
                   "active data.", ("Automotive Fleet", "Analysis", "EDA Explorer"))
    if not S.has_raw(s):
        UI.empty_state("DATASET NOT READY", "Load or upload an Auto MPG dataset to explore it.",
                       "Dataset → Use Project Dataset")
        st.button("Open Dataset", on_click=S.goto, args=("Dataset",))
        return
    df = s["df_clean"] if S.has_clean(s) else s["df_raw"]
    if S.has_clean(s):
        st.caption("Data basis: cleaned data (after cylinder-group horsepower imputation).")
    else:
        st.caption(f"Data basis: raw data — {int(df['horsepower'].isna().sum())} missing horsepower values are "
                   "excluded from horsepower statistics until imputation on Preprocessing.")
    _distribution(df)
    _correlation(df)
    _relationships(s, df)
    UI.viva("Skewness describes the shape of one feature; correlation measures linear association between two. "
            "Neither establishes causation: a strong correlation says the features move together in this sample, "
            "not why.")
