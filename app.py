"""Automotive Fleet Profiling & Efficiency Clustering - Professional Dashboard.
Preserves validated ML methodology; visual redesign only.
M.Tech ML Lab Q4 Assessment 10 - 26MML0003.
"""
import hashlib
import io
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from auto_mpg_utils import (  # noqa: E402  (validated science — do not move)
    CONTINUOUS_COLS, DEFAULT_N_INIT, DEFAULT_RANDOM_STATE,
    K_RANGE, assign_archetypes, build_dendrogram,
    calculate_centroids, calculate_correlations, clean_horsepower,
    correlation_table, dendro_cut_height,
    elbow_suggestion, evaluate_clustering, evaluate_kmeans,
    evaluate_kmeans_range, extract_dendrogram_clusters,
    find_representative_vehicles, fit_ward,
    impute_horsepower_by_cylinder, load_auto_mpg,
    optimal_k_from_silhouette, run_pca, standardize_features,
    standardized_statistics, strength_label, validate_schema,
)
sys.path.insert(0, os.path.dirname(__file__))
from src import theme as TH  # noqa: E402  (presentation only)
from src import ui_components as UI  # noqa: E402  (presentation only)
from src import chart_factory as CF  # noqa: E402  (presentation only)
st.set_page_config(
    page_title="Automotive Fleet Profiling | Auto MPG Analytics",
    page_icon=":material/directions_car:",
    layout="wide", initial_sidebar_state="auto",
)
MUTED = TH.MUTED
CCOLS = TH.cluster_colors(6)
st.markdown(UI.css_base(bool(st.session_state.get("presentation_mode",
                                                           False))),
            unsafe_allow_html=True)
UI.render_topbar(bool(st.session_state.get("df_raw") is not None))
DEFS = {"df_raw": None, "df_clean": None, "df_std": None, "scaler": None,
        "medians": None, "miss_before": None, "miss_after": None,
        "dataset_id": None, "kmeans": None, "ward": None, "krange": None,
        "dendro": None, "dlab": None, "dk": None, "pca": None,
        "cent_std": None, "cent_orig": None, "arch": None, "arch_rule": None,
        "selected_k": 3, "method": "K-Means", "std_done": False,
        "imp_done": False, "page": "Overview", "demo_mode": False,
        "corr_pair": "weight|mpg", "eda_feat": "mpg", "eda_viz": "Histogram",
        "rel_x": "weight", "rel_y": "mpg", "eval_metric": "Silhouette",
        "pca_color": "K-Means", "arch_pick": 0, "why": 0, "arch_compare": [],
        "cmp_a": 0, "cmp_b": 1, "lookup_car": None,
        "lookup_va": None, "lookup_vb": None, "lookup_vc": None}
# Single source of truth for programmatic navigation.
# Buttons must only set "page" (a non-widget key). Sidebar radios are
# aligned to it BEFORE they instantiate (see pre-sidebar sync below).
# Never assign nav1..nav4 after the sidebar has rendered in the same run.
def goto_page(target):
    st.session_state["page"] = target
    st.rerun()
for _k, _v in DEFS.items():
    st.session_state.setdefault(_k, _v)
# Product modes (presentation-only; never alter computation).
for _k, _v in [("presentation_mode", False), ("demo_step", 0),
                ("viva_mode", True), ("pca_size", 7),
                ("pca_labels", False), ("pca_centroids", True),
                ("arch_compare", []), ("dendro_detail", False)]:
    st.session_state.setdefault(_k, _v)
def HR(): return st.session_state["df_raw"] is not None
def HC(): return st.session_state["df_clean"] is not None
def HS(): return st.session_state["df_std"] is not None
def HK(): return st.session_state["kmeans"] is not None
def HW(): return st.session_state["ward"] is not None
def XS(): return st.session_state["df_std"][CONTINUOUS_COLS].to_numpy()
def dark(fig, h=400, title=""):
    CF.style(fig, h, title)
    try:
        fig.update_layout(transition_duration=400)
    except Exception:
        pass
    for ax in (fig.layout.xaxis, fig.layout.yaxis):
        try:
            if ax: ax.title.font.color = "#CBD5E1"
        except Exception:
            pass
    return CF.toolbar_off(fig)
def stage_states():
    s = st.session_state
    return [
        ("01", "DATA", "done" if HR() else "pending"),
        ("02", "CLEAN", "done" if HC() else ("ready" if HR() else "pending")),
        ("03", "SCALE", "done" if HS() else ("ready" if HC() else "pending")),
        ("04", "EXPLORE", "done" if HR() else "pending"),
        ("05", "CLUSTER", "done" if (HK() or HW()) else ("ready" if HS() else "pending")),
        ("06", "EVALUATE", "done" if (HK() or HW()) else "pending"),
        ("07", "PCA", "done" if s["pca"] is not None else ("ready" if (HK() or HW()) else "pending")),
        ("08", "ARCHETYPES", "done" if s["arch"] is not None else ("ready" if HK() else "pending")),
    ]
def status_bar():
    s = st.session_state
    ds = "LOADED" if HR() else "EMPTY"
    rows = f"{s['df_raw'].shape[0]}" if HR() else "--"
    if s["std_done"]: pre = "READY"
    elif s["imp_done"]: pre = "PARTIAL"
    else: pre = "PENDING"
    # Always reflect the live selected K from session state (updates instantly on slider change).
    cl = f"K={s['selected_k']}" + ("" if (HK() or HW()) else " (SELECTED)")
    st.markdown(
        f"<div class='sbar'><span>DATASET <b>{ds}</b></span>"
        f"<span>ROWS <b>{rows}</b></span>"
        f"<span>PREPROCESS <b>{pre}</b></span>"
        f"<span>CLUSTERS <b>{cl}</b></span>"
        f"<span>MODEL <b>{s['method'].upper()}</b></span></div>",
        unsafe_allow_html=True)
def header(eyebrow, title, blurb=""):
    st.markdown(f"<div class='pagehead'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1></div>", unsafe_allow_html=True)
    if blurb: st.markdown(f"<div class='pageblurb'>{blurb}</div>", unsafe_allow_html=True)
    st.markdown("<div class='pageid'>Machine Learning Lab \u2022 Question 4 \u2022 Assessment 10 \u2022 26MML0003</div>", unsafe_allow_html=True)
    status_bar()
    if st.session_state.get("demo_mode"):
        st.markdown("<div class='demobanner'>🎬 <b>Demo Mode</b> — follow the pipeline tracker: DATA → CLEAN → SCALE → EXPLORE → CLUSTER → EVALUATE → PCA → ARCHETYPES. Computation unchanged.</div>", unsafe_allow_html=True)
    st.markdown("<hr class='rule'/>", unsafe_allow_html=True)
def kpi_row(cards):
    html = ["<div class='kpirow'>"]
    for i, (lbl, val, sub, teal) in enumerate(cards):
        cls = "kpi teal" if teal else "kpi"
        html.append(f"<div class='{cls}' style='animation-delay:{i*0.07}s'><div class='l'>{lbl}</div><div class='v'>{val}</div><div class='s'>{sub}</div></div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
def empty_state(title, body, hint=""):
    st.markdown(f"<div class='empty'><b>{title}</b><br>{body}<br><span style='font-size:12px'>{hint}</span></div>", unsafe_allow_html=True)
@st.cache_data(show_spinner=False)
def ckrange(X, did, rs, ni):
    return evaluate_kmeans_range(np.asarray(X), list(K_RANGE), rs, ni)
@st.cache_data(show_spinner=False)
def cpca(X, did):
    return run_pca(np.asarray(X), 2)
def fp(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]
def need_std():
    if not HR():
        empty_state("DATASET NOT READY", "Upload a valid Auto MPG dataset before continuing.", "Go to Dataset \u2192 Use Project Dataset.")
        return False
    if not HS():
        empty_state("PREPROCESSING INCOMPLETE", "Run imputation and standardization first.", "Go to Preprocessing \u2192 Apply imputation \u2192 Standardize.")
        return False
    return True
WIDGET_KEYS = {"demo_mode", "presentation_mode", "viva_mode", "nav1", "nav2", "nav3", "nav4",
             "arch_pick", "why", "arch_compare", "cmp_a", "cmp_b",
             "lookup_car", "lookup_va", "lookup_vb", "lookup_vc",
             "eval_metric", "pca_color", "pca_centroids", "pca_labels", "pca_size"}  # live widget keys; assign only pre-instantiation / in callbacks
def reset_state(keep_nav=True):
    s = st.session_state
    # Preserve navigation intent; skip live widget keys (Streamlit forbids
    # assigning/deleting an already-instantiated widget key mid-run). Stale
    # widget selections bound to results are clamped where each widget is
    # created, before instantiation.
    keep = {k: s[k] for k in ["page"] if keep_nav and k in s}
    for k, v in DEFS.items():
        if k in WIDGET_KEYS:
            continue
        if keep_nav and k in keep: continue
        s[k] = v
    s.update(keep)
    # ANALYSIS RESET must also drop cached sweeps/projections: they are keyed
    # by dataset_id upstream, so a same-shaped new upload could otherwise
    # reuse stale K=2..6 / PCA artefacts downstream (scientific safety).
    try: ckrange.clear()
    except Exception: pass
    try: cpca.clear()
    except Exception: pass
def invalidate_downstream():
    """Reset everything downstream of imputation (scale + models + derived state)."""
    s = st.session_state
    for k in ["df_std", "scaler", "std_done", "kmeans", "ward", "krange", "dendro",
              "dlab", "dk", "pca", "cent_std", "cent_orig", "arch", "arch_rule",
              "why", "arch_compare", "cmp_a", "cmp_b", "arch_pick"]:
        s[k] = DEFS[k]
def pipeline_tracker():
    import streamlit as st
    steps = stage_states()
    st.markdown("<div class='pipe'>" + "".join(
        f"<div class='pstep {stt}'><div class='n'>{n}</div><div class='t'>{t}</div>"
        f"<div class='n'>{'DONE' if stt=='done' else stt.upper()}</div></div>"
        for n, t, stt in steps) + "</div>", unsafe_allow_html=True)
def pg_overview():
    UI.render_crumbs("Automotive Fleet", "Overview")
    header("AUTOMOTIVE FLEET PROFILING", "Efficiency Clustering & Vehicle Archetype Intelligence", "Interactive efficiency and clustering analysis over the Auto MPG fleet. Follow the guided path below.")
    st.markdown("""<div class='hero'><div class='meta'>QUESTION 4 \u00b7 ASSESSMENT 10 \u00b7 MACHINE LEARNING LAB</div><div class='desc'>Cylinder-group median imputation, Z-score scaling, K-Means and Ward clustering over K=2\u20136, PCA visualization and data-driven fleet archetypes. Every number below is computed live from the active dataset.</div></div>""", unsafe_allow_html=True)
    pipeline_tracker()
    s = st.session_state
    if not HR():
        empty_state("DATASET NOT READY", "Upload a valid Auto MPG dataset before continuing.", "Go to Dataset \u2192 Use Project Dataset.")
        cta1, cta2 = st.columns(2)
        with cta1:
            if st.button("Open Dataset Page", type="primary", use_container_width=True):
                goto_page("Dataset")
        with cta2:
            if st.button("Review Methodology", use_container_width=True):
                goto_page("Methodology")
        if st.session_state.get("viva_mode"):
            UI.viva_box("This dashboard mirrors Assessment 10 exactly: cylinder-group median imputation, Z-score scaling of the five continuous features, then K-Means (random_state=42, n_init=10) and Ward linkage over K=2-6. Nothing is hard-coded; every number below is computed live.")
        return
    mb = s["miss_before"] if s["miss_before"] is not None else int(clean_horsepower(s["df_raw"])["horsepower"].isna().sum())
    ma = s["miss_after"] if s["miss_after"] is not None else (s["df_clean"]["horsepower"].isna().sum() if HC() else "--")
    try:
        pv = f"{100*float(np.sum(s['pca']['variance_ratio'])):.2f}%" if s["pca"] is not None else (f"{100*float(np.sum(cpca(XS(), s['dataset_id'])['variance_ratio'])):.2f}%" if HS() else "--")
    except Exception: pv = "--"
    kpi_row([("VEHICLES", f"{s['df_raw'].shape[0]}", "rows in current dataset", False),
             ("RAW FEATURES", f"{s['df_raw'].shape[1]}", "columns incl. car name", False),
             ("MISSING HP", f"{mb} -> {ma}", "before / after", True),
             ("SELECTED CLUSTERS", f"K = {s['selected_k']}", s["method"], True),
             ("PCA COVERAGE", pv, "PC1 + PC2", False)])
    st.caption("Silhouette: higher generally indicates better separation/cohesion. Davies-Bouldin: lower generally indicates better compactness/separation. PCA coverage: variance kept by the first two components.")
def render_walkthrough(s):
    UI.render_section_header("Guided walkthrough", "Advance one step at a time; every step lands on a live page.")
    steps = [
        ("Load data", "Dataset", "Import the Auto MPG file and inspect quality."),
        ("Impute & scale", "Preprocessing", "Cylinder-group median imputation, then z-score."),
        ("Explore", "EDA Explorer", "Distributions, correlations, scatter relationships."),
        ("Cluster", "Clustering Lab", "Pick K, train K-Means or Ward."),
        ("Validate", "Cluster Quality", "Silhouette, Davies-Bouldin, elbow, K sweep."),
        ("Project", "PCA Explorer", "2-D projection with centroids."),
        ("Interpret", "Fleet Archetypes", "Named profiles and representative vehicles."),
        ("Report", "Exports", "Download results and finish."),
    ]
    step = int(s.get("demo_step", 0)) % len(steps)
    title, page, blurb = steps[step]
    st.markdown(f"<div class='darkpanel'><b>STEP {step + 1}/{len(steps)} · {title}</b><br>{blurb}<br><span style='color:#94A3B8;font-size:12px'>Target page: {page}</span></div>", unsafe_allow_html=True)
    nc1, nc2, nc3 = st.columns(3)
    if nc1.button("◀ Previous step", disabled=(step == 0), use_container_width=True):
        s["demo_step"] = step - 1
        st.rerun()
    if nc2.button("Next step ▶", type="primary", disabled=(step == len(steps) - 1), use_container_width=True):
        s["demo_step"] = step + 1
        st.rerun()
    if nc3.button(f"Go to {page}", use_container_width=True):
        goto_page(page)
def pg_overview_b():
    s = st.session_state
    if not HR():
        render_walkthrough(s)
        return
    UI.render_section_header("Executive snapshot", "What the current session state unlocks.")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='darkpanel'><b>Dataset</b><br>{s['df_raw'].shape[0]} vehicles<br><span style='color:#94A3B8;font-size:12px'>Imputation: {'complete' if HC() else 'pending'} - Scale: {'complete' if HS() else 'pending'}</span></div>", unsafe_allow_html=True)
    with c2:
        km, w = s["kmeans"], s["ward"]
        txt = "Not trained yet. Open Clustering Lab."
        if km is not None: txt = f"K-Means K={km['k']} - Sil {km['silhouette']:.4f} - DB {km['db']:.4f}"
        if w is not None: txt += f"<br>Ward K={w['k']} - Sil {w['silhouette']:.4f} - DB {w['db']:.4f}"
        st.markdown(f"<div class='darkpanel'><b>Current model</b><br>{txt}</div>", unsafe_allow_html=True)
    with c3:
        at = "Run K-Means to reveal archetypes." if s["arch"] is None else (" - ".join(sorted(set(s["arch"].values()))))
        st.markdown(f"<div class='darkpanel'><b>Key findings</b><br>{at}<br><span style='color:#94A3B8;font-size:12px'>See Insights for the data-driven report.</span></div>", unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        if st.button("Explore Dataset", use_container_width=True):
            goto_page("Dataset")
    with h2:
        if st.button("Open Clustering Lab", type="primary", use_container_width=True):
            goto_page("Clustering Lab")
    if s["demo_mode"]:
        UI.render_info_callout("<b>DEMO MODE</b> — Step through: Dataset → Preprocessing → EDA → Clustering Lab (K=3, Run) → Quality → PCA → Archetypes → Lookup → Exports. State advances only through real actions.", teal=True)
    render_walkthrough(s)
def load_df_into_state(df):
    df, miss = validate_schema(df)
    if miss:
        st.error(f"Missing required columns: {', '.join(miss)}")
        return False
    reset_state(keep_nav=True)
    st.session_state.update(df_raw=df, dataset_id=fp(df), miss_before=int(pd.to_numeric(df["horsepower"], errors="coerce").isna().sum()))
    return True
def pg_dataset():
    UI.render_crumbs("Automotive Fleet", "Dataset")
    header("STEP 01 - DATA", "Dataset Loading & Inspection", "Whitespace .data and CSV supported; '?' treated as missing.")
    flash = st.session_state.pop("flash", None)
    if flash: st.success(flash)
    st.markdown(
        "<div class='darkpanel' style='text-align:center;padding:22px 16px'>"
        "<b style='font-size:16px;letter-spacing:2px'>IMPORT AUTO MPG DATA</b><br>"
        "<span style='color:#94A3B8;font-size:12px;letter-spacing:1px'>DROP YOUR DATASET HERE - CSV / DATA / TXT</span><br>"
        "<span style='color:#CBD5E1;font-size:12.5px'>Use the uploader below, then press <b>Browse Files</b> — or jump straight to the project file.</span>"
        "</div>", unsafe_allow_html=True)
    up = st.file_uploader("Upload Auto MPG Dataset — drag & drop supported above", type=["csv", "data", "txt"], help="UCI Auto MPG file. '?' in horsepower is treated as missing.")
    fmt = st.selectbox("File format", ["auto", "csv", "data"], index=0, help="auto detects whitespace .data vs CSV.")
    c = st.columns(2)
    if c[0].button("Browse Files - Load uploaded", type="primary", use_container_width=True):
        if up is None: st.error("Choose a file first.")
        else:
            try:
                df, _d = load_auto_mpg(up, fmt)
                if load_df_into_state(df):
                    st.session_state["flash"] = f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols."
                    st.rerun()
            except Exception as e: st.error(f"Could not load dataset: {e}")
    if c[1].button("Use Project Dataset", use_container_width=True):
        pick = "auto-mpg.data" if os.path.exists("auto-mpg.data") else None
        if pick is None: st.error("Project dataset file not found.")
        else:
            try:
                df, _d = load_auto_mpg(pick, "auto")
                if load_df_into_state(df):
                    st.session_state["flash"] = "Project dataset loaded."
                    st.rerun()
            except Exception as e: st.error(f"Could not load dataset: {e}")
    if not HR():
        empty_state("DATASET NOT READY", "Upload a valid Auto MPG dataset before continuing.", "Use the upload zone or Use Project Dataset.")
        return
    df = st.session_state["df_raw"]
    mb = int(pd.to_numeric(df["horsepower"], errors="coerce").isna().sum())
    dups = int(df.duplicated().sum())
    cells = int(df.size)
    miss_cells = int(df.isna().sum().sum())
    completeness = 100 * (1 - miss_cells / max(cells, 1))
    kpi_row([("ROWS", f"{df.shape[0]}", "vehicles", False), ("COLUMNS", f"{df.shape[1]}", "attributes", False),
             ("MISSING HP", f"{mb}", "horsepower ?", True), ("MISSING CELLS", f"{miss_cells}", "all columns", False),
             ("DUPLICATES", f"{dups}", "duplicate rows", True)])
    st.markdown(f"<div class='darkpanel'><b>Dataset validated</b> - {df.shape[0]} records - {df.shape[1]} fields - Horsepower missing: {mb} - Duplicate rows: {dups}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='darkpanel'><b>DATA QUALITY</b> - Completeness <b>{completeness:.2f}%</b> - Schema <b>Valid</b> - Duplicates <b>{dups}</b> - Horsepower issues <b>{mb}</b><br><span style='color:#94A3B8;font-size:12px'>Completeness = 1 - missing cells / total cells, computed live from the loaded frame.</span></div>", unsafe_allow_html=True)
    st.subheader("Premium dataset preview")
    q = st.text_input("Search car name", "", placeholder="e.g. honda, ford, toyota")
    view = df[df["car_name"].str.contains(q, case=False, na=False)] if q else df
    st.caption(f"Showing {len(view)} of {len(df)} rows")
    st.dataframe(view, use_container_width=True, height=380)
    mc = df.isna().sum()
    fig = go.Figure(go.Bar(x=list(mc.index), y=mc.values, marker_color="#D4A72C", hovertemplate="%{x}: %{y} missing<extra></extra>"))
    st.plotly_chart(dark(fig, 320, "Missing values by column"), use_container_width=True)
    with st.expander("Technical detail - dtypes, memory, schema"):
        st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": mc}), use_container_width=True)
        buf = io.StringIO(); df.info(buf=buf)
        st.code(buf.getvalue(), language="text")
def pg_pre():
    UI.render_crumbs("Automotive Fleet", "Preprocessing")
    header("STEP 02-03 - PREPROCESS", "Preprocessing Pipeline", "Cylinder-group median imputation, then Z-score standardization.")
    if not HR():
        empty_state("DATASET NOT READY", "Upload a valid Auto MPG dataset before continuing.", "Go to Dataset first.")
        return
    s = st.session_state
    tmp = clean_horsepower(s["df_raw"])
    mb = int(tmp["horsepower"].isna().sum())
    if s["miss_before"] is None: s["miss_before"] = mb
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='darkpanel'><b>01 HORSEPOWER CLEANING</b><br><span class='pill ok'>READY</span><br><span style='font-size:13px;color:#CBD5E1'>'?' coerced to NaN. {mb} missing detected.</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='darkpanel'><b>02 CYLINDER-GROUP IMPUTATION</b><br><span class='pill {'ok' if HC() else 'warn'}'>{'COMPLETE' if HC() else 'READY'}</span><br><span style='font-size:13px;color:#CBD5E1'>Median HP within each cylinder group.</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='darkpanel'><b>03 Z-SCORE STANDARDIZATION</b><br><span class='pill {'ok' if HS() else 'warn'}'>{'COMPLETE' if HS() else 'PENDING'}</span><br><span style='font-size:13px;color:#CBD5E1'>z = (x - mu) / sigma on 5 features.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='darkpanel' style='text-align:center'><b>RAW &darr; HORSEPOWER CLEANED &darr; CYLINDER MEDIAN IMPUTATION &darr; STANDARDIZATION &darr; READY FOR CLUSTERING</b><br><span style='color:#94A3B8;font-size:12px'>Record counts are live: raw rows are preserved end-to-end (no rows dropped).</span></div>", unsafe_allow_html=True)
    st.subheader("Before -> After: horsepower imputation")
    ma_now = int(s["df_clean"]["horsepower"].isna().sum()) if HC() else mb
    a, b = st.columns(2)
    a.markdown(f"<div class='kpi'><div class='l'>BEFORE</div><div class='v'>{mb} Missing HP</div><div class='s'>raw horsepower '?' values</div></div>", unsafe_allow_html=True)
    b.markdown(f"<div class='kpi teal'><div class='l'>AFTER</div><div class='v'>{ma_now} Missing HP</div><div class='s'>after cylinder-group median</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#D4A72C;font-weight:800'>CYLINDER-GROUP MEDIAN IMPUTATION</div>", unsafe_allow_html=True)
    med = tmp.groupby("cylinders")["horsepower"].median().sort_index()
    miss = tmp.groupby("cylinders")["horsepower"].apply(lambda x: int(x.isna().sum()))
    live = pd.DataFrame({"Median HP": med.round(2), "Missing": miss})
    st.dataframe(live, use_container_width=True)
    if st.session_state.get("viva_mode"):
        UI.viva_box("Horsepower is imputed within each cylinder group because engines with the same cylinder count share a comparable power band. A global median would drag small cars up and big cars down; the grouped median preserves the engineering structure before scaling.")
def pg_pre_b():
    s = st.session_state
    if not HR():
        return
    if st.button("Apply Cylinder-Group Median Imputation", type="primary", use_container_width=True):
        out, medians, mb2, ma2, fb = impute_horsepower_by_cylinder(s["df_raw"])
        s.update(df_clean=out, medians=medians, miss_before=mb2, miss_after=ma2, imp_done=True)
        invalidate_downstream()
        st.success(f"Horsepower imputed: {mb2} -> {ma2}." + (" Fallback used." if fb else ""))
        st.rerun()
    if HC():
        st.progress(1.0 if int(s["df_clean"]["horsepower"].isna().sum()) == 0 else 0.5)
        with st.expander("Show imputed vehicles (actual affected rows)"):
            raw_hp = pd.to_numeric(s["df_raw"]["horsepower"], errors="coerce")
            mask = raw_hp.isna()
            if int(mask.sum()) and (s["medians"] is not None):
                rows = s["df_clean"].loc[mask, ["car_name", "cylinders", "horsepower"]].copy()
                rows["original_horsepower"] = "missing"
                rows["group_median_used"] = rows["cylinders"].map(
                    lambda c: round(float(s["medians"].get(c, float('nan'))), 2))
                rows = rows.rename(columns={"horsepower": "imputed_horsepower"})
                st.dataframe(rows.reset_index(drop=True), use_container_width=True)
            else:
                st.caption("No missing-horsepower rows in the current dataset.")
    st.subheader("Standardization: original units -> comparable scale")
    st.markdown("<div class='darkpanel'><b>Raw features live on different scales.</b> Weight (thousands) - Horsepower (hundreds) - MPG (tens).<br><b style='color:#D4A72C'>z = (x - mu) / sigma</b><br><b>After scaling:</b> Mean ~= 0, Population STD ~= 1, so no single feature dominates distance.</div>", unsafe_allow_html=True)
    if st.button("Standardize Features", type="primary", use_container_width=True):
        if not HC(): st.warning("Apply horsepower imputation first.")
        else:
            std, scaler = standardize_features(s["df_clean"])
            s.update(df_std=std, scaler=scaler, std_done=True)
            for k in ["kmeans", "ward", "krange", "dendro", "dlab", "dk", "pca",
                      "cent_std", "cent_orig", "arch", "arch_rule",
                      "why", "arch_compare", "cmp_a", "cmp_b", "arch_pick"]:
                s[k] = DEFS[k]
            st.success("Features standardized: mean ~= 0, population std ~= 1.")
            st.rerun()
    base = s["df_clean"] if HC() else s["df_raw"]
    st.write("Before (original units):")
    st.dataframe(base[CONTINUOUS_COLS].describe().T.round(4), use_container_width=True)
    if HS():
        st.write("After (mean ~= 0, population std ~= 1):")
        st.dataframe(standardized_statistics(s["df_std"]).round(4), use_container_width=True)
def insight_text(feat, sk, mean, med):
    tail = "positively skewed, with a long upper tail" if sk > 0.5 else ("negatively skewed" if sk < -0.5 else "roughly symmetric")
    skew_w = "Most vehicles sit below the mean; a few high values stretch the tail." if sk > 0.5 else ("Most vehicles sit above the mean." if sk < -0.5 else "Values spread fairly evenly around the centre.")
    return f"{feat.upper()} is {tail} (skew {sk:.2f}; mean {mean:.2f}, median {med:.2f}). {skew_w}"
def pg_eda():
    UI.render_crumbs("Automotive Fleet", "Explore", "Distributions")
    header("STEP 04 - EXPLORE", "EDA Explorer", "Choose a feature and inspect its distribution.")
    if not HR():
        empty_state("DATASET NOT READY", "Upload a valid Auto MPG dataset before continuing.", "")
        return
    df = st.session_state["df_clean"] if HC() else st.session_state["df_raw"]
    st.caption("Exploring cleaned data." if HC() else "Exploring raw data (impute on Preprocessing for final values).")
    c1, c2 = st.columns(2)
    feat = c1.selectbox("Feature", ["mpg", "horsepower", "weight"], index=["mpg", "horsepower", "weight"].index(st.session_state["eda_feat"]), help="Distribution focus feature.")
    ctype = c2.selectbox("Visualization", ["Histogram", "Boxplot"], index=0 if st.session_state["eda_viz"] == "Histogram" else 1)
    st.session_state["eda_feat"], st.session_state["eda_viz"] = feat, ctype
    t1, t2, t3 = st.columns(3)
    show_kde = t1.checkbox("Show KDE", value=bool(st.session_state.get("eda_show_kde", True)))
    show_mean = t2.checkbox("Show Mean", value=bool(st.session_state.get("eda_show_mean", True)))
    show_med = t3.checkbox("Show Median", value=bool(st.session_state.get("eda_show_median", True)))
    st.session_state["eda_show_kde"], st.session_state["eda_show_mean"], st.session_state["eda_show_median"] = show_kde, show_mean, show_med
    x = df[feat].dropna()
    mean, med, mn, mx, sk = float(x.mean()), float(x.median()), float(x.min()), float(x.max()), float(x.skew())
    std = float(x.std())
    kpi_row([("MEAN", f"{mean:.2f}", feat, False), ("MEDIAN", f"{med:.2f}", feat, False), ("STD", f"{std:.2f}", "spread", True), ("SKEWNESS", f"{sk:.2f}", "shape", True)])
    left, right = st.columns([3, 1])
    with left:
        if ctype == "Histogram":
            hist, edges = np.histogram(x, bins=24)
            ctr = (edges[:-1] + edges[1:]) / 2
            try:
                from scipy.stats import gaussian_kde
                kd = gaussian_kde(np.asarray(x))(ctr) * (len(x) * (edges[1] - edges[0]))
            except Exception: kd = np.zeros_like(ctr)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[round(float(v), 2) for v in ctr], y=hist, name="count", marker_color="#2A9D8F"))
            if show_kde:
                fig.add_trace(go.Scatter(x=ctr, y=kd, mode="lines", name="KDE", line=dict(color="#D4A72C", width=3)))
            if show_mean:
                fig.add_vline(x=mean, line_dash="dash", line_color="#E0483C", annotation_text="mean")
            if show_med:
                fig.add_vline(x=med, line_dash="dot", line_color="#3FB950", annotation_text="median")
            st.plotly_chart(dark(fig, 380, f"Distribution of {feat.upper()}"), use_container_width=True)
        else:
            fig = px.box(df, y=feat, color_discrete_sequence=["#2A9D8F"])
            st.plotly_chart(dark(fig, 380, f"Boxplot of {feat.upper()}"), use_container_width=True)
    with right:
        st.markdown(f"<div class='darkpanel'><b>INSIGHT</b><br><span style='font-size:13px'>{insight_text(feat, sk, mean, med)}</span></div>", unsafe_allow_html=True)
def pg_eda_b():
    if not HR():
        return
    df = st.session_state["df_clean"] if HC() else st.session_state["df_raw"]
    st.subheader("Correlation analysis")
    st.caption("Correlation describes linear association and does not establish causation.")
    corr = calculate_correlations(df)
    fig = px.imshow(corr, text_auto=".4f", title="Weight / Horsepower / MPG correlations", color_continuous_scale="Tealgrn", zmin=-1, zmax=1)
    st.plotly_chart(dark(fig, 380, "Weight / Horsepower / MPG correlations"), use_container_width=True)
    st.dataframe(correlation_table(df), use_container_width=True)
    pairs = [("weight", "mpg"), ("horsepower", "mpg"), ("weight", "horsepower")]
    opts = [f"{a} <-> {b}" for a, b in pairs]
    cur = st.session_state["corr_pair"]
    lbl = cur.replace("|", " <-> ") if "|" in cur else opts[0]
    sel = st.selectbox("Select correlation cell", opts, index=opts.index(lbl) if lbl in opts else 0)
    st.session_state["corr_pair"] = sel.replace(" <-> ", "|")
    a, b = sel.split(" <-> ")
    r = float(df[a].corr(df[b]))
    direction = "Positive" if r > 0 else ("Negative" if r < 0 else "Near zero")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='darkpanel'><b>{a.upper()} ↔ {b.upper()}</b><br><span style='font-size:24px;font-weight:800'>{r:.4f}</span><br><span style='color:#94A3B8'>Correlation</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='darkpanel'><b>Direction</b><br><span style='font-size:24px;font-weight:800'>{direction}</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='darkpanel'><b>Strength</b><br><span style='font-size:24px;font-weight:800'>{strength_label(abs(r)).title()}</span><br><span style='color:#94A3B8;font-size:12px'>Association only — not causation.</span></div>", unsafe_allow_html=True)
    st.subheader("Relationship explorer")
    feats = ["weight", "horsepower", "mpg", "displacement", "acceleration"]
    d1, d2, d3 = st.columns([2, 2, 1])
    if d3.button("Swap Axes", use_container_width=True):
        cur_x = st.session_state.get("rel_x", "weight")
        cur_y = st.session_state.get("rel_y", "mpg")
        st.session_state["rel_x"], st.session_state["rel_y"] = cur_y, cur_x
        st.rerun()
    cur_x = st.session_state.get("rel_x", "weight")
    cur_y = st.session_state.get("rel_y", "mpg")
    idx_x = feats.index(cur_x) if cur_x in feats else 0
    idx_y = feats.index(cur_y) if cur_y in feats else 1
    xa = d1.selectbox("X", feats, index=idx_x)
    ya = d2.selectbox("Y", feats, index=idx_y)
    st.session_state["rel_x"], st.session_state["rel_y"] = xa, ya
    hov = df["car_name"] if "car_name" in df.columns else df.index.astype(str)
    fig2 = px.scatter(df, x=xa, y=ya, hover_name=hov, hover_data=["mpg", "horsepower", "weight", "cylinders", "displacement"], color_discrete_sequence=["#2A9D8F"])
    fig2.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color="#0F172A")))
    st.plotly_chart(dark(fig2, 420, f"{xa} vs {ya}"), use_container_width=True)
def cscat(df, labels, k, xt="weight", yt="mpg", show_centroids=False, centers_orig=None):
    d = df.copy()
    d["cluster"] = [f"Cluster {int(v)}" for v in labels]
    fig = px.scatter(d, x=xt, y=yt, color="cluster", hover_name="car_name",
                     hover_data=["mpg", "horsepower", "weight", "cylinders"],
                     title=f"Fleet clusters in {xt} vs {yt} (K={k})", color_discrete_sequence=CCOLS)
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="#0F172A")))
    if show_centroids and centers_orig is not None:
        try:
            co = centers_orig.set_index("cluster") if "cluster" in centers_orig.columns else centers_orig
            fig.add_trace(go.Scatter(x=[float(co.loc[i, xt]) for i in range(k)],
                                     y=[float(co.loc[i, yt]) for i in range(k)],
                                     mode="markers", marker=dict(size=15, symbol="x", color="#F8FAFC",
                                                                 line=dict(width=2, color="#0F172A")),
                                     name="centroids", hovertemplate="centroid %{text}<extra></extra>",
                                     text=[f"Cluster {i}" for i in range(k)]))
        except Exception:
            pass
    return dark(fig, 440, f"Fleet clusters in {xt} vs {yt} (K={k})")
def donut(counts, k):
    tot = int(counts.sum())
    fig = go.Figure(go.Pie(labels=[f"Cluster {i}" for i in counts.index], values=counts.values, hole=0.55,
                           marker=dict(colors=CCOLS), hovertemplate="%{label}<br>%{value} vehicles<br>%{percent}<extra></extra>"))
    fig.add_annotation(text=f"K={k}<br>{tot} cars", showarrow=False, font=dict(size=14, color="#F8FAFC"))
    return dark(fig, 340, f"Cluster distribution (K={k})")
def pg_cluster_a():
    header("STEP 05 - CLUSTER", "Clustering Lab", "Explore the effect of K and compare clustering strategies.")
    if not need_std():
        return False, DEFAULT_RANDOM_STATE, DEFAULT_N_INIT
    s = st.session_state
    X = XS()
    st.subheader("Method selector")
    m1, m2 = st.columns(2)
    km_sel = s["method"] == "K-Means"
    w_sel = s["method"] != "K-Means"
    st.markdown(
        f"<div class='mcard {'sel' if km_sel else ''}'><h4>{'●' if km_sel else '○'} K-MEANS</h4>"
        f"<p>Partition-based clustering</p></div>", unsafe_allow_html=True)
    if m1.button("Use K-Means", use_container_width=True, key="use_km",
                 type="primary" if km_sel else "secondary"):
        if s["method"] != "K-Means":
            s["method"] = "K-Means"
            # Keep both trained models for the comparison views; archetype
            # artefacts auto-recompute from K-Means on next render.
            for kk in ["dlab", "dk", "dendro"]:
                s[kk] = DEFS.get(kk, None)
        st.rerun()
    st.markdown(
        f"<div class='mcard {'sel' if w_sel else ''}'><h4>{'●' if w_sel else '○'} WARD HIERARCHICAL</h4>"
        f"<p>Agglomerative clustering</p></div>", unsafe_allow_html=True)
    if m2.button("Use Ward Hierarchical", use_container_width=True, key="use_w",
                 type="primary" if w_sel else "secondary"):
        if s["method"] == "K-Means":
            s["method"] = "Hierarchical - Ward"
            # Symmetric: lens switch never destroys trained models; only
            # dendrogram artefacts (built from Ward linkage) reset.
            for kk in ["dlab", "dk", "dendro"]:
                s[kk] = DEFS.get(kk, None)
        st.rerun()
    st.markdown(f"<div class='darkpanel'>Active algorithm: <b>{s['method']}</b></div>", unsafe_allow_html=True)
    UI.render_method_card("K-MEANS", "Partition-based clustering.", km_sel, "random_state=42, n_init=10 (assignment defaults). Deterministic configuration enabled.")
    UI.render_method_card("WARD HIERARCHICAL", "Agglomerative clustering with Ward linkage.", w_sel, "Linkage fixed by the assignment; cut extracted with fcluster(maxclust).")
    st.subheader("Number of clusters")
    newk = st.slider("K (2 to 6) — NUMBER OF CLUSTERS", 2, 6, s["selected_k"], help="K range fixed 2-6 by the assignment.")
    st.caption(f"Current K: {s['selected_k']}")
    if newk != s["selected_k"]:
        s["selected_k"] = newk
        for kk in ["kmeans", "ward", "dlab", "dk", "pca", "krange", "dendro",
                   "cent_std", "cent_orig", "arch", "arch_rule",
                   "why", "arch_compare", "cmp_a", "cmp_b", "arch_pick"]:
            s[kk] = DEFS.get(kk, None)
        s["cmp_b"] = min(1, newk - 1)
        st.rerun()
    st.markdown(
        "<div class='darkpanel'><b>Changing K affects:</b> cluster assignments, inertia, silhouette, Davies-Bouldin, "
        "cluster sizes, PCA coloring, archetype grouping, exports. Cluster numbers are identifiers — Cluster 0 at K=2 "
        "is not the same population as Cluster 0 at K=3.</div>", unsafe_allow_html=True)
    with st.expander("Advanced Settings"):
        rs = st.number_input("Random State", 0, 9999, DEFAULT_RANDOM_STATE, help="K-Means random_state (assignment default 42).")
        ni = st.number_input("Initializations (n_init)", 1, 50, DEFAULT_N_INIT, help="K-Means n_init (assignment default 10).")
        st.info("Ward linkage is fixed (required by the assignment). Defaults: random_state=42, n_init=10.")
    return True, int(rs), int(ni)
def do_run_clustering(rs, ni):
    s = st.session_state
    X = XS()
    prog = st.progress(0)
    msg = st.empty()
    msg.info("Preparing standardized feature matrix")
    prog.progress(15)
    if s["method"] == "K-Means":
        msg.info("Training clustering model (K-Means, random_state=42, n_init=10)")
        r = evaluate_kmeans(X, s["selected_k"], rs, ni)
        prog.progress(55)
        msg.info("Calculating metrics (silhouette, Davies-Bouldin)")
        s["kmeans"] = r
        s["krange"] = ckrange(X, s["dataset_id"], rs, ni)
        prog.progress(75)
        msg.info("Preparing visualizations (PCA + centroids + archetypes)")
        s["pca"] = cpca(X, s["dataset_id"])
        cs, co = calculate_centroids(s["scaler"], r["centers"])
        s["cent_std"], s["cent_orig"] = cs, co
        s["arch"], s["arch_rule"] = assign_archetypes(co)
    else:
        msg.info("Training clustering model (Ward agglomerative)")
        r = fit_ward(X, s["selected_k"])
        prog.progress(55)
        msg.info("Calculating metrics (silhouette, Davies-Bouldin)")
        s["ward"] = r
        if s["krange"] is None:
            s["krange"] = ckrange(X, s["dataset_id"], DEFAULT_RANDOM_STATE, DEFAULT_N_INIT)
        prog.progress(75)
        msg.info("Preparing visualizations (PCA + dendrogram)")
        s["pca"] = cpca(X, s["dataset_id"])
        if s["dendro"] is None:
            s["dendro"] = build_dendrogram(X)
    prog.progress(100)
    msg.success("Complete")
def pg_cluster_run(rs, ni):
    s = st.session_state
    if st.button("RUN CLUSTERING", type="primary", use_container_width=True, help="Trains the active algorithm at current K."):
        do_run_clustering(rs, ni)
        st.rerun()
    km, w = s["kmeans"], s["ward"]
    if km is None and w is None:
        empty_state("NO MODEL TRAINED", "Press RUN CLUSTERING to train the active algorithm.", "K-Means and Ward results appear here.")
        return
    active = km if s["method"] == "K-Means" else w
    if active is not None:
        st.markdown(f"<div class='darkpanel'><b>CLUSTERING COMPLETE</b> - Method: <b>{s['method']}</b> - K: <b>{active['k']}</b> - Silhouette: <b>{active['silhouette']:.4f}</b> - Davies-Bouldin: <b>{active['db']:.4f}</b></div>", unsafe_allow_html=True)
    if km is not None:
        st.subheader("K-Means result")
        kpi_row([("K", f"{km['k']}", f"inertia {km['inertia']:.2f}", False), ("SILHOUETTE", f"{km['silhouette']:.4f}", "higher is better", True), ("DAVIES-BOULDIN", f"{km['db']:.4f}", "lower is better", False), ("LARGEST GROUP", f"{int(km['counts'].max())}", f"of {len(km['labels'])}", True)])
        st.plotly_chart(donut(km["counts"], km["k"]), use_container_width=True)
        view = st.selectbox("Cluster view", ["Weight vs MPG (weight / mpg)", "Horsepower vs MPG (horsepower / mpg)", "Displacement vs MPG (displacement / mpg)"], key="km_view")
        xt = view.split("(")[1].split("/")[0].strip()
        show_c = st.checkbox("Show centroid markers (inverse-transformed)", value=False, key="km_cent")
        st.plotly_chart(cscat(s["df_clean"], km["labels"], km["k"], xt=xt, show_centroids=show_c, centers_orig=s["cent_orig"]), use_container_width=True)
    if w is not None:
        st.subheader("Ward hierarchical result")
        kpi_row([("K", f"{w['k']}", "linkage: ward", False), ("SILHOUETTE", f"{w['silhouette']:.4f}", "higher is better", True), ("DAVIES-BOULDIN", f"{w['db']:.4f}", "lower is better", False), ("LARGEST GROUP", f"{int(w['counts'].max())}", f"of {len(w['labels'])}", True)])
        st.plotly_chart(donut(w["counts"], w["k"]), use_container_width=True)
        st.plotly_chart(cscat(s["df_clean"], w["labels"], w["k"]), use_container_width=True)
    if s["cent_orig"] is not None and HK():
        st.subheader("Compare clusters (original units)")
        co = s["cent_orig"].set_index("cluster")
        cpair = st.columns(2)
        k_opts = list(range(km["k"]))
        # Clamp BEFORE the selectboxes instantiate: a stale pick from a
        # previous K (e.g. 2 at K=3 -> retrain at K=2) is out of range, and
        # assigning/deleting a widget key after instantiation raises.
        if s.get("cmp_a") not in k_opts:
            s["cmp_a"] = 0
        if s.get("cmp_b") not in k_opts:
            s["cmp_b"] = min(1, km["k"] - 1)
        ai = cpair[0].selectbox("Cluster A", k_opts, key="cmp_a")
        bi = cpair[1].selectbox("Cluster B", k_opts, key="cmp_b")
        if ai is None or ai not in co.index:
            ai = 0
        if bi is None or bi not in co.index:
            bi = min(1, km["k"] - 1)
        for m in ["mpg", "weight", "horsepower", "displacement", "acceleration"]:
            st.write(f"{m}: A {co.loc[ai, m]:.2f} vs B {co.loc[bi, m]:.2f} (Δ {co.loc[ai, m] - co.loc[bi, m]:+.2f})")
def pg_cluster_b():
    s = st.session_state
    X = XS()
    st.subheader("Elbow explorer (K-Means inertia, K=2-6)")
    st.caption("Inertia: total within-cluster squared distance. It always falls as K rises, so read the bend (elbow), not the minimum.")
    if st.button("Compute Elbow + Silhouette (K=2-6)"):
        s["krange"] = ckrange(X, s["dataset_id"], DEFAULT_RANDOM_STATE, DEFAULT_N_INIT)
        st.rerun()
    if s["krange"] is None:
        empty_state("ELBOW NOT COMPUTED", "Compute the K=2-6 sweep to see elbow and silhouette curves.", "")
        return
    kr = s["krange"]
    def _elbow_fig(n_points):
        sub = kr.iloc[:n_points]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["K"], y=sub["inertia"], mode="lines+markers", name="inertia", marker_color="#D4A72C", line=dict(color="#D4A72C", width=3), hovertemplate="K=%{x}<br>inertia=%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[s["selected_k"]], y=[float(kr.set_index("K").loc[s["selected_k"], "inertia"])], mode="markers", marker=dict(size=14, color="#E0483C", symbol="star"), name="selected K"))
        fig.update_layout(xaxis=dict(tickmode="linear", dtick=1, range=[1.8, 6.2]))
        return dark(fig, 360, "Elbow: inertia vs K")
    rc1, rc2 = st.columns([4, 1])
    # Non-blocking reveal: slider re-renders instantly (no time.sleep fake delay).
    step = rc1.slider("Reveal K=2 -> K=6", 1, len(kr), len(kr), help="Drag to reveal K=2..6 one point at a time. Single pass only.")
    st.plotly_chart(_elbow_fig(step), use_container_width=True)
    rc2.caption("Tip: drag the slider left, then right for a one-pass K=2 → 6 reveal.")
    st.subheader("Silhouette explorer (K=2-6)")
    best = kr.loc[kr["silhouette"].idxmax()]
    sub = kr.iloc[:step]
    fig2 = go.Figure(go.Bar(x=sub["K"].astype(str), y=sub["silhouette"], marker_color=["#E0483C" if k == best["K"] else "#2A9D8F" for k in sub["K"]], hovertemplate="K=%{x}<br>silhouette=%{y:.4f}<extra></extra>"))
    st.plotly_chart(dark(fig2, 360, f"Silhouette by K (highest: K={int(best['K'])}, engineering selection: K=3)"), use_container_width=True)
    st.dataframe(kr.round(4), use_container_width=True)
    sk, sv = optimal_k_from_silhouette(kr)
    ek, _ = elbow_suggestion(kr)
    st.markdown(f"<div class='darkpanel'><b>WHY K = 3?</b> The silhouette score peaks at <b>K={sk}</b> ({sv:.4f}). The inertia curve shows a meaningful bend around <b>K={ek}</b>. K=3 provides finer segmentation for fleet profiling. <b>Decision: K=3 is a joint engineering choice, not a unique mathematical optimum.</b></div>", unsafe_allow_html=True)
    cur = float(kr.set_index('K').loc[s['selected_k'], 'silhouette'])
    UI.render_takeaway(f"The selected K={s['selected_k']} solution (silhouette {cur:.4f}) trades a small gap to the K={sk} peak ({sv:.4f}) for a more useful fleet segmentation. Change K above to see every downstream result update.")
def pg_cluster_c():
    s = st.session_state
    X = XS()
    st.subheader("Ward dendrogram")
    st.caption("Ward linkage on all observations. Visual truncation shows 30 terminal groups only — the linkage itself is never truncated.")
    detail = st.checkbox("Detailed View (more branches)",
                        value=bool(s.get("dendro_detail", False)), key="dendro_detail")
    if st.button("Build Ward dendrogram"):
        s["dendro"] = build_dendrogram(X)
        # One-click path to the exportable artefact: linkage alone leaves the
        # Dendrogram export empty, so extract current-K assignments now too.
        lab = extract_dendrogram_clusters(s["dendro"]["Z_full"], s["selected_k"])
        s["dlab"], s["dk"] = lab, s["selected_k"]
        st.rerun()
    if s["dendro"] is None:
        empty_state("NO DENDROGRAM", "Build the Ward dendrogram to see the merge hierarchy.", "")
    else:
        from scipy.cluster.hierarchy import dendrogram as scd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Z = s["dendro"]["Z_show"]
        cut = dendro_cut_height(Z, s["selected_k"])
        p_level = 60 if bool(st.session_state.get("dendro_detail", False)) else 30
        figm, ax = plt.subplots(figsize=(10, 4))
        figm.patch.set_facecolor("#0F172A"); ax.set_facecolor("#0F172A")
        scd(Z, ax=ax, no_labels=True, color_threshold=cut, above_threshold_color="#94A3B8", p=p_level, truncate_mode="level")
        ax.axhline(cut, color="#D4A72C", linestyle="--", label=f"cut -> K={s['selected_k']}")
        ax.set_title(f"Ward dendrogram (display sample, cut -> K={s['selected_k']})", color="white")
        ax.set_xlabel("Sampled vehicle", color="white"); ax.set_ylabel("Ward distance", color="white")
        ax.tick_params(colors="white"); ax.legend(facecolor="#1E293B", labelcolor="white")
        st.pyplot(figm, use_container_width=True)
        st.caption("Ward linkage computed using all observations. Displayed dendrogram is visually truncated for readability.")
        if st.button("Extract Cluster Assignments (fcluster, maxclust)"):
            lab = extract_dendrogram_clusters(s["dendro"]["Z_full"], s["selected_k"])
            s["dlab"], s["dk"] = lab, s["selected_k"]
            sil, db = evaluate_clustering(X, lab)
            st.success(f"Dendrogram assignments: K={s['selected_k']}, silhouette={sil:.4f}, DB={db:.4f}.")
            st.rerun()
    st.subheader("Cluster assignment table")
    dfc = s["df_clean"].copy()
    if s["kmeans"] is not None: dfc["K-Means Cluster"] = s["kmeans"]["labels"]
    if s["ward"] is not None: dfc["Ward Cluster"] = s["ward"]["labels"]
    if s["dlab"] is not None: dfc["Dendrogram Cluster"] = s["dlab"]
    labcols = [c for c in ["K-Means Cluster", "Ward Cluster", "Dendrogram Cluster"] if c in dfc.columns]
    if labcols:
        f = st.selectbox("Cluster Filter", ["All"] + sorted(dfc[labcols[0]].unique().tolist()))
        show = dfc if f == "All" else dfc[dfc[labcols[0]] == f]
        if f != "All":
            st.caption(f"Showing filtered subset ({len(show)} of {len(dfc)}). Global model metrics above still describe the full K={s['selected_k']} solution.")
        st.dataframe(show[["car_name", "mpg", "horsepower", "weight"] + labcols], use_container_width=True, height=360)
    else:
        st.info("Run clustering to populate assignments.")
def pg_cluster():
    ok, rs, ni = pg_cluster_a()
    if not ok:
        return
    pg_cluster_run(rs, ni)
    pg_cluster_b()
    pg_cluster_c()
def pg_eval():
    UI.render_crumbs("Automotive Fleet", "Clustering", "Evaluation")
    header("STEP 06 - EVALUATE", "Cluster Quality Evaluation", "How coherent and separated are the discovered groups?")
    st.caption("Silhouette: HIGHER ↑ BETTER. Davies-Bouldin: LOWER ↓ BETTER. No single overall score is declared — both lenses matter.")
    if not need_std():
        return
    s = st.session_state
    if not (HK() or HW()):
        empty_state("NO MODEL TRAINED", "Run clustering on Clustering Lab first.", "")
        return
    km, w = s["kmeans"], s["ward"]
    if km is not None and w is not None:
        sl = "K-Means" if km["silhouette"] >= w["silhouette"] else "Ward"
        dl = "K-Means" if km["db"] <= w["db"] else "Ward"
        sgap = abs(km["silhouette"] - w["silhouette"])
        dgap = abs(km["db"] - w["db"])
        st.markdown(f"<div class='darkpanel'><b>CLUSTER QUALITY</b> — Silhouette leader: <b>{sl}</b> (gap {sgap:.4f}) | Davies-Bouldin leader: <b>{dl}</b> (gap {dgap:.4f}). <b>Overall winner: no single winner</b> — the methods are relatively close and reward different structure.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("K-Means")
        if km is not None: kpi_row([("SILHOUETTE", f"{km['silhouette']:.4f}", "higher better", True), ("DAVIES-BOULDIN", f"{km['db']:.4f}", "lower better", False)])
        else: st.info("Not run yet.")
    with c2:
        st.subheader("Ward")
        if w is not None: kpi_row([("SILHOUETTE", f"{w['silhouette']:.4f}", "higher better", True), ("DAVIES-BOULDIN", f"{w['db']:.4f}", "lower better", False)])
        else: st.info("Not run yet.")
    st.subheader("Metric comparison")
    cur_m = s.get("eval_metric", "Silhouette")
    m = st.radio("Metric", ["Silhouette", "Davies-Bouldin"], horizontal=True,
                 index=0 if cur_m == "Silhouette" else 1, key="eval_metric")
    rows = []
    if km is not None: rows.append(("K-Means", km["silhouette"], km["db"]))
    if w is not None: rows.append(("Ward", w["silhouette"], w["db"]))
    if rows:
        vals = [r[1] if m == "Silhouette" else r[2] for r in rows]
        fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=vals, marker_color=["#D4A72C", "#2A9D8F"], hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
        st.plotly_chart(dark(fig, 340, f"{m} comparison ({'higher better' if m=='Silhouette' else 'lower better'})"), use_container_width=True)
def pg_pca():
    UI.render_crumbs("Automotive Fleet", "Clustering", "PCA")
    header("STEP 07 - VISUALIZE", "PCA Cluster Explorer", "From 5D feature space to 2D visual interpretation.")
    st.markdown("<div class='darkpanel'><b>PCA is used for visualization only. Clustering is performed in the five-dimensional standardized feature space.</b></div>", unsafe_allow_html=True)
    if not need_std():
        return
    s = st.session_state
    if not (HK() or HW()):
        empty_state("NO MODEL TRAINED", "Run clustering first so PCA points can be colored by cluster.", "")
        return
    X = XS()
    if s["pca"] is None:
        s["pca"] = cpca(X, s["dataset_id"])
    p = s["pca"]
    v = p["variance_ratio"]
    kpi_row([("PC1 VARIANCE", f"{100*v[0]:.2f}%", "first component", False), ("PC2 VARIANCE", f"{100*v[1]:.2f}%", "second component", False), ("COMBINED", f"{100*float(np.sum(v)):.2f}%", "2D projection", True)])
    trained = [m for m in ("K-Means", "Ward")
               if (m == "K-Means" and HK()) or (m == "Ward" and HW())]
    if not trained:
        empty_state("NO MODEL TRAINED", "Run clustering first so PCA points can be colored by cluster.", "")
        return
    prev = s.get("pca_color", "K-Means")
    mt = st.radio("Color by", trained, horizontal=True,
                  index=trained.index(prev) if prev in trained else 0, key="pca_color")
    c1, c2 = st.columns(2)
    show_cent = c1.checkbox("Show centroids", value=bool(s.get("pca_centroids", True)),
                            key="pca_centroids")
    show_lab = c2.checkbox(f"Show vehicle labels ({len(X)} points — off by default)",
                           value=bool(s.get("pca_labels", False)), key="pca_labels")
    psize = st.slider("Point size", 3, 14, int(s.get("pca_size", 7)), key="pca_size")
    lab = s["kmeans"]["labels"] if (mt == "K-Means" and HK()) else (s["ward"]["labels"] if HW() else None)
    if lab is None:
        empty_state("NO CLUSTER LABELS", "Selected clustering method labels are not available.", "")
        return
    d = s["df_clean"].copy()
    d["PC1"], d["PC2"] = p["coords"][:, 0], p["coords"][:, 1]
    d["cluster"] = [f"Cluster {int(x)}" for x in lab]
    ttl = "K-Means Vehicle Clusters in 2D PCA Space" if mt == "K-Means" else "Ward Hierarchical Vehicle Clusters in 2D PCA Space"
    fig = px.scatter(d, x="PC1", y="PC2", color="cluster", hover_name="car_name", hover_data=["mpg", "horsepower", "weight", "cylinders"], title=ttl, color_discrete_sequence=CCOLS)
    fig.update_traces(marker=dict(size=int(psize), opacity=0.85, line=dict(width=0.5, color="#0F172A")), text=d["car_name"] if show_lab else None, mode="markers+text" if show_lab else "markers")
    xr = [float(p["coords"][:, 0].min()), float(p["coords"][:, 0].max())]
    yr = [float(p["coords"][:, 1].min()), float(p["coords"][:, 1].max())]
    padx = (xr[1] - xr[0]) * 0.05 + 0.5
    pady = (yr[1] - yr[0]) * 0.05 + 0.5
    fig.update_layout(xaxis=dict(range=[xr[0] - padx, xr[1] + padx]), yaxis=dict(range=[yr[0] - pady, yr[1] + pady]))
    st.plotly_chart(dark(fig, 460, ttl), use_container_width=True)
    st.caption("Axis ranges are fixed from the full PCA projection so method switches do not cause misleading jumps.")
    st.markdown(f"<div class='darkpanel'><b>PC1 {'█' * 22} {100*v[0]:.2f}%</b> &nbsp; <b>PC2 {'█' * 5} {100*v[1]:.2f}%</b><br><span style='color:#CBD5E1'>The first two components explain <b>{100*float(np.sum(v)):.2f}%</b> of total standardized variance, making the 2D projection highly informative for visual inspection. Clustering itself remains in 5D.</span></div>", unsafe_allow_html=True)
def pg_arch():
    UI.render_crumbs("Automotive Fleet", "Archetypes")
    header("STEP 08 - INTERPRET", "Fleet Archetypes", "Data-driven vehicle profiles discovered by clustering.")
    if not need_std():
        return
    s = st.session_state
    if not HK() or s["cent_orig"] is None or s["arch"] is None:
        empty_state("NO MODEL TRAINED", "Run K-Means clustering first (archetypes use K-Means centroids).", "")
        return
    km = s["kmeans"]
    co = s["cent_orig"]
    names = s["arch"]
    st.info(s["arch_rule"])
    overall = s["df_clean"][CONTINUOUS_COLS].mean()
    cols = st.columns(km["k"])
    cmin = co.set_index("cluster")[["mpg", "weight", "horsepower", "displacement"]].min()
    cmax = co.set_index("cluster")[["mpg", "weight", "horsepower", "displacement"]].max()
    span = (cmax - cmin).replace(0, 1)
    for i in range(km["k"]):
        row = co.set_index("cluster").loc[i]
        n = int((km["labels"] == i).sum())
        pct = 100 * n / len(km["labels"])
        prof = ((row[["mpg", "weight", "horsepower", "displacement"]] - cmin) / span)
        bars = "".join(
            f"<div style='font-size:11px;color:#475569'>{m.upper()}</div>"
            f"<div class='probar'><div style='width:{max(4, min(100, 100*float(prof[m])))}%'></div></div>"
            for m in ["mpg", "weight", "horsepower", "displacement"])
        chars = []
        chars.append("High MPG" if row["mpg"] >= overall["mpg"] else "Low MPG")
        chars.append("Low Weight" if row["weight"] <= overall["weight"] else "High Weight")
        chars.append("Low HP" if row["horsepower"] <= overall["horsepower"] else "High HP")
        with cols[i % len(cols)]:
            st.markdown(f"<div class='acard' style='animation-delay:{i*0.08}s'><h3>CLUSTER {i}</h3><h2>{names.get(i, f'Group {i}').upper()}</h2><div class='big'>{row['mpg']:.2f} MPG</div><div class='row'><span>Weight</span><b>{row['weight']:.0f} lbs</b></div><div class='row'><span>Horsepower</span><b>{row['horsepower']:.2f} HP</b></div><div class='row'><span>Displacement</span><b>{row['displacement']:.2f}</b></div><div class='row'><span>Acceleration</span><b>{row['acceleration']:.2f}</b></div><div class='row'><span>Vehicles</span><b>{n} ({pct:.1f}%)</b></div><div style='margin-top:8px;font-size:11px;color:#475569'>RELATIVE PROFILE (visual only)</div>{bars}<div style='font-size:11.5px;margin-top:6px'><b>Profile:</b> {'; '.join(chars)}</div></div>", unsafe_allow_html=True)
    st.subheader("Centroid profile (standardized z vs original units)")
    # Clamp BEFORE instantiation: the pick from a previous K may be out of
    # range after retraining; writing a widget key after it renders raises.
    cur_pick = s.get("arch_pick", 0)
    if not isinstance(cur_pick, int) or cur_pick not in range(km["k"]):
        cur_pick = 0
        s["arch_pick"] = 0
    pick = st.selectbox("Select cluster", list(range(km["k"])), index=cur_pick, key="arch_pick",
                        format_func=lambda i: f"Cluster {i} - {names.get(i, f'Cluster {i}')}")
    if pick is None or pick not in s["cent_std"].set_index("cluster").index:
        pick = 0
    cs = s["cent_std"].set_index("cluster").loc[pick]
    zc = [c for c in s["cent_std"].columns if c != "cluster"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c.replace(" (z)", "").upper() for c in zc], y=[float(cs[c]) for c in zc], mode="lines+markers", fill="toself", line=dict(color="#D4A72C", width=3), name="standardized z"))
    st.plotly_chart(dark(fig, 360, f"Cluster {pick} standardized profile (z-scores)"), use_container_width=True)
    st.subheader("Original-unit centroid values (MPG, displacement, HP, lbs, sec):")
    st.dataframe(co.round(2), use_container_width=True)
    st.subheader("Compare archetypes side by side")
    # K can change between runs; prune stale/invalid indices before the widget instantiates.
    def _idx(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    _sel = [_idx(v) for v in (st.session_state.get("arch_compare") or [])]
    st.session_state["arch_compare"] = [i for i in _sel if i is not None and 0 <= i < km["k"]]
    st.multiselect("Pick two or more archetypes to compare", list(range(km["k"])), key="arch_compare",
                   format_func=lambda i: f"Cluster {i} - {names.get(i, f'Cluster {i}')}",
                   help="Comparison table only — never touches the model.")
    sel = list(st.session_state.get("arch_compare") or [])
    if len(sel) >= 2:
        comp = co.set_index("cluster").loc[sel][["mpg", "displacement", "horsepower", "weight", "acceleration"]].T
        comp.columns = [f"Cluster {i}" for i in sel]
        st.dataframe(comp.round(2), use_container_width=True)
        for i in sel[1:]:
            d = co.set_index("cluster").loc[i] - co.set_index("cluster").loc[sel[0]]
            st.write(f"Cluster {i} vs Cluster {sel[0]}: " + ", ".join(f"{f} {d[f]:+.1f}" for f in ["mpg", "weight", "horsepower"]))
    else:
        st.caption("Select at least two archetypes above to see a side-by-side comparison.")
def pg_arch_b():
    s = st.session_state
    if not HK() or s["arch"] is None or s["cent_orig"] is None:
        return
    km = s["kmeans"]
    names = s["arch"]
    st.subheader("Representative vehicles (nearest 5 to centroid)")
    st.caption("Selected by Euclidean distance to centroid in standardized space - never random.")
    reps = find_representative_vehicles(s["df_clean"], XS(), km["centers"], 5)
    for i in range(km["k"]):
        with st.expander(f"Cluster {i} - {names.get(i, f'Cluster {i}')}", expanded=(i == 0)):
            st.dataframe(reps[i], use_container_width=True)
    st.subheader("Why this cluster?")
    overall = s["df_clean"][CONTINUOUS_COLS].mean()
    co = s["cent_orig"]
    w_opts = list(range(km["k"]))
    # Clamp BEFORE instantiation (same stale-K rationale as arch_pick above).
    cur_why = s.get("why")
    if cur_why is None or not isinstance(cur_why, int) or cur_why not in w_opts:
        s["why"] = 0
    wc = st.selectbox("Explain cluster", w_opts, format_func=lambda i: f"{i} - {names.get(i, f'Cluster {i}')}", key="why")
    if wc is None or wc not in co.set_index("cluster").index:
        wc = 0
    row = co.set_index("cluster").loc[wc]
    for f in CONTINUOUS_COLS:
        pct = 100 * (row[f] - overall[f]) / overall[f]
        st.write(f"{f.upper()}: {row[f]:.2f} vs fleet {overall[f]:.2f} ({pct:+.1f}%)")
def pg_lookup():
    UI.render_crumbs("Automotive Fleet", "Lookup")
    header("INTERPRET", "Vehicle Lookup", "Automotive specification search with cluster explanation.")
    viva = st.toggle("Explain like a viva", value=bool(st.session_state.get("viva_mode", True)), key="viva_mode", help="Concise viva-ready explanations; never alters computation.")
    if not need_std():
        return
    s = st.session_state
    if not (HK() or HW()):
        empty_state("NO MODEL TRAINED", "Run clustering first so vehicles can be mapped to archetypes.", "")
        return
    km = s["kmeans"]
    names_list = s["df_clean"]["car_name"].tolist()
    q = st.text_input("Search vehicle", "", placeholder="type to filter, e.g. accord")
    hits = [n for n in names_list if q.lower() in n.lower()] if q else names_list[:30]
    if not hits:
        st.warning("No vehicles match that search.")
        return
    opts = hits[:80]
    # Clamp BEFORE instantiation: the stored car may be filtered out by the
    # search query or left over from a previous/reset dataset. Falling back
    # keeps selection and profile consistent (the reported Index.get_loc(None)
    # edge was the unkeyed selector drifting out of sync with lookup_car).
    dflt = s.get("lookup_car")
    if dflt not in opts:
        dflt = opts[0]
        s["lookup_car"] = dflt
    sel = st.selectbox("Vehicle", opts, index=opts.index(dflt),
                       key="lookup_car")
    match = s["df_clean"][s["df_clean"]["car_name"] == sel]
    if match.empty:
        st.warning("That vehicle is not in the current dataset — pick another.")
        return
    r = match.iloc[0]
    idx = s["df_clean"][s["df_clean"]["car_name"] == sel].index[0]
    pos = s["df_clean"].index.get_loc(idx)
    kl = int(km["labels"][pos]) if HK() else None
    wl = int(s["ward"]["labels"][pos]) if HW() else "--"
    arch = s["arch"].get(kl, "--") if (HK() and kl is not None and s["arch"] is not None) else "--"
    st.markdown(f"<div class='darkpanel'><b style='font-size:18px'>VEHICLE PROFILE - {sel}</b><br><span style='color:#94A3B8'>K-Means {kl if kl is not None else '--'} | Ward {wl} | Archetype {arch}</span></div>", unsafe_allow_html=True)
    st.markdown(f"""<div class='spec'>
<div><div class='k'>MPG</div><div class='w'>{r['mpg']}</div></div>
<div><div class='k'>CYLINDERS</div><div class='w'>{r['cylinders']}</div></div>
<div><div class='k'>DISPLACEMENT</div><div class='w'>{r['displacement']}</div></div>
<div><div class='k'>HORSEPOWER</div><div class='w'>{r['horsepower']}</div></div>
<div><div class='k'>WEIGHT</div><div class='w'>{r['weight']}</div></div>
<div><div class='k'>ACCELERATION</div><div class='w'>{r['acceleration']}</div></div>
<div><div class='k'>MODEL YEAR</div><div class='w'>{r['model_year']}</div></div>
<div><div class='k'>ORIGIN</div><div class='w'>{r['origin']}</div></div>
</div>""", unsafe_allow_html=True)
    st.subheader("Why this vehicle belongs here")
    overall = s["df_clean"][CONTINUOUS_COLS].mean()
    for f, lbl in [("mpg", "MPG"), ("weight", "Weight"), ("horsepower", "Horsepower")]:
        pct = 100 * (r[f] - overall[f]) / overall[f]
        direction = "above" if pct >= 0 else "below"
        st.write(f"{lbl}: {pct:+.1f}% {direction} fleet average ({r[f]} vs {overall[f]:.1f})")
    if viva and HK() and kl is not None:
        try:
            ctr = s["kmeans"]["centers"]
            X = XS()
            d_all = [float(np.linalg.norm(X[pos] - c)) for c in np.asarray(ctr)]
            order = sorted(range(len(d_all)), key=lambda i: d_all[i])
            UI.viva_box(f"The vehicle is closer to Cluster {kl}'s centroid (distance {d_all[kl]:.3f} in standardized space) than to Cluster {order[1]} ({d_all[order[1]]:.3f}). Assignment is nearest-centroid in the 5D standardized space — not a manual rule.")
        except Exception:
            pass
    st.subheader("Compare vehicles (up to 3 — display only, never affects the model)")
    names_list2 = s["df_clean"]["car_name"].tolist()
    # Keyed widgets keep each column's pick stable across reruns/searches;
    # clamp BEFORE instantiation so a reset dataset cannot leave a stale name.
    cA, cB, cC = st.columns(3)
    vA_idx = names_list2.index(sel) if (sel and sel in names_list2) else 0
    if s.get("lookup_va") not in names_list2:
        s["lookup_va"] = names_list2[vA_idx]
    if s.get("lookup_vb") not in names_list2:
        s["lookup_vb"] = names_list2[min(1, len(names_list2) - 1)]
    if s.get("lookup_vc") not in names_list2:
        s["lookup_vc"] = names_list2[min(2, len(names_list2) - 1)]
    vA = cA.selectbox("Vehicle A", names_list2, key="lookup_va")
    vB = cB.selectbox("Vehicle B", names_list2, key="lookup_vb")
    vC = cC.selectbox("Vehicle C", names_list2, key="lookup_vc")
    rows = []
    for v in [vA, vB, vC]:
        if v not in s["df_clean"]["car_name"].values:
            continue
        m = s["df_clean"][s["df_clean"]["car_name"] == v].iloc[0]
        rows.append(m[["mpg", "horsepower", "weight", "displacement", "acceleration", "cylinders"]])
    if not rows:
        st.info("Pick vehicles above to compare them.")
        return
    comp = pd.DataFrame(rows, index=[f"A: {vA}", f"B: {vB}", f"C: {vC}"][:len(rows)]).T
    st.dataframe(comp, use_container_width=True)
def pg_export():
    UI.render_crumbs("Automotive Fleet", "Exports")
    header("SYSTEM", "Results & Exports", "Every file reflects the current session state and K.")
    st.caption("Descriptive filenames — source data is never overwritten.")
    if not HR():
        empty_state("DATASET NOT READY", "Upload a valid Auto MPG dataset before continuing.", "")
        return
    s = st.session_state
    def dl(label, df, fname):
        st.download_button(label, df.to_csv(index=False), file_name=fname, mime="text/csv", use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data**")
        if HC(): dl("Download Cleaned Dataset", s["df_clean"], "auto_mpg_cleaned.csv")
        else: st.info("Cleaned dataset: run imputation first.")
        if HS(): dl("Download Standardized Dataset", s["df_std"], "auto_mpg_standardized.csv")
        else: st.info("Standardized dataset: run standardization first.")
        st.markdown("**Clustering**")
        if HK():
            d = s["df_clean"].copy(); d["kmeans_cluster"] = s["kmeans"]["labels"]
            dl("Download K-Means Results", d, "auto_mpg_kmeans_results.csv")
        else: st.info("K-Means results: train K-Means first.")
        if HW():
            d = s["df_clean"].copy(); d["ward_cluster"] = s["ward"]["labels"]
            dl("Download Ward Results", d, "auto_mpg_ward_results.csv")
        else: st.info("Hierarchical results: run Ward first.")
    with c2:
        st.markdown("**Evaluation**")
        if s["dlab"] is not None:
            d = s["df_clean"][["car_name"]].copy(); d["dendrogram_cluster"] = s["dlab"]
            dl("Download Dendrogram Assignments", d, "auto_mpg_dendrogram_clusters.csv")
        else: st.info("Dendrogram assignments: extract them on Clustering Lab.")
        if s["cent_orig"] is not None: dl("Download Centroids", s["cent_orig"], "auto_mpg_centroids.csv")
        else: st.info("Centroids: train K-Means first.")
        rows = []
        if HK(): rows.append(("K-Means", s["kmeans"]["k"], round(s["kmeans"]["silhouette"], 4), round(s["kmeans"]["db"], 4)))
        if HW(): rows.append(("Ward", s["ward"]["k"], round(s["ward"]["silhouette"], 4), round(s["ward"]["db"], 4)))
        if rows: dl("Download Evaluation Metrics", pd.DataFrame(rows, columns=["method", "k", "silhouette", "davies_bouldin"]), "auto_mpg_evaluation.csv")
        else: st.info("Evaluation metrics: train a model first.")
        st.markdown("**Complete**")
        if HK() or HW():
            d = s["df_clean"].copy()
            if HK(): d["kmeans_cluster"] = s["kmeans"]["labels"]
            if HW(): d["ward_cluster"] = s["ward"]["labels"]
            if s["dlab"] is not None: d["dendrogram_cluster"] = s["dlab"]
            dl("Download Complete Clustered Dataset", d, "auto_mpg_complete_clustered.csv")
        else: st.info("Complete dataset: train a model first.")
def pg_insights():
    UI.render_crumbs("Automotive Fleet", "Insights")
    header("INTERPRET", "Engineering Insights", "A concise data-driven report — every finding carries its evidence.")
    if not HR():
        empty_state("DATASET NOT READY", "Upload a valid Auto MPG dataset before continuing.", "")
        return
    df = st.session_state["df_clean"] if HC() else st.session_state["df_raw"]
    corr = calculate_correlations(df)
    st.subheader("Fleet structure & efficiency")
    try:
        r_wm = float(corr.loc["weight", "mpg"])
        UI.render_insight_card("Weight ↔ MPG",
                               f"Heavier vehicles tend to have lower MPG (r = {r_wm:.4f}, {strength_label(r_wm)} {'negative' if r_wm < 0 else 'positive'}).",
                               f"DATA EVIDENCE: r = {r_wm:.4f}. Association only — not causation.")
        r_hw = float(corr.loc["horsepower", "weight"])
        UI.render_insight_card("Horsepower ↔ Weight",
                               f"Power and mass move together (r = {r_hw:.4f}).",
                               f"DATA EVIDENCE: r = {r_hw:.4f}. Association only — not causation.")
    except Exception:
        pass
    st.subheader("Clustering behaviour")
    s = st.session_state
    if s["kmeans"] is not None:
        km = s["kmeans"]
        UI.render_insight_card(f"K-Means K={km['k']}",
                               f"Silhouette {km['silhouette']:.4f} (higher better), Davies-Bouldin {km['db']:.4f} (lower better), inertia {km['inertia']:.2f}.",
                               "MODEL EVIDENCE: live K-Means result on the standardized matrix.")
    else:
        st.caption("Train K-Means to populate clustering findings.")
    st.subheader("Model comparison")
    if s["kmeans"] is not None and s["ward"] is not None:
        sl = "K-Means" if s["kmeans"]["silhouette"] >= s["ward"]["silhouette"] else "Ward"
        dl = "K-Means" if s["kmeans"]["db"] <= s["ward"]["db"] else "Ward"
        UI.render_insight_card("K-Means vs Ward",
                               f"Silhouette leader: {sl}. Davies-Bouldin leader: {dl}. Overall: no single winner.",
                               f"MODEL EVIDENCE: gaps {abs(s['kmeans']['silhouette'] - s['ward']['silhouette']):.4f} / {abs(s['kmeans']['db'] - s['ward']['db']):.4f}.")
    else:
        st.caption("Run both methods to compare.")
    st.caption("Correlation describes linear association and does not establish causation.")
def pg_method():
    UI.render_crumbs("Automotive Fleet", "Methodology")
    header("SYSTEM", "What Did We Do?", "Full methodology trace — every step maps to the assignment.")
    st.markdown("<div class='darkpanel'><b>DATA</b> ↓ CLEAN (? → NaN) ↓ IMPUTE (cylinder-group median) ↓ STANDARDIZE (z = (x−μ)/σ, 5 features) ↓ CLUSTER (K-Means 42/10 + Ward, K=2–6) ↓ EVALUATE (Silhouette, Davies-Bouldin, Elbow) ↓ PCA (2D visualization only) ↓ ARCHETYPE (inverse-transformed centroids, data-driven names)</div>", unsafe_allow_html=True)
    st.markdown("<div class='darkpanel'><b>Part A</b> ✓ missing horsepower · distributions · correlations · z-score<br><b>Part B</b> ✓ K-Means · Ward · elbow · silhouette · dendrogram<br><b>Part C</b> ✓ silhouette · Davies-Bouldin · PCA · centroids · archetypes · engineering inference<br><b>Interface</b> ✓ upload · training · K/method controls · visualization · exports</div>", unsafe_allow_html=True)
    if st.session_state.get("viva_mode"):
        UI.viva_box("Why grouped imputation? Same-cylinder engines share a power band. Why standardize? Weight (thousands) would otherwise dominate MPG (tens) in Euclidean distance. Why K=2–6? Assignment range — silhouette peaks at K=2 while K=3 gives finer fleet segmentation. Why Ward? A second agglomerative lens to cross-check K-Means. Why PCA only for display? It is a projection, not the clustering space.")
    s = st.session_state
    qa = [("Dataset loaded", HR()), ("Horsepower imputed", HC()), ("Standardized", HS()),
          ("Clustering complete", HK() or HW()), ("Evaluation complete", HK() or HW()),
          ("PCA ready", s["pca"] is not None or HS()), ("Archetypes ready", s["arch"] is not None),
          ("Exports ready", HC())]
    st.subheader("Quality assurance (live state)")
    for lbl, ok in qa:
        st.markdown(f"<div class='darkpanel'>{lbl} &nbsp; {'✓' if ok else '○'}</div>", unsafe_allow_html=True)
def pg_about():
    UI.render_crumbs("Automotive Fleet", "About")
    header("SYSTEM", "About This Project", "Methodology, compliance and scientific notes.")
    st.markdown("<div class='darkpanel'><b>Automotive Fleet Profiling & Efficiency Clustering</b><br>Student: <b>Prakhar Agrawal</b> — Register Number: <b>26MML0003</b><br>Course: <b>Machine Learning Lab (MACSE504)</b> — Question <b>4</b>, Assessment <b>10</b><br><span style='color:#94A3B8'>Methodology: Python, Pandas, Scikit-learn, SciPy, Plotly, Streamlit.</span></div>", unsafe_allow_html=True)
    st.markdown("""<div class='panel'><b>Automotive Fleet Profiling & Efficiency Clustering</b><br>
Question 4 - Machine Learning Lab (MACSE504)<br>Prakhar Agrawal - 26MML0003<br><br>
<b>Methodology:</b> EDA, cylinder-group median imputation, Z-score standardization, K-Means (random_state=42, n_init=10, K=2-6),
Ward hierarchical clustering, Elbow Method, Silhouette Coefficient, Davies-Bouldin Index, PCA visualization (2D only),
centroid-based archetype interpretation.<br><br>
Dataset source: <b>UCI Auto MPG Dataset</b>.<br>
Interface contract: load dataset, training, parameter adjustment, visualization.<br><br>
<b>Scientific notes:</b> PCA never performs clustering. Silhouette higher is better; Davies-Bouldin lower is better;
inertia must be read via the elbow. Ward dendrogram display is a truncated sample; extraction uses full linkage via fcluster.</div>""", unsafe_allow_html=True)
PAGES = {"Overview": (pg_overview, pg_overview_b), "Dataset": (pg_dataset,), "Preprocessing": (pg_pre, pg_pre_b),
         "EDA Explorer": (pg_eda, pg_eda_b), "Clustering Lab": (pg_cluster,), "Cluster Quality": (pg_eval,),
         "PCA Explorer": (pg_pca,), "Fleet Archetypes": (pg_arch, pg_arch_b,), "Vehicle Lookup": (pg_lookup,),
         "Insights": (pg_insights,), "Methodology": (pg_method,), "Exports": (pg_export,), "About": (pg_about,)}
st.sidebar.markdown("<div class='brand-lockup'><div class='b1'>AUTO MPG ANALYTICS</div><div class='b2'>M.Tech ML Lab \u2022 Q4 \u2022 26MML0003</div></div>", unsafe_allow_html=True)
st.sidebar.toggle("Demo Mode", key="demo_mode", help="Highlights the primary workflow; never changes computation.")
st.sidebar.toggle("Presentation Mode", key="presentation_mode", help="Cleaner, larger projector layout; never changes computation.")
def _sync_page(key):
    page = st.session_state[key]
    if page is None:
        return
    st.session_state["page"] = page
    # Clear other groups (safe: callbacks run before widgets instantiate this run).
    for k in ("nav1", "nav2", "nav3", "nav4"):
        if k != key and st.session_state.get(k) is not None:
            st.session_state[k] = None
G1 = ["Overview", "Dataset", "Preprocessing", "EDA Explorer"]
G2 = ["Clustering Lab", "Cluster Quality", "PCA Explorer"]
G3 = ["Fleet Archetypes", "Vehicle Lookup", "Insights"]
G4 = ["Methodology", "Exports", "About"]
# Pre-sidebar sync: programmatic navigation (buttons) only mutates the
# non-widget "page" key and reruns. Align the sidebar radios BEFORE they
# instantiate so the highlight always matches the rendered page without
# ever writing a widget key after instantiation.
_cur = st.session_state.get("page", "Overview")
def _nav_group(label, options):
    st.sidebar.markdown(f"<div class='navlbl'>{label}</div>", unsafe_allow_html=True)
    for opt in options:
        active = (_cur == opt)
        kind = "primary" if active else "secondary"
        if st.sidebar.button(opt, key=f"navbtn_{opt}", type=kind, use_container_width=True):
            goto_page(opt)
_nav_group("ANALYSIS", G1)
_nav_group("MODELING", G2)
_nav_group("INTERPRETATION", G3)
_nav_group("SYSTEM", G4)
st.sidebar.divider()
if st.sidebar.button("Reset Analysis", use_container_width=True, help="Clears session state; project files are preserved."):
    pg = st.session_state.get("page", "Overview")
    reset_state(keep_nav=True)
    st.session_state["page"] = pg
    st.rerun()
st.sidebar.caption("Load -> Inspect -> Clean -> Standardize -> Explore -> Tune -> Cluster -> Evaluate -> Visualize -> Interpret")
for fn in PAGES[st.session_state.get("page", "Overview")]:
    fn()
