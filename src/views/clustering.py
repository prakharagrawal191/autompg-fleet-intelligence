"""Clustering Lab - choose the algorithm and K, fit, and read the partition."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, fcluster  # noqa: E402

from src import chart_factory as CF  # noqa: E402
from src import state as S  # noqa: E402
from src import theme as T  # noqa: E402
from src import ui_components as UI  # noqa: E402
from src.auto_mpg_utils import (  # noqa: E402
    CONTINUOUS_COLS, DEFAULT_N_INIT, DEFAULT_RANDOM_STATE, K_RANGE, dendro_cut_height,
    evaluate_clustering, optimal_k_from_silhouette,
)

# Option labels are part of the existing test contract.
VIEWS = {
    "Weight vs MPG (weight / mpg)": ("weight", "mpg"),
    "Horsepower vs MPG (horsepower / mpg)": ("horsepower", "mpg"),
    "Displacement vs MPG (displacement / mpg)": ("displacement", "mpg"),
    "PCA projection (PC1 / PC2)": ("PC1", "PC2"),
}
AXIS = {"mpg": "MPG", "weight": "Weight (lb)", "horsepower": "Horsepower (hp)",
        "displacement": "Displacement (cu in)", "acceleration": "Acceleration (s)"}
UNIT = {"mpg": "MPG", "weight": "lb", "horsepower": "hp", "displacement": "cu in", "acceleration": "s"}
METHOD_INFO = {
    S.KMEANS: "Partitions the fleet around K centroids by minimizing within-cluster squared distance.",
    S.WARD: "Merges vehicles bottom-up, each time joining the two clusters whose merge least increases "
            "within-cluster variance.",
}


def _method_note(s, method):
    if method == S.WARD:
        return "Ward linkage. Labels from AgglomerativeClustering; the dendrogram cut uses fcluster(maxclust)."
    rs, ni = S.km_params(s)
    valid = (rs, ni) == (DEFAULT_RANDOM_STATE, DEFAULT_N_INIT)
    return (f"random_state={rs}, n_init={ni}" + (" (validated configuration)." if valid else " (non-default).")
            + " Inertia is defined.")


# ---- callbacks (run before the script, so they may write any key) ----------
def _k_changed():
    S.set_k(st.session_state["k_select"])


def _run():
    r = S.fit_active()
    st.session_state["flash"] = (f"{S.METHOD_TITLES[st.session_state['method']]} fitted at K={r['k']} · "
                                 f"silhouette {r['silhouette']:.4f}")


def _sweep():
    S.compute_sweep()
    st.session_state["flash"] = "K=2–6 sweep computed"


def _build_dendro():
    S.build_dendro()
    st.session_state["flash"] = f"Ward linkage built on all {len(S.X())} vehicles"


def _extract():
    S.extract_dendro()
    st.session_state["flash"] = f"Dendrogram assignments extracted at K={st.session_state['dk']}"


# ---- controls --------------------------------------------------------------
def _method_card(col, s, method, label, selected):
    with col.container(border=True, height="stretch", key=f"mcard_{'on' if selected else 'off'}_{method}"):
        state = UI.pill("Selected", "gold") if selected else UI.pill("Not selected", "neutral")
        UI.md(f"<div class='ui-section'>{state}<div class='ui-section-title' role='heading' aria-level='3'>"
              f"{UI.esc(S.METHOD_TITLES[method])}</div><div class='ui-step-body'>{UI.esc(METHOD_INFO[method])}"
              f"</div><div class='ui-section-sub'>{UI.esc(_method_note(s, method))}</div></div>")
        st.button(label, key=f"use_{method}", width="stretch", on_click=S.set_method, args=(method,),
                  icon=":material/check:" if selected else None)


def _controls(s):
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        UI.section("Algorithm", eyebrow="Step 1")
        a, b = st.columns(2)
        _method_card(a, s, S.KMEANS, "Use K-Means", s["method"] == S.KMEANS)
        _method_card(b, s, S.WARD, "Use Ward Hierarchical", s["method"] == S.WARD)
    with right:
        UI.section("Number of clusters", eyebrow="Step 2")
        if s.get("k_select") != s["selected_k"]:
            s["k_select"] = s["selected_k"]  # align before the widget renders
        st.slider("K (2 to 6) — NUMBER OF CLUSTERS", min(K_RANGE), max(K_RANGE), key="k_select",
                  on_change=_k_changed, help="The assignment fixes the candidate range at K=2–6.")
        st.caption(f"Current K: **{s['selected_k']}**. Changing K refits nothing by itself: it clears the fitted "
                   "assignments, metrics, cluster sizes, centroids, PCA coloring and archetypes until you run "
                   "clustering again. Cluster numbers are identifiers — Cluster 0 at K=2 is not the same "
                   "population as Cluster 0 at K=3.")
        with st.expander("Reproducibility & advanced settings", key="tech_repro"):
            c1, c2 = st.columns(2)
            c1.number_input("random_state", 0, 9999, key="km_rs", on_change=S.invalidate_kmeans,
                            help="K-Means seed. Validated configuration: 42.")
            c2.number_input("n_init", 1, 50, key="km_ni", on_change=S.invalidate_kmeans,
                            help="K-Means restarts. Validated configuration: 10. Changing a setting clears the "
                                 "K-Means fit.")
            UI.keyvals([("Ward linkage", "ward (fixed)"), ("Candidate K", f"{min(K_RANGE)}–{max(K_RANGE)}"),
                        ("Features", ", ".join(CONTINUOUS_COLS)), ("Dataset fingerprint", s["dataset_id"])])


def _run_panel(s, r):
    method = S.METHOD_TITLES[s["method"]]
    a, b = st.columns([1, 2], vertical_alignment="center")
    a.button("RUN CLUSTERING", type="primary", width="stretch", icon=":material/play_arrow:", on_click=_run,
             help="Fits the selected algorithm at the selected K on the standardized features.")
    with b:
        if r is not None:
            UI.md(f"<div class='ui-prov'>{UI.pill('Fitted', 'success')}<b>{UI.esc(S.provenance(s['method']))}</b>"
                  "</div>")
        else:
            UI.md(f"<div class='ui-prov'>{UI.pill('Not fitted', 'neutral')}"
                  f"<span>{UI.esc(method)} has no result at K={s['selected_k']} yet.</span></div>")
    rs, ni = S.km_params(s)
    if (rs, ni) != (DEFAULT_RANDOM_STATE, DEFAULT_N_INIT):  # visible even when the settings panel is hidden
        UI.callout(f"Non-default K-Means settings (random_state={rs}, n_init={ni}). K-Means results no longer "
                   f"correspond to the validated configuration (random_state={DEFAULT_RANDOM_STATE}, "
                   f"n_init={DEFAULT_N_INIT}).", tone="warning")


# ---- primary results -------------------------------------------------------
def _model_card(s, r):
    counts = r["counts"]
    inertia = f"{r['inertia']:.2f}" if s["method"] == S.KMEANS else "n/a"
    top = [str(int(i)) for i in counts.index[counts == counts.max()]]
    UI.kpis([
        {"label": "Algorithm", "value": S.METHOD_TITLES[s["method"]], "sub": f"K={r['k']}"},
        {"label": "Silhouette", "value": f"{r['silhouette']:.4f}", "sub": "higher is better (−1 to 1)"},
        {"label": "Davies–Bouldin", "value": f"{r['db']:.4f}", "sub": "lower is better (≥ 0)"},
        {"label": "Inertia", "value": inertia,
         "sub": "within-cluster sum of squares" if s["method"] == S.KMEANS else "not defined for Ward"},
        {"label": "Largest cluster", "value": f"{int(counts.max())}",
         "sub": f"of {int(counts.sum())} vehicles · Cluster {', '.join(top)}" + (" (tie)" if len(top) > 1 else "")},
    ])


def _primary_view(s, r):
    k, labels = r["k"], r["labels"]
    UI.section("Cluster map", "Every point is a vehicle; color and marker shape both identify its cluster.")
    c1, c2, c3 = st.columns([1.3, 1.2, 1.5])
    view = c1.selectbox("Cluster view", list(VIEWS), key="cluster_view")
    show_c = c2.checkbox("Show centroid markers", key="lab_centres",
                         help="K-Means: its centroids mapped back with scaler.inverse_transform. Ward: the mean of "
                              "each cluster's members. PCA view: the same centers projected with the fitted PCA.")
    visible = c3.pills("Highlight clusters", list(range(k)), selection_mode="multi", default=list(range(k)),
                       format_func=T.cluster_label, key=f"lab_visible_{s['method']}_{k}",
                       help="Changes the view only; metrics always describe the full solution.")
    x, y = VIEWS[view]
    frame = s["df_clean"]
    centers = None
    if x == "PC1":
        p = S.ensure_pca(s)
        frame = frame.assign(PC1=p["coords"][:, 0], PC2=p["coords"][:, 1])
        v = p["variance_ratio"]
        xt, yt = f"PC1 ({100 * v[0]:.1f}% of variance)", f"PC2 ({100 * v[1]:.1f}% of variance)"
        if show_c:
            centers = p["model"].transform(S.centers_std(s["method"], s))
    else:
        xt, yt = AXIS[x], AXIS[y]
        if show_c:
            c = S.centers_orig(s["method"], s)
            centers = c[:, [CONTINUOUS_COLS.index(x), CONTINUOUS_COLS.index(y)]]
    shown = sorted(visible) if visible else list(range(k))
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        fig = CF.cluster_scatter(frame, x, y, labels, k, xtitle=xt, ytitle=yt,
                                 title=f"{S.METHOD_TITLES[s['method']]} clusters · K={k}",
                                 centers=centers, center_label="Centroid" if s["method"] == S.KMEANS else "Mean",
                                 visible=shown if len(shown) < k else None, height=460)
        CF.show(fig, alt=f"Scatter of {yt} against {xt} for {len(labels)} vehicles colored by "
                         f"{S.METHOD_TITLES[s['method']]} cluster (K={k}).")
        if show_c:
            st.caption("× markers: " + ("K-Means centroids (inverse-transformed to original units)"
                                       if s["method"] == S.KMEANS else "Ward cluster means")
                       + (", projected with the fitted PCA." if x == "PC1" else "."))
        if len(shown) < k:
            n_sub = int(np.isin(labels, shown).sum())
            st.caption(f"Highlighting {len(shown)} of {k} clusters ({n_sub} of {len(labels)} vehicles). "
                       "Silhouette, Davies–Bouldin and inertia above still describe the full solution.")
    with right:
        CF.show(CF.cluster_sizes(r["counts"], "Cluster sizes", f"{int(r['counts'].sum())} vehicles"),
                alt=f"Horizontal bar chart of vehicles per cluster for K={k}.")
        if s["method"] == S.KMEANS and s.get("arch"):
            UI.md("<div class='ui-section-sub'>Archetypes (from K-Means centroids)</div>"
                  + "<br>".join(UI.chip(i, s["arch"].get(i, "")) for i in range(k)))


# ---- secondary tabs ----------------------------------------------------------
def _elbow_fig(kr, upto, sel):
    d = kr[kr["K"] <= upto]
    ks, inert = d["K"].to_numpy(), d["inertia"].to_numpy()
    prev = [""] + [f"−{a - b:.2f} ({100 * (a - b) / a:.1f}%) vs K={k0}"
                   for k0, a, b in zip(ks[:-1], inert[:-1], inert[1:])]
    fig = go.Figure(go.Scatter(x=ks, y=inert, mode="lines+markers", name="Inertia",
                               line=dict(color=T.TEAL, width=2), marker=dict(size=9, color=T.TEAL),
                               customdata=prev, hovertemplate="K=%{x}<br>Inertia %{y:.2f}<br>%{customdata}"
                                                              "<extra></extra>"))
    if sel in ks:
        fig.add_trace(go.Scatter(x=[sel], y=[float(d.set_index("K").loc[sel, "inertia"])], mode="markers",
                                 name=f"Selected K={sel}", hoverinfo="skip",
                                 marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=3, color=T.GOLD))))
    if len(ks) > 1:
        drops = inert[:-1] - inert[1:]
        j = int(np.argmax(drops))
        fig.add_annotation(x=(ks[j] + ks[j + 1]) / 2, y=(inert[j] + inert[j + 1]) / 2,
                           text=f"largest drop<br>−{drops[j]:.1f} ({100 * drops[j] / inert[j]:.1f}%)",
                           showarrow=False, xshift=34, yshift=14, font=dict(color=T.TEXT_2, size=11))
    fig = CF.style(fig, 330, "Elbow · K-Means inertia", "Inertia always falls as K grows; read the bend.",
                   "Number of clusters (K)", "Inertia (within-cluster SS)")
    fig.update_xaxes(tickvals=list(K_RANGE), range=[min(K_RANGE) - 0.4, max(K_RANGE) + 0.4])
    return fig


def _why_k(s, kr):
    sel = int(s["selected_k"])
    best_k, best_v = optimal_k_from_silhouette(kr)
    sil = kr.set_index("K")["silhouette"]
    rank = int((sil > sil[sel]).sum()) + 1
    parts = [f"Silhouette is highest at <b>K={best_k}</b> ({best_v:.4f}) among K={min(K_RANGE)}–{max(K_RANGE)}."]
    if sel == best_k:
        parts.append(f"The selected K={sel} matches that maximum.")
    else:
        parts.append(f"The selected <b>K={sel}</b> scores {sil[sel]:.4f} (rank {rank} of {len(sil)}).")
    d = kr.sort_values("K")
    inert, ks = d["inertia"].to_numpy(), d["K"].to_numpy()
    drops = inert[:-1] - inert[1:]
    j = int(np.argmax(drops))  # same criterion as the elbow annotation: largest absolute reduction
    later = ", ".join(f"{v:.2f}" for v in drops[j + 1:])
    parts.append(f"Inertia falls most from K={ks[j]} to K={ks[j + 1]} (−{drops[j]:.2f}, "
                 f"{100 * drops[j] / inert[j]:.1f}% of the K={ks[j]} inertia)"
                 + (f"; later steps reduce it by {later}." if later else "."))
    if S.is_project_data(s) and S.DOCUMENTED_K in sil.index:
        dk = S.DOCUMENTED_K
        if dk == best_k:
            parts.append(f"The Assessment 10 notebook documents K={dk} for this dataset, which is also the "
                         "silhouette maximum.")
        else:
            parts.append(f"For this project dataset, the Assessment 10 notebook documents <b>K={dk}</b> as a joint "
                         f"engineering choice: finer segmentation than K={best_k} at a lower silhouette "
                         f"({sil[dk]:.4f} vs {best_v:.4f}) — a trade-off, not the metric optimum.")
    UI.takeaway(" ".join(parts), html=True)


def _choosing_k(s):
    a, b = st.columns([1, 1.4], vertical_alignment="bottom", gap="large")
    a.button("Compute Elbow + Silhouette (K=2–6)", icon=":material/insights:", on_click=_sweep, width="stretch",
             help="K-Means at every K in 2–6 with the current random_state and n_init.")
    kr = s.get("krange")
    if kr is None:
        UI.empty_state("ELBOW NOT COMPUTED", "Compute the K=2–6 sweep to see the elbow and silhouette curves.")
        return
    rs, ni = S.km_params(s)
    sel = int(s["selected_k"])
    ks = [int(k) for k in kr["K"]]
    upto = b.select_slider("Reveal K=2→6 (presentation aid)", options=ks, value=ks[-1], key="reveal_k",
                           help="Shows the sweep one K at a time; it never recomputes anything.")
    shown = f"K={ks[0]}–{upto}" + (f"; selected K={sel} marked" if sel <= upto else "")
    a, b = st.columns(2, gap="large")
    with a:
        CF.show(_elbow_fig(kr, upto, sel), alt=f"Line chart of K-Means inertia for {shown}; the largest drop "
                                               "is annotated.")
    with b:
        documented = S.DOCUMENTED_K if S.is_project_data(s) else None
        CF.show(CF.silhouette_by_k(kr, sel, upto, documented), alt=f"Bar chart of silhouette score for {shown}.")
    _why_k(s, kr)
    with st.expander(f"Sweep table · K-Means (random_state={rs}, n_init={ni})", key="tech_sweep"):
        d = kr.sort_values("K").copy()
        d["Δ inertia vs previous K"] = -d["inertia"].diff()
        st.dataframe(d.rename(columns={"inertia": "Inertia", "silhouette": "Silhouette", "db": "Davies–Bouldin"}),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f")
                                    for c in ["Inertia", "Silhouette", "Davies–Bouldin", "Δ inertia vs previous K"]})
    best_k, _ = optimal_k_from_silhouette(kr)
    note = (f"K is compared over {min(K_RANGE)}–{max(K_RANGE)} because the assignment fixes that range. "
            f"Here silhouette peaks at K={best_k}; weigh that against the elbow bend.")
    if S.is_project_data(s) and best_k != S.DOCUMENTED_K:
        note += (f" The project's documented K={S.DOCUMENTED_K} trades some silhouette for a finer, more useful "
                 "fleet segmentation.")
    UI.viva(note)


def _compare_clusters(s, r):
    k = r["k"]
    opts = list(range(k))
    if s.get("cmp_a") not in opts:
        s["cmp_a"] = 0
    if s.get("cmp_b") not in opts:
        s["cmp_b"] = min(1, k - 1)
    kind = "K-Means centroids (inverse-transformed)" if s["method"] == S.KMEANS else "Ward cluster means"
    st.caption(f"Comparing {kind} in original units for the current {S.METHOD_TITLES[s['method']]} K={k} fit.")
    a, b = st.columns(2)
    ca = a.selectbox("Cluster A", opts, key="cmp_a", format_func=T.cluster_label)
    cb = b.selectbox("Cluster B", opts, key="cmp_b", format_func=T.cluster_label)
    c = S.centers_orig(s["method"], s)
    rows = [{"Feature": f, "Unit": UNIT[f], f"Cluster {ca}": c[ca][j], f"Cluster {cb}": c[cb][j],
             "Δ (A − B)": c[ca][j] - c[cb][j]} for j, f in enumerate(CONTINUOUS_COLS)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={col: st.column_config.NumberColumn(format="%.2f")
                                for col in [f"Cluster {ca}", f"Cluster {cb}", "Δ (A − B)"]})


def _link_colors(Z, labels):
    """Color each merge by the cluster all its leaves share (else neutral)."""
    below = [{int(v)} for v in labels]
    for a, b in Z[:, :2].astype(int):
        below.append(below[a] | below[b])
    return lambda i: T.cluster_color(next(iter(below[i]))) if len(below[i]) == 1 else T.MUTED


def _dendrogram(s):
    st.caption("Ward linkage over every vehicle in the standardized space. The tree is drawn with lastp "
               "truncation for readability; the linkage itself is never truncated.")
    a, b = st.columns([1, 1])
    a.button("Build Ward dendrogram", icon=":material/account_tree:", on_click=_build_dendro,
             help="Computes the full Ward linkage and extracts the current-K assignments.")
    detail = b.checkbox("Detailed View (more branches)", key="dendro_detail")
    d = s.get("dendro")
    if d is None:
        UI.empty_state("NO DENDROGRAM", "Build the Ward dendrogram to see the merge hierarchy.",
                       "Press Build Ward dendrogram")
        return
    Z, k = d["Z_full"], int(s["selected_k"])
    n = Z.shape[0] + 1
    cut = dendro_cut_height(Z, k)
    maxclust = fcluster(Z, t=k, criterion="maxclust") - 1
    by_height = fcluster(Z, t=cut, criterion="distance")
    same = len(set(by_height)) == len(set(maxclust)) == k and len(set(zip(maxclust, by_height))) == k
    # Color by the Ward fit's ids when it is the same partition as the cut (checked), so the tree
    # matches the Ward scatter; otherwise by the extracted dendrogram ids.
    w = S.model(S.WARD, s)
    ward_ids = w is not None and w["k"] == k and len(set(zip(w["labels"], maxclust))) == len(set(maxclust))
    ids = np.asarray(w["labels"]) if ward_ids else maxclust
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=110)
    CF.mpl_dark(fig, ax)
    drawn = dendrogram(Z, ax=ax, truncate_mode="lastp", p=60 if detail else 30, show_leaf_counts=True,
                       leaf_rotation=90, leaf_font_size=8, link_color_func=_link_colors(Z, ids))
    merges = len(drawn["icoord"])
    cut_line = ax.axhline(cut, color=T.GOLD, linestyle="--", linewidth=1.4, label=f"cut → K={k} (height {cut:.2f})")
    ax.set_title(f"Ward dendrogram · {n} vehicles · last {merges} merges shown", loc="left", fontsize=11,
                 color=T.WHITE)  # set_title resets the color mpl_dark applied
    ax.set_xlabel("Branches: (n) = vehicles merged in that branch; plain number = one vehicle (row index)")
    ax.set_ylabel("Ward linkage distance")
    ax.legend(handles=[cut_line] + [Patch(color=T.cluster_color(i), label=f"Cluster {i}") for i in sorted(set(ids))],
              loc="upper right", facecolor=T.SURFACE, edgecolor=T.BORDER, labelcolor=T.WHITE, fontsize=9)
    st.pyplot(fig, width="stretch", alt=f"Ward dendrogram of {n} vehicles showing the last {merges} merges, "
                                        f"branches colored by cluster, with the cut line for K={k}.")
    plt.close(fig)
    st.caption(f"Linkage computed on all {n} observations; the plot shows the last {merges} merges. Branch colors "
               + ("follow the Ward fit's cluster numbers (the same partition as this cut). " if ward_ids else
                  "follow the extracted dendrogram cluster numbers. ")
               + (f"Cutting at height {cut:.2f} yields exactly the partition fcluster(maxclust) extracts."
                  if same else "Warning: tied merge heights — the cut height and fcluster(maxclust) do not give "
                               f"{k} identical clusters."))
    st.button("Extract Cluster Assignments (fcluster, maxclust)", icon=":material/call_split:", on_click=_extract,
              help="fcluster(Z, t=K, criterion='maxclust') on the full linkage.")
    if s.get("dlab") is not None and s.get("dk") == k:
        counts = pd.Series(s["dlab"]).value_counts().sort_index()
        sil, db = evaluate_clustering(S.X(s), s["dlab"])
        UI.keyvals([("Extracted clusters", f"{len(counts)} (requested K={k}) · sizes "
                     + " / ".join(str(int(v)) for v in counts)),
                    ("Silhouette (dendrogram labels)", f"{sil:.4f}"),
                    ("Davies–Bouldin (dendrogram labels)", f"{db:.4f}")])
    UI.viva("Ward joins, at every step, the two clusters whose merge adds the least within-cluster variance. "
            "Assignments are extracted by cutting the same full linkage with fcluster(maxclust); Cluster Quality "
            "reports how closely they agree with the AgglomerativeClustering labels.")


def _assignments(s):
    frame = s["df_clean"][["car_name", "mpg", "horsepower", "weight"]].copy()
    cols = []
    for m, col in ((S.KMEANS, "K-Means Cluster"), (S.WARD, "Ward Cluster")):
        r = S.model(m, s)
        if r is not None:
            frame[col] = r["labels"]
            cols.append(col)
    if s.get("dlab") is not None and s.get("dk") == s["selected_k"]:
        frame["Dendrogram Cluster"] = s["dlab"]
        cols.append("Dendrogram Cluster")
    if not cols:
        UI.empty_state("NO ASSIGNMENTS", "No clustering has been fitted at the current K.", "Press RUN CLUSTERING")
        return
    act = "K-Means Cluster" if s["method"] == S.KMEANS and "K-Means Cluster" in cols else cols[0]
    f = st.selectbox("Cluster Filter", ["All"] + sorted(frame[act].unique().tolist()),
                     format_func=lambda v: v if v == "All" else T.cluster_label(v),
                     help=f"Filters rows by the {act} column.")
    show = frame if f == "All" else frame[frame[act] == f]
    st.caption(f"Filtering by **{act}** · showing {len(show)} of {len(frame)} vehicles. Model metrics always "
               f"describe the full K={s['selected_k']} solution.")
    st.dataframe(show, hide_index=True, width="stretch", height=360, column_config={
        "car_name": st.column_config.TextColumn("Vehicle"),
        "mpg": st.column_config.NumberColumn("MPG", format="%.1f"),
        "horsepower": st.column_config.NumberColumn("Horsepower (hp)", format="%.0f"),
        "weight": st.column_config.NumberColumn("Weight (lb)", format="%.0f")})


def render():
    s = st.session_state
    UI.page_header("Clustering Lab", "Choose an algorithm and K, fit it on the standardized features, and read "
                   "the resulting partition.", ("Automotive Fleet", "Modeling", "Clustering Lab"))
    if not S.has_raw(s):
        UI.empty_state("DATASET NOT READY", "Load or upload an Auto MPG dataset before clustering.",
                       "Dataset → Use Project Dataset")
        st.button("Open Dataset", on_click=S.goto, args=("Dataset",))
        return
    if not S.has_std(s):
        UI.empty_state("PREPROCESSING INCOMPLETE", "Clustering runs on the imputed, standardized features.",
                       "Preprocessing → Apply imputation → Standardize")
        st.button("Open Preprocessing", on_click=S.goto, args=("Preprocessing",))
        return
    _controls(s)
    r = S.active_model(s)
    _run_panel(s, r)
    if r is None:
        other = [m for m in S.fitted_methods(s) if m != s["method"]]
        hint = (f" {S.METHOD_TITLES[other[0]]} is fitted at K={s['selected_k']} — switch the algorithm to view it."
                if other else "")
        UI.empty_state("NO CLUSTER MODEL",
                       f"{S.METHOD_TITLES[s['method']]} is not fitted at K={s['selected_k']}.{hint}",
                       "Press RUN CLUSTERING")
    else:
        _model_card(s, r)
        _primary_view(s, r)
    # Keyed + tracked so the open tab survives layout changes above it (e.g. the first fit).
    tabs = st.tabs(["Choosing K", "Compare clusters", "Ward dendrogram", "Assignments"], key="lab_tab",
                   on_change="rerun")
    with tabs[0]:
        _choosing_k(s)
    with tabs[1]:
        if r is None:
            UI.empty_state("NOTHING TO COMPARE", f"{S.METHOD_TITLES[s['method']]} is not fitted at "
                           f"K={s['selected_k']}.", "Press RUN CLUSTERING")
        else:
            _compare_clusters(s, r)
    with tabs[2]:
        _dendrogram(s)
    with tabs[3]:
        _assignments(s)
