"""Engineering Insights - a short, evidence-led reading of the fleet."""
import numpy as np
import streamlit as st

from src import state as S
from src import ui_components as UI
from src.auto_mpg_utils import calculate_correlations, optimal_k_from_silhouette, strength_label

PAIRS = [("weight", "mpg", "Weight ↔ MPG"), ("horsepower", "mpg", "Horsepower ↔ MPG"),
         ("weight", "horsepower", "Weight ↔ Horsepower")]


def _grid(cards):
    cols = st.columns(2)
    for j, card in enumerate(cards):
        with cols[j % 2]:
            UI.insight(*card)


def _data_findings(df):
    corr = calculate_correlations(df)
    out = []
    for a, b, title in PAIRS:
        r = float(corr.loc[a, b])
        if not np.isfinite(r):
            continue
        text = (f"Higher {a} tends to come with {'lower' if r < 0 else 'higher'} {b} in this dataset."
                if abs(r) >= 0.2 else f"No meaningful linear association between {a} and {b} in this dataset.")
        out.append((title, text,
                    f"Pearson r = {r:.4f} · {strength_label(abs(r))} {'negative' if r < 0 else 'positive'}",
                    "DATA EVIDENCE"))
    return out


def _model_findings(s):
    out, km, w = [], S.model(S.KMEANS, s), S.model(S.WARD, s)
    kr = s.get("krange")
    if kr is not None:
        best_k, best_v = optimal_k_from_silhouette(kr)
        sel = int(s["selected_k"])
        sel_v = float(kr.set_index("K").loc[sel, "silhouette"])
        verdict = ("matches that maximum" if sel == best_k
                   else f"scores {sel_v:.4f}, below the maximum — a granularity trade-off, not the optimum")
        out.append(("Silhouette across K=2–6",
                    f"The highest silhouette among tested K is at K={best_k}; the selected K={sel} {verdict}.",
                    f"K-Means sweep · max {best_v:.4f} at K={best_k}", "MODEL EVIDENCE"))
    if km is not None and w is not None:
        sl, dl = S.metric_leader(km, w, "silhouette", True), S.metric_leader(km, w, "db", False)
        if sl is None and dl is None:
            text = f"K-Means and Ward score the same on silhouette and Davies–Bouldin at K={km['k']}."
        elif sl == dl or None in (sl, dl):
            best = S.METHOD_TITLES[sl or dl]
            text = (f"{best} leads on both silhouette and Davies–Bouldin at K={km['k']}." if sl == dl else
                    f"{best} leads on one metric and ties on the other at K={km['k']}.")
        else:
            text = (f"Silhouette favors {S.METHOD_TITLES[sl]} while Davies–Bouldin favors {S.METHOD_TITLES[dl]}: "
                    "no unique overall winner from these two metrics alone.")
        out.append(("K-Means vs Ward", text,
                    f"Silhouette {km['silhouette']:.4f} vs {w['silhouette']:.4f} · "
                    f"Davies–Bouldin {km['db']:.4f} vs {w['db']:.4f}", "MODEL EVIDENCE"))
    elif km is not None or w is not None:
        m = S.KMEANS if km is not None else S.WARD
        r = km if km is not None else w
        out.append((f"{S.METHOD_TITLES[m]} at K={r['k']}",
                    f"The fitted {S.METHOD_TITLES[m]} solution has silhouette {r['silhouette']:.4f} "
                    f"(higher is better) and Davies–Bouldin {r['db']:.4f} (lower is better).",
                    f"{len(r['labels'])} vehicles · fit both methods to compare them", "MODEL EVIDENCE"))
    if km is not None and s.get("cent_orig") is not None and s.get("arch"):
        co = s["cent_orig"].set_index("cluster")
        heavy, eff = int(co["weight"].idxmax()), int(co["mpg"].idxmax())
        n = km["counts"]
        out.append(("Archetype extremes",
                    f"Cluster {heavy} ({s['arch'].get(heavy)}) has the highest mean weight; Cluster {eff} "
                    f"({s['arch'].get(eff)}) has the highest mean MPG.",
                    f"{co.loc[heavy, 'weight']:,.0f} lb · {co.loc[heavy, 'mpg']:.2f} MPG ({int(n[heavy])} vehicles) vs "
                    f"{co.loc[eff, 'weight']:,.0f} lb · {co.loc[eff, 'mpg']:.2f} MPG ({int(n[eff])} vehicles)",
                    "MODEL EVIDENCE"))
    if s.get("pca") is not None and (km is not None or w is not None):
        v = s["pca"]["variance_ratio"]
        share = float(np.sum(v))
        reach = "most, but not all," if share >= 0.8 else "only part"
        out.append(("Two-dimensional view",
                    f"PC1 and PC2 retain {100 * share:.2f}% of the standardized variance, so the PCA map shows "
                    f"{reach} of the 5-D clustering space.",
                    f"PC1 {100 * v[0]:.2f}% · PC2 {100 * v[1]:.2f}% · threshold for 'most': 80%", "MODEL EVIDENCE"))
    return out


def render():
    s = st.session_state
    UI.page_header("Insights", "Engineering findings from the data and fitted models; every card shows the "
                   "numbers behind it.", ("Automotive Fleet", "Interpretation", "Insights"))
    if not S.has_raw(s):
        UI.empty_state("DATASET NOT READY", "Load or upload an Auto MPG dataset to generate findings.",
                       "Dataset → Use Project Dataset")
        st.button("Open Dataset", on_click=S.goto, args=("Dataset",))
        return
    df = s["df_clean"] if S.has_clean(s) else s["df_raw"]
    UI.section("Fleet structure & efficiency", "Pearson correlations on the "
               + ("cleaned data." if S.has_clean(s) else "raw data (missing horsepower excluded pairwise)."))
    _grid(_data_findings(df))
    found = _model_findings(s)
    if found:
        UI.section("Clustering & interpretation", "From the fitted models, the K sweep and PCA.")
        _grid(found)
    missing = [txt for cond, txt in (
        (s.get("krange") is None, "compute the K=2–6 sweep"),
        (not S.has_km(s), "fit K-Means for archetype findings"),
        (not (S.has_km(s) and S.has_ward(s)), "fit both methods to compare them")) if cond]
    if missing:
        st.caption("More findings unlock when you " + "; ".join(missing) + " (Clustering Lab).")
    st.caption("Correlation describes association, not causation. Findings describe this dataset and "
               "preprocessing only; they are not general claims about vehicles or algorithms.")
