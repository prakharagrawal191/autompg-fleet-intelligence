"""Overview - the analysis story at a glance; every value is computed from live state."""
import numpy as np
import streamlit as st

from src import state as S
from src import ui_components as UI
from src.auto_mpg_utils import CONTINUOUS_COLS, calculate_correlations, strength_label


def _step(delta):
    s = st.session_state
    s["demo_step"] = max(0, min(len(S.DEMO_STEPS) - 1, int(s["demo_step"]) + delta))


def _next_stage(stages):
    return next((x for x in stages if x["state"] == "ready"), None)


def _findings(s):
    """Up to four evidence-led findings; each is omitted when its inputs don't exist."""
    out = []
    df = s["df_clean"] if S.has_clean(s) else s["df_raw"]
    r = float(calculate_correlations(df).loc["weight", "mpg"])
    if np.isfinite(r):
        text = (f"Heavier vehicles tend to have {'lower' if r < 0 else 'higher'} MPG in this dataset."
                if abs(r) >= 0.2 else "No meaningful linear association between weight and MPG in this dataset.")
        out.append(("DATA EVIDENCE", "Weight ↔ MPG", text,
                    f"Pearson r = {r:.4f} · {strength_label(abs(r))} "
                    f"{'negative' if r < 0 else 'positive'} · "
                    f"{'cleaned' if S.has_clean(s) else 'raw'} data · association, not causation"))
    kr = s.get("krange")
    if kr is not None and S.has_std(s):
        best = kr.loc[kr["silhouette"].idxmax()]
        cur = kr.set_index("K").loc[int(s["selected_k"]), "silhouette"]
        out.append(("MODEL EVIDENCE", "Silhouette across K=2–6",
                    f"The highest silhouette among tested K is at K={int(best['K'])}; "
                    f"the selected K={int(s['selected_k'])} scores {cur:.4f}.",
                    f"K-Means sweep · max silhouette {best['silhouette']:.4f}"))
    km = S.model(S.KMEANS, s)
    if km is not None and s.get("arch"):
        counts = km["counts"]
        top = [int(i) for i in counts.index[counts == counts.max()]]
        share = f"{int(counts.max())} of {int(counts.sum())} vehicles ({100 * counts.max() / counts.sum():.1f}%)"
        if len(top) == 1:
            out.append(("MODEL EVIDENCE", "Largest fleet segment",
                        f"{s['arch'].get(top[0], f'Cluster {top[0]}')} is the largest K-Means group at "
                        f"K={km['k']}.", f"Cluster {top[0]}: {share}"))
        else:
            out.append(("MODEL EVIDENCE", "Largest fleet segment",
                        f"Clusters {', '.join(map(str, top))} tie for the largest K-Means group at K={km['k']}.",
                        f"Each: {share}"))
    if s.get("pca") is not None and S.has_model(s):
        v = s["pca"]["variance_ratio"]
        out.append(("MODEL EVIDENCE", "Two-dimensional view",
                    f"PC1 and PC2 retain {100 * float(np.sum(v)):.2f}% of the standardized variance.",
                    f"PC1 {100 * v[0]:.2f}% · PC2 {100 * v[1]:.2f}% · clustering stays in 5-D"))
    return out[:4]


def _walkthrough(s):
    UI.section("Guided walkthrough", "Step through the analysis; each step opens a live page.")
    n = len(S.DEMO_STEPS)
    i = int(s.get("demo_step", 0)) % n
    title, page, text = S.DEMO_STEPS[i]
    done = S.demo_step_done(i)
    state = "" if done is None else (" · done" if done else " · not yet done")
    UI.callout(f"<b>Step {i + 1} of {n} · {UI.esc(title)}</b><br>{UI.esc(text)}"
               f"<br><span style='color:var(--muted);font-size:.8rem'>Opens {UI.esc(page)}{state}</span>",
               tone="gold", html=True)
    c1, c2, c3 = st.columns(3)
    c1.button("◀ Previous step", disabled=i == 0, width="stretch", on_click=_step, args=(-1,))
    c2.button("Next step ▶", type="primary", disabled=i == n - 1, width="stretch",
              on_click=_step, args=(1,))
    c3.button(f"Go to {page}", width="stretch", on_click=S.goto, args=(page,))


def render():
    s = st.session_state
    UI.hero("AUTOMOTIVE FLEET PROFILING", "Efficiency Clustering & Vehicle Archetype Intelligence",
            "Assessment 10 · Question 4 · Machine Learning Lab",
            "Explore vehicle structure, fuel efficiency, and cluster behavior using "
            "standardized feature-space analysis.")
    stages = S.stages(s)
    nxt = _next_stage(stages)
    a, b, c = st.columns([1, 1, 1.4])
    a.button("Explore Dataset", icon=":material/table_view:", width="stretch",
             type="secondary" if S.has_raw(s) else "primary", on_click=S.goto, args=("Dataset",))
    b.button("Open Clustering Lab", icon=":material/bubble_chart:", width="stretch",
             type="primary" if S.has_std(s) else "secondary", on_click=S.goto, args=("Clustering Lab",))
    if nxt is not None:
        c.button(f"Next: {nxt['label']} — {nxt['detail']}", icon=":material/arrow_forward:",
                 width="stretch", on_click=S.goto, args=(nxt["page"],))

    UI.section("Analysis pipeline", "States reflect computed results only — opening a page never completes a stage.",
               eyebrow="Data → Clean → Scale → Explore → Cluster → Evaluate → PCA → Archetypes")
    UI.pipeline(stages)

    if not S.has_raw(s):
        UI.empty_state("DATASET NOT READY", "Load or upload an Auto MPG dataset to begin.",
                       "Dataset → Use Project Dataset")
        UI.viva("The dashboard mirrors Assessment 10: cylinder-group median imputation, z-score scaling "
                "of five continuous features, then K-Means (random_state=42, n_init=10) and Ward linkage "
                "over K=2–6. Nothing is hard-coded — every number is computed from the active dataset.")
        _walkthrough(s)
        return

    df = s["df_raw"]
    rep = s.get("validation") or {}
    mb, ma = s.get("miss_before"), s.get("miss_after")
    hp = f"{mb} → {ma}" if S.has_clean(s) else f"{mb} → pending"
    m_title, m_sub = "Not fitted", "Run clustering in the Clustering Lab"
    m = s["method"] if S.active_model(s) is not None else (S.fitted_methods(s) or [None])[0]
    act = S.model(m, s) if m else None
    if act is not None:
        m_title = f"K={act['k']}"
        m_sub = f"{S.METHOD_TITLES[m]} · silhouette {act['silhouette']:.4f} · Davies–Bouldin {act['db']:.4f}"
    UI.section("Dataset snapshot", eyebrow="Current session")
    UI.kpis([
        {"label": "Vehicles", "value": f"{len(df):,}",
         "sub": f"Current dataset · {df.shape[1]} source attributes"},
        {"label": "Clustering features", "value": str(len(CONTINUOUS_COLS)),
         "sub": "mpg · displacement · horsepower · weight · acceleration"},
        {"label": "Horsepower missing", "value": hp, "sub": "before → after cylinder-group median imputation"},
        {"label": "Validation", "value": "Validated" if rep.get("ok") else "—",
         "sub": f"{len(rep.get('warnings', []))} warning(s) · {rep.get('duplicates', 0)} duplicate row(s)",
         "tone": "success" if rep.get("ok") else None},
        {"label": "Model", "value": m_title, "sub": m_sub, "tone": "success" if act is not None else None},
    ])
    src = "Project dataset" if S.is_project_data(s) else "Uploaded file"
    UI.provenance(f"{src} · {s.get('dataset_name')} · fingerprint {s.get('dataset_id')}", "Source")

    found = _findings(s)
    if found:
        UI.section("Findings", "Generated from the current data and fitted models; each card shows its evidence.")
        cols = st.columns(2)
        for j, (kind, title, text, ev) in enumerate(found):
            with cols[j % 2]:
                UI.insight(title, text, ev, kind)
    _walkthrough(s)
