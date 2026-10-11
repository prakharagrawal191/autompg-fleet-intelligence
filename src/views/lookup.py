"""Vehicle Lookup - find a vehicle, its cluster, and the distances behind it."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import chart_factory as CF
from src import state as S
from src import theme as T
from src import ui_components as UI
from src.auto_mpg_utils import CONTINUOUS_COLS

UNIT = {"mpg": "MPG", "displacement": "cu in", "horsepower": "hp", "weight": "lb", "acceleration": "s"}


def _year(v):
    """UCI files store two-digit years (70 → 1970); uploads may already hold four digits."""
    if pd.isna(v):
        return "—"
    y = int(v)
    return f"19{y:02d}" if y < 100 else str(y)


def _count(v):
    return "—" if pd.isna(v) else f"{v:.0f}"


def _labels(df):
    """Unique option labels; 56 Auto MPG names repeat, so the year, cylinders and MPG disambiguate."""
    def one(r):
        return f"{r['car_name']} · {_year(r['model_year'])} · {_count(r['cylinders'])} cyl · {r['mpg']:.1f} mpg"
    base = [one(r) for _, r in df.iterrows()]
    seen = pd.Series(base).duplicated(keep=False).to_numpy()
    return [f"{b} · row {i + 1}" if dup else b for i, (b, dup) in enumerate(zip(base, seen))]


def _pct(v, base):
    return "n/a" if abs(base) < 1e-9 else f"{100 * (v - base) / base:+.1f}%"


def _clamp(key, options):
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def _profile(s, pos, row, imputed, fitted):
    chips = []
    for m in fitted:
        r = S.model(m, s)
        c = int(r["labels"][pos])
        name = s["arch"].get(c, "") if (m == S.KMEANS and s.get("arch")) else ""
        chips.append(f"<span class='ui-section-sub'>{S.METHOD_TITLES[m]}</span> {UI.chip(c, name)}")
    UI.md(f"<div class='ui-section'><div class='ui-section-title' role='heading' aria-level='2'>"
          f"{UI.esc(row['car_name'])}</div><div class='ui-prov'>{' '.join(chips)}</div></div>")
    hp = f"{row['horsepower']:.0f} hp" + (" (imputed)" if imputed else "")
    UI.spec_grid([("MPG", f"{row['mpg']:.1f}"), ("Cylinders", _count(row["cylinders"])),
                  ("Displacement", f"{row['displacement']:.1f} cu in"), ("Horsepower", hp),
                  ("Weight", f"{row['weight']:,.0f} lb"), ("Acceleration", f"{row['acceleration']:.1f} s"),
                  ("Model year", _year(row["model_year"])), ("Origin code", _count(row["origin"]))])
    if len(fitted) > 1:
        st.caption("Each method numbers its clusters independently, so the same number need not mean the same "
                   "vehicles.")
    if imputed:
        group = s["medians"].get(row["cylinders"]) if pd.notna(row["cylinders"]) else None
        st.caption("Horsepower was missing in the source file; the value shown is "
                   + ("the cylinder-group median." if group is not None and pd.notna(group)
                      else "the global-median fallback (no cylinder-group median applied)."))


def _comparison(s, pos, row, method):
    df = s["df_clean"]
    fleet = df[CONTINUOUS_COLS].mean()
    r = S.model(method, s)
    c = int(r["labels"][pos])
    center = S.centers_orig(method, s)[c]
    kind = "K-Means centroid" if method == S.KMEANS else "Ward cluster mean"
    rows = [{"Feature": f"{f} ({UNIT[f]})", "Vehicle": row[f], "Fleet mean": fleet[f], "Δ vs fleet": _pct(row[f], fleet[f]),
             f"{kind} (Cluster {c})": center[j], "Δ vs cluster": _pct(row[f], center[j])}
            for j, f in enumerate(CONTINUOUS_COLS)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        col: st.column_config.NumberColumn(format="%.2f")
        for col in ["Vehicle", "Fleet mean", f"{kind} (Cluster {c})"]})
    st.caption("Δ columns are percentage differences relative to the named baseline (fleet mean of the cleaned "
               "data, or the vehicle's own cluster center).")


def _distances(s, pos, method):
    r = S.model(method, s)
    z = S.X(s)[pos]
    d = np.linalg.norm(S.centers_std(method, s) - z, axis=1)
    k, lab = r["k"], int(r["labels"][pos])
    order = np.argsort(d)
    fig = go.Figure(go.Bar(x=d, y=[T.cluster_label(i) for i in range(k)], orientation="h",
                           marker=dict(color=[T.cluster_color(i) for i in range(k)],
                                       line=dict(width=[3 if i == lab else 0 for i in range(k)], color=T.WHITE)),
                           text=[f"{v:.3f}" for v in d], textposition="outside", cliponaxis=False,
                           textfont=dict(color=T.TEXT_2), hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig = CF.style(fig, 80 + 40 * k, "Distance to each cluster center", "Standardized 5-D space · outlined = assigned",
                   "Euclidean distance (standardized units)", None, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, float(d.max()) * 1.2])
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        CF.show(fig, alt=f"Bar chart of the vehicle's distance to each of the {k} cluster centers.")
    with b:
        if method == S.KMEANS:
            nearest = int(order[0])
            text = (f"K-Means assigns each vehicle to the nearest centroid in the 5-D standardized space. This "
                    f"vehicle is {d[lab]:.3f} from Cluster {lab}'s centroid")
            text += (f" and {d[order[1]]:.3f} from the next nearest (Cluster {int(order[1])}), a margin of "
                     f"{d[order[1]] - d[lab]:.3f}." if nearest == lab and k > 1 else
                     f"; the nearest centroid is Cluster {nearest} ({d[nearest]:.3f}), so the label and nearest "
                     "centroid disagree for this vehicle.")
        else:
            text = (f"Ward assigns clusters by hierarchical merging, not by nearest center. The vehicle is in Cluster "
                    f"{lab}; distances to Ward cluster means are shown for reference only.")
        UI.callout(text, title="Why this cluster?", tone="teal")


def _compare(s, df, labels):
    UI.section("Compare vehicles", "Three vehicles side by side — display only, never touches the model.")
    opts = list(range(len(df)))
    cols = st.columns(3)
    picks = []
    for col, key, label, default in zip(cols, ("lookup_va", "lookup_vb", "lookup_vc"),
                                        ("Vehicle A", "Vehicle B", "Vehicle C"), (0, 1, 2)):
        if st.session_state.get(key) not in opts:
            st.session_state[key] = min(default, len(opts) - 1)
        picks.append(col.selectbox(label, opts, key=key, format_func=lambda i: labels[i]))
    rows = {}
    for tag, i in zip("ABC", picks):
        r = df.iloc[i]
        entry = {f"{f} ({UNIT[f]})": f"{r[f]:,.2f}" for f in CONTINUOUS_COLS}
        entry["cylinders"] = f"{r['cylinders']:.0f}"
        entry["model year"] = _year(r["model_year"])
        for m in S.fitted_methods(s):
            c = int(S.model(m, s)["labels"][i])
            name = s["arch"].get(c, "") if (m == S.KMEANS and s.get("arch")) else ""
            entry[f"{S.METHOD_TITLES[m]} cluster"] = f"{c}" + (f" · {name}" if name else "")
        rows[f"{tag}: {r['car_name']}"] = entry
    st.dataframe(pd.DataFrame(rows), width="stretch")


def render():
    s = st.session_state
    UI.page_header("Vehicle Lookup", "Find a vehicle, see its cluster, and the distances that put it there.",
                   ("Automotive Fleet", "Interpretation", "Vehicle Lookup"))
    if not S.has_std(s):
        UI.empty_state("PREPROCESSING INCOMPLETE", "Lookup explains cluster membership in the standardized space.",
                       "Preprocessing → Apply imputation → Standardize")
        st.button("Open Preprocessing", on_click=S.goto, args=("Preprocessing",))
        return
    fitted = S.fitted_methods(s)
    if not fitted:
        UI.empty_state("NO CLUSTER MODEL", "Run clustering so vehicles can be mapped to clusters.",
                       "Clustering Lab → RUN CLUSTERING")
        st.button("Open Clustering Lab", type="primary", on_click=S.goto, args=("Clustering Lab",))
        return
    df = s["df_clean"]
    method = s["method"] if s["method"] in fitted else fitted[0]
    r = S.model(method, s)
    labels = _labels(df)
    a, b, c = st.columns([2, 1.2, 1.4], vertical_alignment="bottom")
    q = a.text_input("Search vehicle name", key="lookup_q", placeholder="e.g. accord, pinto")
    cyl = b.multiselect("Cylinders", sorted(int(v) for v in df["cylinders"].dropna().unique()), key="lookup_cyl")
    cl = c.multiselect(f"{S.METHOD_TITLES[method]} cluster", list(range(r["k"])), format_func=T.cluster_label,
                       key=f"lookup_cl_{method}_{r['k']}")
    mask = np.ones(len(df), dtype=bool)
    if q:
        mask &= df["car_name"].astype(str).str.contains(q, case=False, na=False, regex=False).to_numpy()
    if cyl:
        mask &= df["cylinders"].isin(cyl).to_numpy()
    if cl:
        mask &= np.isin(r["labels"], cl)
    matches = [int(i) for i in np.flatnonzero(mask)]
    st.caption(f"{len(matches)} matching vehicle(s) of {len(df)}. Names can repeat, so each option shows year, "
               "cylinders and MPG.")
    if not matches:
        UI.empty_state("NO MATCHES", "No vehicle matches the search and filters.",
                       "Clear the search, cylinder or cluster filters")
    else:
        _clamp("lookup_car", matches)
        pos = st.selectbox("Vehicle", matches, key="lookup_car", format_func=lambda i: labels[i])
        row = df.iloc[pos]
        imputed = bool(pd.isna(pd.to_numeric(s["df_raw"]["horsepower"], errors="coerce").iloc[pos]))
        _profile(s, pos, row, imputed, fitted)
        UI.section("How it compares", f"Against the fleet mean and its {S.METHOD_TITLES[method]} cluster center.")
        _comparison(s, pos, row, method)
        _distances(s, pos, method)
    _compare(s, df, labels)
    UI.viva("Lookup never refits anything: it reads the vehicle's standardized vector and measures its Euclidean "
            "distance to each fitted center — the same space K-Means uses to assign clusters.")
