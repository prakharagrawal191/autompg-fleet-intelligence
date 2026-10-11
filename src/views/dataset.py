"""Dataset - import, validate and inspect the active data."""
import io

import plotly.graph_objects as go
import streamlit as st

from src import chart_factory as CF
from src import state as S
from src import theme as T
from src import ui_components as UI


def _load_upload():
    s = st.session_state
    f = s.get("upload_file")
    if f is None:
        return
    rep = S.load_from_source(f, "upload", f.name, s.get("upload_format", "auto"))
    if rep["ok"]:
        s["flash"] = f"Loaded {f.name} · {rep['rows']} rows"


def _use_project():
    rep = S.load_project()
    if rep["ok"]:
        st.session_state["flash"] = f"Project dataset loaded · {rep['rows']} rows"


def _dismiss():
    st.session_state["last_report"] = None


def _issue_list(items):
    return "<ul>" + "".join(
        f"<li><b>{UI.esc(i['what'])}</b><br>{UI.esc(i['why'])}"
        f"<br><span class='ui-next'>Next step: {UI.esc(i['next'])}</span></li>" for i in items) + "</ul>"


def _import_area():
    left, right = st.columns([1.6, 1], gap="large")
    with left.container(border=True):
        UI.section("Import Auto MPG data", "The uploader is the drop zone — drag a file onto it or browse.")
        st.file_uploader("Drop a supported dataset here", type=["csv", "data", "txt"], key="upload_file",
                         max_upload_size=5,
                         help="UCI whitespace .data (car names in quotes) or a comma-separated CSV with the nine "
                              "Auto MPG columns in standard order; .txt is read as either. '?' means missing.")
        a, b = st.columns([1, 1], vertical_alignment="bottom")
        a.selectbox("File format", ["auto", "csv", "data"], key="upload_format",
                    help="auto detects comma-separated vs whitespace-separated text.")
        b.button("Load Uploaded File", type="primary", width="stretch", icon=":material/upload_file:",
                 disabled=st.session_state.get("upload_file") is None, on_click=_load_upload)
    with right.container(border=True):
        UI.section("Use the project dataset", "auto-mpg.data — the UCI Auto MPG file bundled with this project.")
        st.button("Use Project Dataset", width="stretch", icon=":material/database:", on_click=_use_project,
                  type="secondary" if S.has_raw() else "primary")


def _failure(rep):
    UI.callout(_issue_list(rep["errors"]), title="DATASET VALIDATION FAILED", tone="danger", html=True)
    if rep.get("detail"):
        with st.expander("Parser detail"):
            st.code(rep["detail"], language="text")
    a, b = st.columns([3, 1], vertical_alignment="center")
    a.caption("The previously loaded dataset is still active." if S.has_raw()
              else "No dataset is active; nothing was loaded.")
    b.button("Dismiss", width="stretch", on_click=_dismiss)


def _validated(df, rep):
    checks = [
        f"Required columns present ({len(df.columns) - len(rep['extra_columns'])} of 9)",
        f"Row count within limits ({rep['rows']:,}; allowed {S.MIN_ROWS}–{S.MAX_ROWS:,})",
        "Clustering features complete: " + ", ".join(S.FEATURES_COMPLETE),
    ] + rep["notes"]
    body = (f"<b>{rep['rows']:,} records · {rep['cols']} fields</b>"
            + "<ul>" + "".join(f"<li>{UI.esc(c)}</li>" for c in checks) + "</ul>")
    UI.callout(body, title="DATASET VALIDATED", tone="success", html=True)
    if rep["warnings"]:
        UI.callout(_issue_list(rep["warnings"]), title="WARNINGS", tone="warning", html=True)


def _quality(df, rep):
    cells = df.size
    missing = int(df.isna().sum().sum())
    UI.section("Data quality", "Each measure is shown separately — there is no composite quality score.")
    UI.kpis([
        {"label": "Records", "value": f"{len(df):,}", "sub": "rows in the active file"},
        {"label": "Fields", "value": str(df.shape[1]), "sub": "source columns"},
        {"label": "Horsepower missing", "value": str(int(df["horsepower"].isna().sum())),
         "sub": "imputed on Preprocessing"},
        {"label": "Duplicate rows", "value": str(rep["duplicates"]), "sub": "kept as separate vehicles"},
        {"label": "Completeness", "value": f"{100 * (1 - missing / max(cells, 1)):.2f}%",
         "sub": f"non-missing cells / {len(df):,} × {df.shape[1]} cells"},
    ])
    miss = df.isna().sum()
    fig = go.Figure(go.Bar(x=[str(c) for c in miss.index], y=miss.values, marker=dict(color=T.TEAL),
                           text=[str(int(v)) for v in miss.values], textposition="outside", cliponaxis=False,
                           textfont=dict(color=T.TEXT_2), hovertemplate="%{x}: %{y} missing<extra></extra>"))
    fig = CF.style(fig, 260, "Missing values by column", "As loaded, before imputation", None, "Missing values",
                   legend=False)
    fig.update_yaxes(range=[0, max(1, int(miss.max())) * 1.3])
    CF.show(fig, alt="Bar chart of missing values per column in the loaded dataset.")


def _preview(s, df):
    UI.section("Records", "Search by vehicle name or filter by cylinder count.")
    a, b, c = st.columns([2, 1.4, 1.2], vertical_alignment="bottom")
    q = a.text_input("Search vehicle name", key="ds_search", placeholder="e.g. honda, ford pinto")
    cyl = b.multiselect("Cylinders", sorted(int(v) for v in df["cylinders"].dropna().unique()), key="ds_cyl")
    fitted = S.fitted_methods(s)
    with_labels = c.toggle("Show cluster labels", key="ds_labels", disabled=not fitted,
                           help="Adds the labels of models fitted on this dataset." if fitted
                           else "Fit a model in the Clustering Lab to add labels.")
    view = df.copy()
    if with_labels:
        for m in fitted:
            view[f"{S.METHOD_TITLES[m]} cluster (K={S.model(m, s)['k']})"] = S.model(m, s)["labels"]
    if q:
        view = view[view["car_name"].astype(str).str.contains(q, case=False, na=False, regex=False)]
    if cyl:
        view = view[view["cylinders"].isin(cyl)]
    st.caption(f"Showing {len(view):,} of {len(df):,} rows · empty cells are missing values.")
    cols = ["car_name"] + [c for c in view.columns if c != "car_name"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=380, column_config={
        "car_name": st.column_config.TextColumn("Vehicle", pinned=True),
        "mpg": st.column_config.NumberColumn("MPG", format="%.1f"),
        "cylinders": st.column_config.NumberColumn("Cylinders", format="%d"),
        "displacement": st.column_config.NumberColumn("Displacement (cu in)", format="%.1f"),
        "horsepower": st.column_config.NumberColumn("Horsepower (hp)", format="%.0f"),
        "weight": st.column_config.NumberColumn("Weight (lb)", format="%.0f"),
        "acceleration": st.column_config.NumberColumn("Acceleration (s)", format="%.1f"),
        "model_year": st.column_config.NumberColumn("Model year", format="%d"),
        "origin": st.column_config.NumberColumn("Origin code", format="%d")})


def render():
    s = st.session_state
    UI.page_header("Dataset", "Import an Auto MPG file, validate it, and inspect the records.",
                   ("Automotive Fleet", "Analysis", "Dataset"))
    _import_area()
    rep = s.get("last_report")
    if rep is not None and not rep.get("ok"):
        _failure(rep)
    if not S.has_raw(s):
        UI.empty_state("NO DATASET", "Load or upload an Auto MPG dataset to begin.",
                       "Use Project Dataset, or drop a file above")
        return
    df, rep = s["df_raw"], s["validation"]
    src = "Project dataset" if S.is_project_data(s) else "Uploaded file"
    UI.provenance(f"{src} · {s['dataset_name']} · fingerprint {s['dataset_id']}", "Active")
    _validated(df, rep)
    _quality(df, rep)
    _preview(s, df)
    with st.expander("Technical details — types, missing values, memory", key="tech_dataset"):
        info = df.dtypes.astype(str).rename("dtype").to_frame()
        info["missing"] = df.isna().sum()
        st.dataframe(info, width="stretch")
        buf = io.StringIO()
        df.info(buf=buf)
        st.code(buf.getvalue(), language="text")
    UI.viva("'?' marks unknown horsepower in the UCI file, so it is parsed as missing rather than zero. "
            "Validation blocks files whose other clustering features are incomplete, because the validated "
            "pipeline imputes horsepower only.")
