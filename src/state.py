"""Session state, central invalidation and pipeline actions (no presentation).

Every action takes the state mapping ``s`` (``st.session_state`` in the app, a
plain dict in tests), so the UI and the regression checks run the same code.
All science comes from src/auto_mpg_utils.py, which this module never alters.
"""
import hashlib
import io
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.auto_mpg_utils import (
    COLUMN_NAMES, CONTINUOUS_COLS, DEFAULT_N_INIT, DEFAULT_RANDOM_STATE, K_RANGE,
    assign_archetypes, build_dendrogram, calculate_centroids, evaluate_kmeans,
    evaluate_kmeans_range, extract_dendrogram_clusters, fit_ward,
    impute_horsepower_by_cylinder, load_auto_mpg, run_pca, standardize_features,
    validate_schema,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_FILE = os.path.join(ROOT, "auto-mpg.data")
KMEANS, WARD = "K-Means", "Ward"
METHOD_TITLES = {KMEANS: "K-Means", WARD: "Ward hierarchical"}
# Assessment 10 notebook: K=3 is the documented joint engineering choice for the
# project dataset (silhouette peaks at K=2). Shown only for that dataset.
DOCUMENTED_K = 3
MIN_ROWS, MAX_ROWS = 10, 5000      # input safety: K<=6 needs rows; Ward/silhouette are O(n^2)
MAX_BYTES = 5 * 1024 * 1024
FEATURES_COMPLETE = ["mpg", "displacement", "weight", "acceleration"]  # only horsepower is imputed

# Analysis state - cleared when a dataset loads or the analysis resets.
ANALYSIS = {
    "df_raw": None, "df_clean": None, "df_std": None, "scaler": None,
    "medians": None, "miss_before": None, "miss_after": None, "fallback_used": False,
    "dataset_id": None, "dataset_source": None, "dataset_name": None, "validation": None,
    "kmeans": None, "ward": None, "krange": None, "dendro": None, "dlab": None, "dk": None,
    "pca": None, "cent_std": None, "cent_orig": None, "arch": None, "arch_rule": None,
    "selected_k": 3, "k_select": 3, "method": KMEANS, "std_done": False, "imp_done": False,
}
# Selections bound to results (several are widget keys; only assigned in
# callbacks or before their widget renders).
SELECTIONS = {
    "corr_pair": "weight|mpg", "eda_feat": "mpg", "eda_viz": "Histogram",
    "rel_x": "weight", "rel_y": "mpg", "eval_metric": "Silhouette", "pca_color": KMEANS,
    "arch_pick": 0, "why": 0, "arch_compare": [], "cmp_a": 0, "cmp_b": 1,
    "lookup_car": None, "lookup_va": None, "lookup_vb": None, "lookup_vc": None,
    "km_rs": DEFAULT_RANDOM_STATE, "km_ni": DEFAULT_N_INIT,
}
# Navigation, presentation preferences and messages - survive a reset.
UI_STATE = {
    "page": "Overview", "demo_mode": False, "demo_step": 0, "presentation_mode": False,
    "viva_mode": False, "pca_size": 7, "pca_labels": False, "pca_centroids": True,
    "dendro_detail": False, "flash": None, "last_report": None,
}
_MODEL_KEYS = ["kmeans", "ward", "dlab", "dk", "pca", "cent_std", "cent_orig", "arch", "arch_rule"]
_RESULT_PICKS = {"arch_pick": 0, "why": 0, "arch_compare": [], "cmp_a": 0, "cmp_b": 1}
# Keys bound to widgets that only render on one page. Streamlit drops a widget's value
# at the end of any run where the widget is absent; re-assigning keeps it for the session.
_WIDGET_KEYS = list(SELECTIONS) + ["k_select", "pca_size", "pca_labels", "pca_centroids", "dendro_detail"]


def _s(s):
    return st.session_state if s is None else s


def init(s=None):
    s = _s(s)
    for group in (ANALYSIS, SELECTIONS, UI_STATE):
        for k, v in group.items():
            s.setdefault(k, list(v) if isinstance(v, list) else v)
    for k in _WIDGET_KEYS:
        s[k] = s[k]


# ---- predicates -----------------------------------------------------------
def has_raw(s=None):
    return _s(s).get("df_raw") is not None


def has_clean(s=None):
    return _s(s).get("df_clean") is not None


def has_std(s=None):
    return _s(s).get("df_std") is not None


def model(method, s=None):
    """Fitted result for ``method`` - only if it belongs to the active dataset."""
    s = _s(s)
    r = s.get("kmeans" if method == KMEANS else "ward")
    if r is None or r.get("dataset_id") != s.get("dataset_id"):
        return None
    return r


def has_km(s=None):
    return model(KMEANS, s) is not None


def has_ward(s=None):
    return model(WARD, s) is not None


def has_model(s=None):
    return has_km(s) or has_ward(s)


def active_model(s=None):
    s = _s(s)
    return model(s.get("method", KMEANS), s)


def fitted_methods(s=None):
    return [m for m in (KMEANS, WARD) if model(m, s) is not None]


def X(s=None):
    """Standardized five-feature matrix used by every clustering call."""
    return _s(s)["df_std"][CONTINUOUS_COLS].to_numpy()


def is_project_data(s=None):
    return _s(s).get("dataset_source") == "project"


# ---- invalidation (the only place downstream state is cleared) -------------
def _clear(s, keys):
    for k in keys:
        v = ANALYSIS.get(k, SELECTIONS.get(k))
        s[k] = list(v) if isinstance(v, list) else v


def invalidate_models(s=None):
    """K changed or features re-standardized: every fitted artefact is stale.

    The K=2-6 sweep and the Ward linkage do not depend on the selected K, so a
    K change keeps them; re-standardization clears them via invalidate_features.
    """
    s = _s(s)
    _clear(s, _MODEL_KEYS)
    s.update({k: (list(v) if isinstance(v, list) else v) for k, v in _RESULT_PICKS.items()})
    s["cmp_b"] = min(1, int(s.get("selected_k", 3)) - 1)


def invalidate_features(s=None):
    s = _s(s)
    invalidate_models(s)
    _clear(s, ["krange", "dendro"])


def invalidate_kmeans(s=None):
    """K-Means settings changed: its fit, sweep and centroid artefacts are stale."""
    s = _s(s)
    _clear(s, ["kmeans", "krange", "cent_std", "cent_orig", "arch", "arch_rule"])
    s.update({k: (list(v) if isinstance(v, list) else v) for k, v in _RESULT_PICKS.items()})


def reset_analysis(s=None, keep_page=True):
    s = _s(s)
    page = s.get("page", "Overview")
    _clear(s, list(ANALYSIS) + list(SELECTIONS))
    s["last_report"] = None
    if keep_page:
        s["page"] = page
    _sweep.clear()
    _pca.clear()


# ---- dataset loading and validation ---------------------------------------
def fingerprint(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]


def _issue(what, why, next_step):
    return {"what": what, "why": why, "next": next_step}


def _read(source):
    if hasattr(source, "getvalue"):
        data = source.getvalue()
    elif hasattr(source, "read"):
        data = source.read()
    else:
        with open(source, "rb") as f:
            data = f.read()
    return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)


def validate_dataset(df, fmt="data", header=None):
    """Structured validation report (never raises). ``ok`` gates loading."""
    rep = {"ok": False, "format": fmt, "rows": int(df.shape[0]), "cols": int(df.shape[1]),
           "missing_required": [], "extra_columns": [], "missing_by_col": {},
           "duplicates": 0, "errors": [], "warnings": [], "notes": []}
    if fmt == "csv" and header is not None and list(header) != COLUMN_NAMES:
        if sorted(header) == sorted(COLUMN_NAMES):
            rep["errors"].append(_issue(
                "Columns are in a non-standard order.",
                "The CSV parser maps the nine Auto MPG fields by position, so a reordered "
                "header would silently mislabel features.",
                "Save the file with columns ordered: " + ", ".join(COLUMN_NAMES) + "."))
            return df, rep  # later checks would only describe the mislabelled columns
        if len(header) == len(COLUMN_NAMES):
            rep["notes"].append("Header names differ from the standard names; the nine columns "
                                "were mapped by position (" + ", ".join(map(str, header)) + ").")
    df, missing = validate_schema(df)
    rep["missing_required"] = list(missing)
    rep["extra_columns"] = [str(c) for c in df.columns if c not in COLUMN_NAMES]
    rep["missing_by_col"] = {str(c): int(df[c].isna().sum()) for c in df.columns}
    rep["duplicates"] = int(df.duplicated().sum())
    if missing:
        rep["errors"].append(_issue(
            "Required columns are missing: " + ", ".join(missing) + ".",
            "The clustering pipeline cannot create its expected feature matrix.",
            "Upload a dataset containing the nine Auto MPG fields: " + ", ".join(COLUMN_NAMES) + "."))
        return df, rep
    n = rep["rows"]
    if n < MIN_ROWS:
        rep["errors"].append(_issue(
            f"Only {n} rows were found (minimum {MIN_ROWS}).",
            "K=2–6 clustering and silhouette analysis need more vehicles than clusters.",
            "Upload the full dataset."))
    if n > MAX_ROWS:
        rep["errors"].append(_issue(
            f"{n:,} rows exceed the {MAX_ROWS:,}-row limit.",
            "Ward linkage and silhouette scale quadratically with rows; this dashboard is sized "
            "for fleets of hundreds of vehicles.",
            f"Upload a sample of at most {MAX_ROWS:,} rows."))
    distinct = int((~df[CONTINUOUS_COLS].duplicated()).sum())
    if MIN_ROWS <= n and distinct < max(K_RANGE):
        rep["errors"].append(_issue(
            f"Only {distinct} distinct vehicles by the five clustering features.",
            f"K up to {max(K_RANGE)} needs at least {max(K_RANGE)} distinct feature vectors; otherwise "
            "K-Means leaves clusters empty.",
            "Upload a dataset with more distinct vehicles (duplicates do not add new points)."))
    gaps = {c: int(df[c].isna().sum()) for c in FEATURES_COMPLETE if df[c].isna().any()}
    if gaps:
        rep["errors"].append(_issue(
            "Missing or non-numeric values in " + ", ".join(f"{c} ({v})" for c, v in gaps.items()) + ".",
            "The validated pipeline imputes only horsepower (cylinder-group median); the other "
            "clustering features must be complete numbers.",
            "Fix or remove the affected rows and upload again."))
    numeric = df[COLUMN_NAMES[:8]].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        rep["errors"].append(_issue(
            "Infinite values were found in numeric columns.",
            "Standardization and distance calculations require finite numbers.",
            "Replace the infinite values and upload again."))
    hp_missing = int(df["horsepower"].isna().sum())
    if n and hp_missing == n:
        rep["errors"].append(_issue(
            "Horsepower has no observed values.",
            "Cylinder-group median imputation needs observed horsepower in each group.",
            "Upload a dataset with recorded horsepower values."))
    elif hp_missing:
        rep["notes"].append(f"Horsepower has {hp_missing} missing value(s) ('?' or non-numeric); "
                            "they are imputed on Preprocessing with cylinder-group medians.")
        orphan = int((df["horsepower"].isna() & df["cylinders"].isna()).sum())
        if orphan:
            rep["warnings"].append(_issue(
                f"{orphan} row(s) miss both horsepower and cylinders.",
                "Their cylinder group is unknown, so the documented global-median fallback applies.",
                "Optional: supply cylinder counts for these rows."))
    other = {c: v for c, v in rep["missing_by_col"].items()
             if v and c in ("cylinders", "model_year", "origin", "car_name")}
    if other:
        rep["warnings"].append(_issue(
            "Missing values outside the clustering features: "
            + ", ".join(f"{c} ({v})" for c, v in other.items()) + ".",
            "These columns are not clustering features; affected rows still cluster.",
            "Optional: complete them for cleaner lookup and exports."))
    if rep["duplicates"]:
        rep["warnings"].append(_issue(
            f"{rep['duplicates']} duplicate row(s).",
            "Duplicates are kept and count as separate vehicles in every statistic.",
            "Optional: de-duplicate the file if repeats are unintended."))
    if rep["extra_columns"]:
        rep["warnings"].append(_issue(
            "Extra columns: " + ", ".join(rep["extra_columns"]) + ".",
            "They are ignored by the analysis but kept in exported tables.",
            "No action needed."))
    rep["ok"] = not rep["errors"]
    return df, rep


def load_from_source(source, origin, name, file_format="auto", s=None):
    """Parse, validate and (only if valid) activate a dataset. Returns the report."""
    s = _s(s)
    try:
        text = _read(source)
        if len(text.encode("utf-8", errors="ignore")) > MAX_BYTES:
            raise ValueError(f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
        if "\x00" in text[:4096]:
            raise ValueError("binary content - not a text data file")
        df, fmt = load_auto_mpg(io.StringIO(text), file_format)
        header = None
        if fmt == "csv":
            header = [str(c).strip() for c in pd.read_csv(io.StringIO(text), nrows=0).columns]
    except Exception as e:  # parser failure -> structured, user-facing report
        rep = {"ok": False, "format": None, "rows": 0, "cols": 0, "missing_required": [],
               "extra_columns": [], "missing_by_col": {}, "duplicates": 0, "warnings": [],
               "notes": [], "errors": [_issue(
                   "The file could not be read as Auto MPG data.",
                   "Without a parsed table there is nothing to validate or cluster.",
                   "Upload the UCI whitespace .data file or a CSV with the nine Auto MPG columns.")],
               "detail": f"{type(e).__name__}: {e}"}
    else:
        df, rep = validate_dataset(df, fmt, header)
    rep.update(source=origin, name=name)
    if not rep["ok"]:
        s["last_report"] = rep
        return rep
    reset_analysis(s)
    s.update(df_raw=df, dataset_id=fingerprint(df), dataset_source=origin, dataset_name=name,
             validation=rep, miss_before=int(df["horsepower"].isna().sum()))
    return rep


def load_project(s=None):
    return load_from_source(PROJECT_FILE, "project", os.path.basename(PROJECT_FILE), "auto", s)


# ---- preprocessing --------------------------------------------------------
def apply_imputation(s=None):
    s = _s(s)
    out, medians, mb, ma, fb = impute_horsepower_by_cylinder(s["df_raw"])
    s.update(df_clean=out, medians=medians, miss_before=mb, miss_after=ma,
             fallback_used=bool(fb), imp_done=True)
    _clear(s, ["df_std", "scaler", "std_done"])
    invalidate_features(s)


def apply_standardization(s=None):
    s = _s(s)
    if not has_clean(s):
        return False
    std, scaler = standardize_features(s["df_clean"])
    s.update(df_std=std, scaler=scaler, std_done=True)
    invalidate_features(s)
    return True


# ---- clustering -----------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=32)
def _sweep(Xm, dataset_id, random_state, n_init):
    return evaluate_kmeans_range(np.asarray(Xm), list(K_RANGE), random_state, n_init)


@st.cache_data(show_spinner=False, max_entries=8)
def _pca(Xm, dataset_id):
    return run_pca(np.asarray(Xm), 2)


def km_params(s=None):
    s = _s(s)
    return int(s.get("km_rs", DEFAULT_RANDOM_STATE)), int(s.get("km_ni", DEFAULT_N_INIT))


def set_k(k, s=None):
    s = _s(s)
    k = int(k)
    if k != s.get("selected_k"):
        s["selected_k"] = k
        invalidate_models(s)


def set_method(method, s=None):
    _s(s)["method"] = method


def compute_sweep(s=None):
    s = _s(s)
    s["krange"] = _sweep(X(s), s["dataset_id"], *km_params(s))


def ensure_pca(s=None):
    s = _s(s)
    if s.get("pca") is None and has_std(s):
        s["pca"] = _pca(X(s), s["dataset_id"])
    return s.get("pca")


def fit_active(s=None):
    """Fit the active method at the selected K on the standardized matrix."""
    s = _s(s)
    Xm, k, did = X(s), int(s["selected_k"]), s["dataset_id"]
    if s["method"] == KMEANS:
        rs, ni = km_params(s)
        r = evaluate_kmeans(Xm, k, rs, ni)
        r["dataset_id"] = did
        r["counts"] = r["counts"].reindex(range(k), fill_value=0)  # list empty ids explicitly
        s["kmeans"] = r
        cs, co = calculate_centroids(s["scaler"], r["centers"])
        s["cent_std"], s["cent_orig"] = cs, co
        s["arch"], s["arch_rule"] = assign_archetypes(co)
        s.update({k_: (list(v) if isinstance(v, list) else v) for k_, v in _RESULT_PICKS.items()})
        s["cmp_b"] = min(1, k - 1)
    else:
        r = fit_ward(Xm, k)
        r["dataset_id"] = did
        r["counts"] = r["counts"].reindex(range(k), fill_value=0)
        s["ward"] = r
        if s.get("dendro") is None:
            s["dendro"] = build_dendrogram(Xm)
    if s.get("krange") is None:
        compute_sweep(s)
    ensure_pca(s)
    return r


def build_dendro(s=None):
    s = _s(s)
    s["dendro"] = build_dendrogram(X(s))
    extract_dendro(s)


def extract_dendro(s=None):
    s = _s(s)
    k = int(s["selected_k"])
    s["dlab"], s["dk"] = extract_dendrogram_clusters(s["dendro"]["Z_full"], k), k


TIE = 5e-5  # metric values equal at the 4-decimal display count as a tie


def metric_leader(km, w, key, higher):
    """KMEANS or WARD for the better value of ``key``; None when tied at display precision."""
    a, b = km[key], w[key]
    if abs(a - b) < TIE:
        return None
    return KMEANS if (a > b) == higher else WARD


def cluster_means(Xm, labels, k):
    """Mean of each cluster's rows (a Ward cluster's centroid; any space)."""
    labels = np.asarray(labels)
    return np.vstack([np.asarray(Xm)[labels == i].mean(axis=0) for i in range(int(k))])


def centers_std(method, s=None):
    """Cluster centres in standardized space: K-Means centroids or Ward member means."""
    s = _s(s)
    r = model(method, s)
    if r is None:
        return None
    return np.asarray(r["centers"]) if method == KMEANS else cluster_means(X(s), r["labels"], r["k"])


def centers_orig(method, s=None):
    """Centres in original units (K-Means: scaler.inverse_transform of its centroids)."""
    s = _s(s)
    c = centers_std(method, s)
    return None if c is None else s["scaler"].inverse_transform(c)


# ---- navigation, status and pipeline stages --------------------------------
def goto(page, s=None):
    _s(s)["page"] = page


def provenance(method, s=None):
    s = _s(s)
    r = model(method, s)
    if r is None:
        return ""
    src = "project dataset" if is_project_data(s) else f"upload {s.get('dataset_name') or ''}".strip()
    text = f"{METHOD_TITLES[method]} · K={r['k']} · {len(r['labels'])} vehicles · {src}"
    if method == KMEANS and (r["random_state"], r["n_init"]) != (DEFAULT_RANDOM_STATE, DEFAULT_N_INIT):
        text += f" · random_state={r['random_state']}, n_init={r['n_init']} (non-default)"
    return text


def status(s=None):
    """(label, tone, detail) grounded in state - shown in the top bar."""
    s = _s(s)
    if not has_raw(s):
        return "NO DATASET", "neutral", "Load a dataset to begin"
    m = active_model(s)
    if m is not None:
        return "MODEL READY", "success", f"{METHOD_TITLES[s['method']]} · K={m['k']}"
    other = fitted_methods(s)
    if other:
        return "MODEL READY", "success", f"{METHOD_TITLES[other[0]]} · K={model(other[0], s)['k']}"
    if has_std(s):
        return "PREPROCESSING READY", "teal", "Standardized · ready to cluster"
    return "DATASET LOADED", "teal", f"{len(s['df_raw'])} rows · preprocessing pending"


def stages(s=None):
    """Pipeline stages with truthful states: complete / ready / available / blocked."""
    s = _s(s)
    raw, clean, std_, any_m = has_raw(s), has_clean(s), has_std(s), has_model(s)
    km = model(KMEANS, s)
    act_m = s["method"] if active_model(s) is not None else (fitted_methods(s) or [None])[0]
    act = model(act_m, s) if act_m else None

    def st_(done, ready):
        return "complete" if done else ("ready" if ready else "blocked")

    return [
        {"n": "01", "label": "Data", "page": "Dataset", "state": st_(raw, True),
         "detail": f"{len(s['df_raw'])} rows" if raw else "Load a dataset"},
        {"n": "02", "label": "Clean", "page": "Preprocessing", "state": st_(clean, raw),
         "detail": f"HP missing {s['miss_before']} → {s['miss_after']}" if clean else "Impute horsepower"},
        {"n": "03", "label": "Scale", "page": "Preprocessing", "state": st_(std_, clean),
         "detail": "5 features z-scored" if std_ else "Standardize features"},
        {"n": "04", "label": "Explore", "page": "EDA Explorer",
         "state": "available" if raw else "blocked", "detail": "Distributions, correlations"},
        {"n": "05", "label": "Cluster", "page": "Clustering Lab", "state": st_(any_m, std_),
         "detail": f"{METHOD_TITLES[act_m]} · K={act['k']}" if act else "Fit K-Means or Ward"},
        {"n": "06", "label": "Evaluate", "page": "Cluster Quality", "state": st_(any_m, False),
         "detail": f"Silhouette {act['silhouette']:.4f}" if act else "Needs a fitted model"},
        {"n": "07", "label": "PCA", "page": "PCA Explorer",
         "state": st_(s.get("pca") is not None and any_m, any_m),
         "detail": (f"{100 * float(np.sum(s['pca']['variance_ratio'])):.2f}% variance"
                    if s.get("pca") is not None and any_m else "Needs a fitted model")},
        {"n": "08", "label": "Archetypes", "page": "Fleet Archetypes",
         "state": st_(s.get("arch") is not None and km is not None, False),
         "detail": f"{len(s['arch'])} archetypes" if (s.get("arch") and km) else "Needs K-Means"},
    ]


DEMO_STEPS = [  # (title, page, instruction)
    ("Load data", "Dataset", "Use the project dataset or upload an Auto MPG file."),
    ("Clean / impute", "Preprocessing", "Apply cylinder-group median imputation to horsepower."),
    ("Standardize", "Preprocessing", "Z-score the five continuous features."),
    ("Explore", "EDA Explorer", "Inspect distributions and the weight-horsepower-MPG correlations."),
    ("Select K", "Clustering Lab", "Choose K between 2 and 6 and the algorithm."),
    ("Run clustering", "Clustering Lab", "Fit the selected algorithm at the selected K."),
    ("Evaluate models", "Cluster Quality", "Compare silhouette and Davies-Bouldin for both methods."),
    ("Visualize with PCA", "PCA Explorer", "Project the five-feature space onto PC1 and PC2."),
    ("Interpret archetypes", "Fleet Archetypes", "Read centroid profiles and representative vehicles."),
]


def demo_step_done(i, s=None):
    s = _s(s)
    checks = [has_raw, has_clean, has_std, None, None, has_model, lambda s_: has_km(s_) and has_ward(s_),
              lambda s_: s_.get("pca") is not None and has_model(s_),
              lambda s_: s_.get("arch") is not None and has_km(s_)]
    fn = checks[i]
    return None if fn is None else bool(fn(s))


# ---- test support ---------------------------------------------------------
def seed(k=3, methods=(KMEANS,), dendro=False, path=None):
    """Run the real pipeline headlessly and return the resulting state dict."""
    s = {}
    init(s)
    rep = load_from_source(path or PROJECT_FILE, "project" if path is None else "upload",
                           os.path.basename(path or PROJECT_FILE), "auto", s)
    if not rep["ok"]:
        raise ValueError(rep["errors"])
    apply_imputation(s)
    apply_standardization(s)
    set_k(k, s)
    s["k_select"] = int(k)
    for m in methods:
        set_method(m, s)
        fit_active(s)
    if dendro:
        build_dendro(s)
    set_method(methods[0] if methods else KMEANS, s)
    return s
