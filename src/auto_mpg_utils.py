"""Shared Auto MPG logic for Question 4.

Used by both the Part A notebook and the Streamlit interface (app.py),
so preprocessing is implemented once and reused everywhere.
"""
import io

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name",
]

CONTINUOUS_COLS = ["mpg", "displacement", "horsepower", "weight", "acceleration"]
CORR_COLS = ["weight", "horsepower", "mpg"]

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/auto-mpg/auto-mpg.data")


def _read_text(source):
    """Read text from a path or an uploaded file-like object."""
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        return data
    with open(source, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _looks_like_csv(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    sample = lines[:20]
    return sum(1 for ln in sample if "," in ln) >= len(sample) / 2


def parse_whitespace(text):
    """Parse the UCI raw .data format (whitespace numbers + car name)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if '"' in line:
            parts = line.split('"')
            numbers = parts[0].split()
            car = parts[1].strip() if len(parts) > 1 else ""
        else:
            tokens = line.split()
            numbers, car = tokens[:8], " ".join(tokens[8:])
        rows.append(numbers + [car])
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    for col in COLUMN_NAMES[:8]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # '?' -> NaN
    df["car_name"] = df["car_name"].astype(str).str.strip().str.strip('"')
    return df


def parse_csv(text):
    """Parse a comma-separated Auto MPG file ('?' treated as missing)."""
    df = pd.read_csv(io.StringIO(text), na_values=["?", "NA", "", " "])
    df.columns = [str(c).strip() for c in df.columns]
    if list(df.columns) != COLUMN_NAMES and df.shape[1] == len(COLUMN_NAMES):
        df.columns = COLUMN_NAMES
    return df


def load_auto_mpg(source, file_format="auto"):
    """Load the Auto MPG dataset from a path or uploaded file.

    file_format: 'auto' (detect), 'csv', or 'data' (whitespace .data).
    Returns (dataframe, detected_format).
    """
    text = _read_text(source)
    if file_format == "csv" or (file_format == "auto" and _looks_like_csv(text)):
        return parse_csv(text), "csv"
    return parse_whitespace(text), "data"


def clean_horsepower(df):
    """Return a copy with horsepower converted to numeric ('?' -> NaN)."""
    out = df.copy()
    out["horsepower"] = pd.to_numeric(out["horsepower"], errors="coerce")
    return out


def impute_horsepower_by_cylinder(df):
    """Impute missing horsepower with the median of its cylinder group.

    Returns (imputed_df, medians, missing_before, missing_after, fallback_used).
    A global-median fallback is documented and only used when a whole
    cylinder group has no observed horsepower.
    """
    out = clean_horsepower(df)
    missing_before = int(out["horsepower"].isna().sum())
    medians = out.groupby("cylinders")["horsepower"].median().sort_index()
    out["horsepower"] = (
        out.groupby("cylinders")["horsepower"]
           .transform(lambda x: x.fillna(x.median()))
    )
    fallback_used = False
    if int(out["horsepower"].isna().sum()) > 0:
        out["horsepower"] = out["horsepower"].fillna(out["horsepower"].median())
        fallback_used = True
    missing_after = int(out["horsepower"].isna().sum())
    return out, medians, missing_before, missing_after, fallback_used


def distribution_summary(df):
    """Descriptive stats + skewness for weight, horsepower and mpg."""
    cols = ["weight", "horsepower", "mpg"]
    return df[cols].describe().T, df[cols].skew()


def calculate_correlations(df):
    """Correlation matrix for weight, horsepower and mpg."""
    return df[CORR_COLS].corr()


def strength_label(abs_r):
    """Strength convention used for the interpretation table."""
    if abs_r >= 0.7:
        return "strong"
    if abs_r >= 0.4:
        return "moderate"
    if abs_r >= 0.2:
        return "weak"
    return "very weak / negligible"


def correlation_table(df):
    """Data-based interpretation table (no hard-coded values)."""
    rows = []
    for x, y in [("weight", "mpg"), ("horsepower", "mpg"),
                 ("weight", "horsepower")]:
        r = float(df[x].corr(df[y]))
        direction = "positive" if r > 0 else ("negative" if r < 0 else "near zero")
        rows.append({"Feature 1": x, "Feature 2": y,
                     "Correlation": round(r, 4), "Direction": direction,
                     "Strength": strength_label(abs(r))})
    return pd.DataFrame(rows)


def standardize_features(df, columns=None):
    """Z-score standardization; returns (standardized_df, scaler)."""
    columns = columns or CONTINUOUS_COLS
    scaler = StandardScaler()
    out = df.copy()
    out[columns] = scaler.fit_transform(df[columns])
    return out, scaler



# ---- Extended analytical helpers (Parts B / C) ----
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as sch_linkage

REQUIRED_COLUMNS = list(COLUMN_NAMES)
K_RANGE = [2, 3, 4, 5, 6]
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_INIT = 10


def coerce_numerics(df):
    """Coerce the 8 numeric Auto MPG columns; '?' / junk -> NaN."""
    out = df.copy()
    for col in COLUMN_NAMES[:8]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_schema(df):
    """Check required columns. Returns (df_coerced, missing_list)."""
    out = coerce_numerics(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    return out, missing


def cluster_counts(labels):
    return pd.Series(np.asarray(labels)).value_counts().sort_index()


def standardized_statistics(df_standardized, columns=None):
    """Mean / sample STD / population STD per standardized column."""
    columns = columns or CONTINUOUS_COLS
    return pd.DataFrame({
        "Mean": df_standardized[columns].mean(),
        "Sample_STD": df_standardized[columns].std(),
        "Population_STD": df_standardized[columns].std(ddof=0),
    })


def evaluate_clustering(X, labels):
    """Silhouette (higher better) + Davies-Bouldin (lower better)."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(labels) <= len(np.unique(labels)):
        return float("nan"), float("nan")
    return float(silhouette_score(X, labels)), float(davies_bouldin_score(X, labels))


def evaluate_kmeans(X, k, random_state=DEFAULT_RANDOM_STATE,
                    n_init=DEFAULT_N_INIT):
    model = KMeans(n_clusters=int(k), n_init=int(n_init),
                   random_state=int(random_state))
    labels = model.fit_predict(X)
    sil, db = evaluate_clustering(X, labels)
    return {"k": int(k), "model": model, "labels": np.asarray(labels),
            "inertia": float(model.inertia_), "silhouette": sil,
            "db": db, "counts": cluster_counts(labels),
            "centers": np.asarray(model.cluster_centers_),
            "random_state": int(random_state), "n_init": int(n_init)}


def evaluate_kmeans_range(X, ks=None, random_state=DEFAULT_RANDOM_STATE,
                          n_init=DEFAULT_N_INIT):
    ks = list(ks or K_RANGE)
    rows = []
    for k in ks:
        r = evaluate_kmeans(X, k, random_state, n_init)
        rows.append({"K": int(k), "inertia": r["inertia"],
                     "silhouette": r["silhouette"], "db": r["db"]})
    return pd.DataFrame(rows)


def fit_ward(X, k):
    model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
    labels = np.asarray(model.fit_predict(X))
    sil, db = evaluate_clustering(X, labels)
    return {"k": int(k), "model": model, "labels": labels,
            "silhouette": sil, "db": db,
            "counts": cluster_counts(labels), "linkage": "ward"}


def build_dendrogram(X, sample_size=80, seed=42):
    """Full Ward linkage + display subsample for the dendrogram plot."""
    Z_full = sch_linkage(np.asarray(X), method="ward")
    n = len(X)
    if n > sample_size:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n, size=sample_size, replace=False))
    else:
        idx = np.arange(n)
    Z_show = sch_linkage(np.asarray(X)[idx], method="ward")
    return {"Z_full": np.asarray(Z_full), "Z_show": np.asarray(Z_show),
            "sample_idx": np.asarray(idx)}


def dendro_cut_height(Z, k):
    """Cut height giving K clusters: midpoint of merges around it."""
    Z = np.asarray(Z)
    k = int(k)
    if k <= 1:
        return float(Z[-1, 2]) + 1e-6
    return float((Z[-k, 2] + Z[-k + 1, 2]) / 2.0)


def extract_dendrogram_clusters(Z, k):
    return np.asarray(fcluster(np.asarray(Z), t=int(k),
                               criterion="maxclust")) - 1


def run_pca(X, n_components=2):
    pca = PCA(n_components=int(n_components))
    coords = pca.fit_transform(np.asarray(X))
    return {"model": pca, "coords": np.asarray(coords),
            "variance_ratio": np.asarray(pca.explained_variance_ratio_)}


def calculate_centroids(scaler, centers_std, columns=None):
    """Centroids in standardized and original units (labelled)."""
    columns = columns or CONTINUOUS_COLS
    centers_std = np.asarray(centers_std)
    centers_orig = scaler.inverse_transform(centers_std)
    std_df = pd.DataFrame(centers_std, columns=[c + " (z)" for c in columns])
    orig_df = pd.DataFrame(centers_orig, columns=columns)
    std_df.insert(0, "cluster", range(len(std_df)))
    orig_df.insert(0, "cluster", range(len(orig_df)))
    return std_df, orig_df


def assign_archetypes(centroids_orig):
    """Transparent data-driven archetype naming from centroid profiles."""
    df = centroids_orig.set_index("cluster")
    r = lambda s: s.rank(method="min", ascending=True)
    heavy_score = r(df["weight"]) + r(df["horsepower"])
    eff_score = r(df["mpg"]) + r(-df["weight"])
    heavy = heavy_score.idxmax()
    order = eff_score.sort_values(ascending=False).index.tolist()
    efficient = next((c for c in order if c != heavy), heavy)
    names = {}
    for c in df.index:
        if c == heavy:
            names[c] = "Heavy / High-Power"
        elif c == efficient:
            names[c] = "Fuel-Efficient Compact"
        else:
            names[c] = "Balanced / Mid-Range"
    bal = sorted([c for c, n in names.items() if n == "Balanced / Mid-Range"],
                 key=lambda c: df.loc[c, "mpg"], reverse=True)
    if len(bal) > 1:
        for i, c in enumerate(bal, 1):
            names[c] = f"Balanced / Mid-Range {i}"
    rule = ("Archetype assignment logic: highest weight + horsepower rank "
            "-> Heavy / High-Power; highest MPG with lowest weight rank "
            "-> Fuel-Efficient Compact; remaining profile(s) -> Balanced / "
            "Mid-Range.")
    return names, rule


def find_representative_vehicles(df_clean, X_std, centers_std, n=5):
    """Nearest n vehicles to each centroid in standardized space."""
    X = np.asarray(X_std)
    out = {}
    for i, c in enumerate(np.asarray(centers_std)):
        d = np.linalg.norm(X - c, axis=1)
        idx = np.argsort(d)[: int(n)]
        cols = ["car_name", "mpg", "horsepower", "weight",
                "displacement", "cylinders"]
        cols = [c_ for c_ in cols if c_ in df_clean.columns]
        sub = df_clean.iloc[idx][cols].copy()
        sub["distance_to_centroid"] = np.round(d[idx], 4)
        out[i] = sub.reset_index(drop=True)
    return out


def optimal_k_from_silhouette(range_df):
    row = range_df.loc[range_df["silhouette"].idxmax()]
    return int(row["K"]), float(row["silhouette"])


def elbow_suggestion(range_df):
    """Largest relative inertia drop (bend hint) - advisory only."""
    d = range_df.sort_values("K")
    inert = d["inertia"].to_numpy(dtype=float)
    drops = inert[:-1] - inert[1:]
    rel = drops / np.maximum(inert[:-1], 1e-12)
    j = int(np.argmax(rel[1:]) + 1) if len(rel) > 1 else 0
    return int(d["K"].to_numpy()[j + 1]), [round(float(v), 4) for v in rel]


