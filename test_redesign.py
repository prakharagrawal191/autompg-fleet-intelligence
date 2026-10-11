"""Regression checks for the redesigned shell: state contract, validation, geometry, pages."""
import io
import logging
import pathlib
import re
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score
from streamlit.testing.v1 import AppTest

from src import state as S
from src.auto_mpg_utils import COLUMN_NAMES, dendro_cut_height

logging.disable(logging.WARNING)
FAILS = []


def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(label)


def fresh():
    d = {}
    S.init(d)
    return d


def load_text(text):
    d = fresh()
    return d, S.load_from_source(io.StringIO(text), "upload", "test.csv", "auto", d)


# --- science through the state layer (reference values, never typed into the UI) ---
both = S.seed(methods=(S.KMEANS, S.WARD), dendro=True)
km, w = both["kmeans"], both["ward"]
check("K-Means K=3 silhouette 0.3819", abs(km["silhouette"] - 0.3819) < 1e-4, f"{km['silhouette']:.4f}")
check("K-Means K=3 Davies-Bouldin 0.9835", abs(km["db"] - 0.9835) < 1e-4, f"{km['db']:.4f}")
check("K-Means K=3 inertia 570.68", abs(km["inertia"] - 570.68) < 0.05, f"{km['inertia']:.2f}")
check("Ward K=3 silhouette 0.3710 / DB 0.8993",
      abs(w["silhouette"] - 0.3710) < 1e-4 and abs(w["db"] - 0.8993) < 1e-4)
check("Sweep silhouettes K=2..6", np.allclose(both["krange"]["silhouette"].round(4),
                                               [0.5249, 0.3819, 0.3580, 0.3493, 0.3237]))
v = both["pca"]["variance_ratio"]
check("PCA 78.47 / 14.26 / 92.73", abs(100 * v[0] - 78.47) < 0.01 and abs(100 * v[1] - 14.26) < 0.01
      and abs(100 * v.sum() - 92.73) < 0.01)

# --- central invalidation ---
d = S.seed(methods=(S.KMEANS, S.WARD), dendro=True)
S.set_k(4, d)
check("K change clears both models and K-bound results",
      all(d[k] is None for k in ("kmeans", "ward", "cent_orig", "arch", "dlab", "pca")))
check("K change keeps the K-independent sweep and linkage", d["krange"] is not None and d["dendro"] is not None)
d = S.seed(methods=(S.KMEANS,))
S.apply_standardization(d)
check("Re-standardizing clears models, sweep and PCA",
      all(d[k] is None for k in ("kmeans", "krange", "pca", "arch")))
d = S.seed(methods=(S.KMEANS, S.WARD))
d["km_ni"] = 5
S.invalidate_kmeans(d)
check("K-Means setting change clears only K-Means artefacts", d["kmeans"] is None and d["ward"] is not None)
d = S.seed(methods=(S.KMEANS,))
d["dataset_id"] = "another-dataset"
check("Model from another dataset is never served", S.model(S.KMEANS, d) is None and not S.has_model(d))
d = S.seed(methods=(S.KMEANS,))
d["page"] = "Exports"
S.reset_analysis(d)
check("Reset clears analysis, keeps page", d["df_raw"] is None and d["kmeans"] is None and d["page"] == "Exports")

# --- dataset validation: failures are structured and never activate the file ---
raw = pathlib.Path(S.PROJECT_FILE).read_text()
cases = {
    "missing columns": "mpg,cylinders\n1,2\n",
    "reordered header": ",".join(reversed(COLUMN_NAMES)) + "\n" + "\n".join(
        "x%d,1,70,12,3504,130,307,8,18" % i for i in range(12)),
    "binary content": "\x00\x01\x02",
    "empty file": "",
    "missing mpg": raw.replace("18.0   8", "?   8", 1),
    "too many rows": ",".join(COLUMN_NAMES) + "\n" + "18,8,307,130,3504,12,70,1,car\n" * (S.MAX_ROWS + 1),
    "infinite value": raw.replace("3504.", "inf", 1),
}
for name, text in cases.items():
    d, rep = load_text(text)
    check(f"Validation rejects: {name}", not rep["ok"] and d["df_raw"] is None and rep["errors"]
          and all({"what", "why", "next"} <= set(e) for e in rep["errors"]),
          rep["errors"][0]["what"][:70] if rep["errors"] else "no error")
d, rep = load_text("a,b,c,d,e,f,g,h,i\n" + "\n".join(f"{18 + i},8,{300 + i},130,3504,12,70,1,car{i}"
                                                       for i in range(12)))
check("Renamed header loads with a positional-mapping note", rep["ok"] and rep["notes"])
d, rep = load_text("\n".join(raw.splitlines()[:4] * 3))
check("Validation rejects too few distinct vehicles for K up to 6", not rep["ok"],
      rep["errors"][0]["what"] if rep["errors"] else "accepted")

# --- geometry behind the visuals ---
Z = both["dendro"]["Z_full"]
for k in range(2, 7):
    by_height = fcluster(Z, t=dendro_cut_height(Z, k), criterion="distance")
    by_count = fcluster(Z, t=k, criterion="maxclust")
    check(f"Dendrogram cut line reproduces fcluster(maxclust) at K={k}",
          len(set(by_height)) == k and adjusted_rand_score(by_height, by_count) == 1.0)
p = both["pca"]
for m in (S.KMEANS, S.WARD):
    r = S.model(m, both)
    proj = p["model"].transform(S.centers_std(m, both))
    means = np.vstack([p["coords"][r["labels"] == i].mean(axis=0) for i in range(r["k"])])
    check(f"{m} centres projected into PCA coordinates", np.allclose(proj, means, atol=1e-9))
dist = np.linalg.norm(S.X(both)[:, None, :] - km["centers"][None], axis=2)
check("Lookup claim holds: every K-Means label is the nearest centroid", bool((dist.argmin(1) == km["labels"]).all()))

# --- every page renders in every state ---
PAGES = ["Overview", "Dataset", "Preprocessing", "EDA Explorer", "Clustering Lab", "Cluster Quality",
         "PCA Explorer", "Fleet Archetypes", "Vehicle Lookup", "Insights", "Methodology", "Exports", "About"]
raw_only = fresh()
S.load_project(raw_only)
std_only = fresh()
S.load_project(std_only)
S.apply_imputation(std_only)
S.apply_standardization(std_only)
STATES = {"empty": {}, "raw": raw_only, "standardized": std_only, "ward only": S.seed(methods=(S.WARD,)),
          "both K=2": S.seed(k=2, methods=(S.KMEANS, S.WARD), dendro=True),
          "both K=6": S.seed(k=6, methods=(S.KMEANS, S.WARD))}
for sname, state in STATES.items():
    bad = []
    for page in PAGES:
        at = AppTest.from_file("app.py", default_timeout=120)
        for k, val in state.items():
            at.session_state[k] = val
        at.session_state["page"] = page
        at.run()
        if at.exception:
            bad.append(f"{page}: {str(at.exception[0].value)[:80]}")
    check(f"All {len(PAGES)} pages render · {sname}", not bad, "; ".join(bad))



def render(state, page, **extra):
    at = AppTest.from_file("app.py", default_timeout=120)
    for k, val in {**state, **extra, "page": page}.items():
        at.session_state[k] = val
    at.run()
    return at


# --- Demo Mode never collides with page gate buttons (review finding: duplicate element ids) ---
for sname in ("empty", "standardized"):
    bad = [f"step {i} · {page}" for i in range(len(S.DEMO_STEPS)) for page in PAGES
           if render(STATES[sname], page, demo_mode=True, demo_step=i).exception]
    check(f"Demo Mode renders every step on every page · {sname}", not bad, "; ".join(bad[:3]))

# --- page-local settings survive navigation (review finding: n_init silently reverted) ---
at = render(S.seed(methods=()), "Clustering Lab")
at.number_input(key="km_ni").set_value(5)
at.run()
for target in ("Methodology", "Clustering Lab"):
    next(b for b in at.button if b.label == target and (b.key or "").startswith("navbtn_")).click()
    at.run()
check("K-Means n_init setting survives a page round trip", at.session_state["km_ni"] == 5,
      str(at.session_state["km_ni"]))

# --- ties are reported as ties (review finding: K-Means named leader on equal scores) ---
rng = np.random.default_rng(0)
centers = np.array([[15, 350, 160, 4200, 12], [22, 200, 100, 3000, 15], [30, 110, 75, 2200, 17],
                    [38, 90, 65, 1900, 19]])
rows = [centers[i % 4] * (1 + rng.normal(0, 0.004, 5)) for i in range(40)]
text = ",".join(COLUMN_NAMES) + "\n" + "\n".join(
    f"{r[0]:.2f},4,{r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},75,1,car{i}" for i, r in enumerate(rows))
tie = fresh()
S.load_from_source(io.StringIO(text), "upload", "groups.csv", "auto", tie)
S.apply_imputation(tie)
S.apply_standardization(tie)
S.set_k(4, tie)
for m in (S.KMEANS, S.WARD):
    S.set_method(m, tie)
    S.fit_active(tie)
same = adjusted_rand_score(tie["kmeans"]["labels"], tie["ward"]["labels"]) == 1.0
md = " ".join(str(m.value) for m in render(tie, "Cluster Quality").markdown)
check("Identical partitions are reported as a tie", same and "Both methods score the same" in md,
      md[md.find("Leaders"):][:120] if "Leaders" in md else "no leaders panel")

# --- dendrogram colors follow the Ward ids when the partition matches ---
cap = " ".join(c.value for c in render(both, "Clustering Lab").caption)
check("Dendrogram branch colors follow the Ward fit's cluster numbers", "follow the Ward fit's" in cap)

# --- Plotly text is escaped too (it renders an HTML subset) ---
frame = both["df_clean"].copy()
frame.loc[0, "car_name"] = "<a href='https://example.com'>x</a>"
from src import chart_factory as CF  # noqa: E402
fig = CF.cluster_scatter(frame, "weight", "mpg", km["labels"], km["k"], xtitle="w", ytitle="m", show_text=True)
texts = [str(t) for tr in fig.data for arr in (tr.hovertext, tr.text) if arr is not None for t in arr]
check("Vehicle names are escaped in Plotly text and hover", not any("<a " in t for t in texts)
      and any("&lt;a " in t for t in texts))

# --- uploaded text is data, never markup ---
evil = both.copy()
evil["df_clean"] = both["df_clean"].copy()
evil["df_clean"].loc[0, "car_name"] = "<img src=x onerror=alert(1)>"
at = AppTest.from_file("app.py", default_timeout=120)
for k, val in evil.items():
    at.session_state[k] = val
at.session_state["page"] = "Vehicle Lookup"
at.session_state["lookup_car"] = 0
at.run()
html = " ".join(str(m.value) for m in at.markdown)
check("Vehicle names are HTML-escaped", "<img" not in html and "&lt;img" in html)

# --- source hygiene ---
views = sorted(pathlib.Path("src/views").glob("*.py")) + [pathlib.Path("app.py")]
src = {f.name: f.read_text(encoding="utf-8") for f in views}
check("No deprecated use_container_width", not [n for n, t in src.items() if "use_container_width" in t])
check("No fake delays", not [n for n, t in src.items() if "time.sleep" in t])
pat = re.compile(r"0\.3819|0\.9835|0\.3710|0\.8993|0\.5249|78\.47|92\.73|570\.68|848\.4")
check("No reference metrics hard-coded in pages", not [n for n, t in src.items() if pat.search(t)],
      ", ".join(n for n, t in src.items() if pat.search(t)))

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:")
    for f in FAILS:
        print(" -", f)
    sys.exit(1)
print("ALL REDESIGN CHECKS PASSED")
