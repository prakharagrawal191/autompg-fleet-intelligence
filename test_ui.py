"""Headless UI walkthrough using Streamlit AppTest (verification only)."""
import sys
from streamlit.testing.v1 import AppTest

FAILS = []
def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(label)

at = AppTest.from_file("app.py", default_timeout=120)

def nav(target):
    for b in at.button:
        if b.label == target and (b.key or "").startswith("navbtn_"):
            b.click()
            at.run()
            return
    raise AssertionError(f"nav button for {target!r} not found")

at.run()
check("Overview loads without exception", not at.exception, str(at.exception))
check("Hero visible", any("AUTOMOTIVE FLEET" in str(m.value) for m in at.markdown))
check("Empty state before dataset", any("DATASET NOT READY" in str(m.value) for m in at.markdown))

nav("Dataset")
check("Dataset page loads", not at.exception, str(at.exception))
btn = [b for b in at.button if "Use Project Dataset" in b.label]
check("Project dataset button present", len(btn) == 1)
btn[0].click()
at.run()
check("Dataset loaded", at.session_state["df_raw"] is not None)
check("Rows = 398", at.session_state["df_raw"].shape[0] == 398)
check("Missing HP = 6", at.session_state["miss_before"] == 6)

nav("Preprocessing")
check("Preprocessing loads", not at.exception, str(at.exception))
imp = [b for b in at.button if "Apply Cylinder-Group Median Imputation" in b.label][0]
imp.click(); at.run()
check("Imputation done (6 -> 0)", at.session_state["imp_done"] and at.session_state["miss_after"] == 0)
std = [b for b in at.button if b.label == "Standardize Features"][0]
std.click(); at.run()
check("Standardization done", at.session_state["std_done"])
check("Std weight mean ~ 0", abs(float(at.session_state["df_std"]["weight"].mean())) < 1e-9)

nav("Clustering Lab")
check("Clustering Lab loads", not at.exception, str(at.exception))
run = [b for b in at.button if b.label == "RUN CLUSTERING"][0]
run.click(); at.run()
km = at.session_state["kmeans"]
check("K-Means trained", km is not None)
check("K=3 silhouette 0.3819", km and abs(km["silhouette"] - 0.3819) < 1e-4, f"{km['silhouette']:.4f}" if km else "")
check("K=3 DB 0.9835", km and abs(km["db"] - 0.9835) < 1e-4, f"{km['db']:.4f}" if km else "")
check("K=3 inertia 570.68", km and abs(km["inertia"] - 570.68) < 0.05, f"{km['inertia']:.2f}" if km else "")

for K, sil in [(2, 0.5249), (4, 0.3580), (5, 0.3493), (6, 0.3237), (3, 0.3819)]:
    sl = [w for w in at.slider if w.label.startswith("K (2 to 6)")][0]
    sl.set_value(K); at.run()
    check(f"K change to {K} invalidates results", at.session_state["kmeans"] is None)
    run = [b for b in at.button if b.label == "RUN CLUSTERING"][0]
    run.click(); at.run()
    got = at.session_state["kmeans"]["silhouette"]
    check(f"K={K} silhouette {sil}", abs(got - sil) < 1e-4, f"{got:.4f}")
ek = [b for b in at.button if "Compute Elbow + Silhouette" in b.label][0]
ek.click(); at.run()
kr = at.session_state["krange"]
check("Elbow sweep computed", kr is not None and len(kr) == 5)
if kr is not None:
    check("K=2 inertia 848.40", abs(float(kr.set_index("K").loc[2, "inertia"]) - 848.40) < 0.05)

use_w = [b for b in at.button if b.label == "Use Ward Hierarchical"][0]
use_w.click(); at.run()
check("Method switched to Ward", at.session_state["method"] != "K-Means")
run = [b for b in at.button if b.label == "RUN CLUSTERING"][0]
run.click(); at.run()
w = at.session_state["ward"]
check("Ward trained", w is not None)
check("Ward K=3 silhouette 0.3710", w and abs(w["silhouette"] - 0.3710) < 1e-4, f"{w['silhouette']:.4f}" if w else "")
check("Ward K=3 DB 0.8993", w and abs(w["db"] - 0.8993) < 1e-4, f"{w['db']:.4f}" if w else "")

db = [b for b in at.button if "Build Ward dendrogram" in b.label][0]
db.click(); at.run()
check("Dendrogram built", at.session_state["dendro"] is not None)
ex = [b for b in at.button if "Extract Cluster Assignments" in b.label][0]
ex.click(); at.run()
check("Dendrogram assignments extracted", at.session_state["dlab"] is not None)

for page in ["Overview", "Dataset", "Preprocessing", "EDA Explorer", "Clustering Lab",
             "Cluster Quality", "PCA Explorer", "Fleet Archetypes", "Vehicle Lookup",
             "Exports", "About"]:
    nav(page)
    check(f"Page loads: {page}", not at.exception, str(at.exception)[:400])

nav("PCA Explorer")
p = at.session_state["pca"]
check("PCA PC1 78.47", p and abs(100 * p["variance_ratio"][0] - 78.47) < 0.01)
check("PCA PC2 14.26", p and abs(100 * p["variance_ratio"][1] - 14.26) < 0.01)
check("PCA total 92.73", p and abs(100 * float(sum(p["variance_ratio"])) - 92.73) < 0.01)

nav("Fleet Archetypes")
check("Archetypes computed", at.session_state["arch"] is not None)
check("3 archetype names", at.session_state["arch"] is not None and len(at.session_state["arch"]) == 3)
nav("Vehicle Lookup")
sel = [s for s in at.selectbox if s.label == "Vehicle"]
check("Lookup selectbox present", len(sel) == 1)
if sel:
    sel[0].select_index(len(sel[0].options) - 1); at.run()  # options are row positions (names repeat)
    check("Lookup selection works", not at.exception, str(at.exception)[:300])

nav("EDA Explorer")
feat = [s for s in at.selectbox if s.label == "Feature"][0]
feat.set_value("horsepower"); at.run()
check("EDA feature switch", not at.exception, str(at.exception)[:300])
viz = [s for s in at.selectbox if s.label == "Visualization"][0]
viz.set_value("Boxplot"); at.run()
check("EDA viz switch", not at.exception, str(at.exception)[:300])

nav("Exports")
labels = [b.label for b in at.download_button]
for name in ["Cleaned", "Standardized", "K-Means Results", "Ward Results",
             "Centroids", "Evaluation Metrics", "Dendrogram Assignments", "Complete Clustered"]:
    check(f"Export present: {name}", any(name in l for l in labels))

rst = [b for b in at.button if b.label == "Reset Analysis"][0]
rst.click(); at.run()
check("Reset clears analysis", at.session_state["df_raw"] is None and at.session_state["kmeans"] is None)
check("Reset keeps navigation", at.session_state["page"] == "Exports")

# --- Fresh session: reset while Archetype widgets are on screen ---
at2 = AppTest.from_file("app.py", default_timeout=120)
at2.run()
def nav2t(t):
    for b in at2.button:
        if b.label == t and (b.key or "").startswith("navbtn_"):
            b.click(); at2.run(); return
    raise AssertionError(f"nav2 button for {t!r} not found")
nav2t("Dataset")
[b for b in at2.button if "Use Project Dataset" in b.label][0].click(); at2.run()
nav2t("Preprocessing")
[b for b in at2.button if "Apply Cylinder-Group Median Imputation" in b.label][0].click(); at2.run()
[b for b in at2.button if b.label == "Standardize Features"][0].click(); at2.run()
nav2t("Clustering Lab")
[b for b in at2.button if b.label == "RUN CLUSTERING"][0].click(); at2.run()
nav2t("Fleet Archetypes")
check("Archetypes page live", not at2.exception, str(at2.exception)[:300])
whys = [s for s in at2.selectbox if s.label.startswith("Why does this car")]
if whys:
    whys[0].set_value(whys[0].options[-1]); at2.run()
    check("Archetype why-select works", not at2.exception, str(at2.exception)[:300])
rst2 = [b for b in at2.button if b.label == "Reset Analysis"][0]
rst2.click(); at2.run()
check("Reset on Archetypes page raises nothing", not at2.exception, str(at2.exception)[:300])
check("Reset cleared analysis state", at2.session_state["df_raw"] is None and at2.session_state["arch"] is None)

# --- Ward-only session (K-Means never run) ---
at3 = AppTest.from_file("app.py", default_timeout=120)
at3.run()
def nav3t(t):
    for b in at3.button:
        if b.label == t and (b.key or "").startswith("navbtn_"):
            b.click(); at3.run(); return
    raise AssertionError(f"nav3 button for {t!r} not found")
nav3t("Dataset")
[b for b in at3.button if "Use Project Dataset" in b.label][0].click(); at3.run()
nav3t("Preprocessing")
[b for b in at3.button if "Apply Cylinder-Group Median Imputation" in b.label][0].click(); at3.run()
[b for b in at3.button if b.label == "Standardize Features"][0].click(); at3.run()
nav3t("Clustering Lab")
[b for b in at3.button if b.label == "Use Ward Hierarchical"][0].click(); at3.run()
[b for b in at3.button if b.label == "RUN CLUSTERING"][0].click(); at3.run()
check("Ward-only training", at3.session_state["ward"] is not None and at3.session_state["kmeans"] is None)
for p in ["Cluster Quality", "Fleet Archetypes", "Vehicle Lookup", "Exports", "About", "Overview"]:
    nav3t(p)
    check(f"Ward-only page: {p}", not at3.exception, str(at3.exception)[:300])

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:")
    for f in FAILS:
        print(" -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
