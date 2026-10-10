"""Spot-check for newly wired product features (verification only)."""
import sys
from streamlit.testing.v1 import AppTest

FAILS = []
def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(label)

def nav(at, target):
    for b in at.button:
        if b.label == target and (b.key or "").startswith("navbtn_"):
            b.click()
            at.run()
            return
    raise AssertionError(f"nav button for {target!r} not found")

at = AppTest.from_file("app.py", default_timeout=120)
at.run()
check("Overview loads", not at.exception, str(at.exception)[:300])

# --- Guided walkthrough (demo_step) ---
nxt = [b for b in at.button if b.label == "Next step ▶"]
check("Walkthrough Next button present", len(nxt) == 1)
if nxt:
    nxt[0].click(); at.run()
    check("Walkthrough advances", at.session_state["demo_step"] == 1, str(at.session_state.get("demo_step")))
    check("No exception after step", not at.exception, str(at.exception)[:300])
    prev = [b for b in at.button if b.label == "◀ Previous step"]
    check("Previous enabled after advancing", bool(prev) and not prev[0].disabled)

# --- Full pipeline ---
nav(at, "Dataset")
[b for b in at.button if "Use Project Dataset" in b.label][0].click(); at.run()
nav(at, "Preprocessing")
[b for b in at.button if "Apply Cylinder-Group Median Imputation" in b.label][0].click(); at.run()
[b for b in at.button if b.label == "Standardize Features"][0].click(); at.run()
nav(at, "Clustering Lab")
[b for b in at.button if b.label == "RUN CLUSTERING"][0].click(); at.run()
check("Trained", at.session_state["kmeans"] is not None)

# --- New cluster view selectbox + centroid overlay ---
cv = [s for s in at.selectbox if s.label == "Cluster view"]
check("Cluster view selectbox present", len(cv) == 1)
if cv:
    cv[0].set_value("Horsepower vs MPG (horsepower / mpg)"); at.run()
    check("Cluster view switch OK", not at.exception, str(at.exception)[:300])
cb = [c for c in at.checkbox if c.label.startswith("Show centroid markers")]
check("Centroid toggle present", len(cb) == 1)
if cb:
    cb[0].check(); at.run()
    check("Centroid overlay OK", not at.exception, str(at.exception)[:300])

# --- Compare clusters block ---
ca = [s for s in at.selectbox if s.label == "Cluster A"]
cbx = [s for s in at.selectbox if s.label == "Cluster B"]
check("Compare clusters A/B present", len(ca) == 1 and len(cbx) == 1)
if ca and cbx:
    ca[0].set_value(0); cbx[0].set_value(2); at.run()
    check("Compare clusters OK", not at.exception, str(at.exception)[:300])

# --- Archetypes multiselect (arch_compare) ---
nav(at, "Fleet Archetypes")
check("Archetypes loads", not at.exception, str(at.exception)[:300])
ms = [m for m in at.multiselect if m.label.startswith("Pick two or more archetypes")]
check("arch_compare multiselect present", len(ms) == 1)
if ms:
    ms[0].set_value([0, 1]); at.run()
    check("arch_compare sets state", list(at.session_state["arch_compare"]) == [0, 1])
    check("arch_compare renders", not at.exception, str(at.exception)[:300])

# --- K change prunes stale arch_compare ---
sl = [w for w in at.slider if w.label.startswith("K (2 to 6)")]
# slider lives on Clustering Lab; navigate there, set K=3 stays, but simulate stale index
at.session_state["arch_compare"] = [0, 5]
nav(at, "Fleet Archetypes")
check("Stale arch_compare pruned without exception", not at.exception, str(at.exception)[:300])
check("Stale index removed", 5 not in list(at.session_state.get("arch_compare", [])), str(at.session_state.get("arch_compare")))

# --- Insights + Methodology pages ---
nav(at, "Insights")
check("Insights loads", not at.exception, str(at.exception)[:300])
nav(at, "Methodology")
check("Methodology loads", not at.exception, str(at.exception)[:300])

# --- Presentation mode toggle ---
nav(at, "Overview")
pt = [t for t in at.toggle if t.label == "Presentation Mode"]
check("Presentation Mode toggle present", len(pt) == 1)
if pt:
    pt[0].set_value(True); at.run()
    check("Presentation mode applies", at.session_state["presentation_mode"] is True)
    check("No exception in presentation mode", not at.exception, str(at.exception)[:300])

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES:")
    for f in FAILS:
        print(" -", f)
    sys.exit(1)
print("ALL SPOT CHECKS PASSED")
