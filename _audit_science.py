"""Temporary audit script: verify scientific reference values (not part of submission)."""
import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, "src")
from auto_mpg_utils import (load_auto_mpg, impute_horsepower_by_cylinder,
                            standardize_features, evaluate_kmeans, evaluate_kmeans_range,
                            fit_ward, build_dendrogram, extract_dendrogram_clusters,
                            run_pca, calculate_correlations)

FAILS = []
def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + (f"  {extra}" if extra else ""))
    if not cond:
        FAILS.append(label)

df, fmt = load_auto_mpg("auto-mpg.data", "auto")
check("rows = 398", df.shape[0] == 398, str(df.shape))
check("cols = 9", df.shape[1] == 9, str(df.shape))
check("format detected", fmt == "data", fmt)

imp, medians, mb, ma, fb = impute_horsepower_by_cylinder(df)
check("missing before = 6", mb == 6, str(mb))
check("missing after = 0", ma == 0, str(ma))
check("no global fallback used", fb is False)
expect = {3: 98.5, 4: 78.0, 5: 77.0, 6: 100.0, 8: 150.0}
for c, v in expect.items():
    got = float(medians.loc[c])
    check(f"cyl {c} median = {v}", abs(got - v) < 1e-9, str(got))

corr = calculate_correlations(imp)
check("weight~mpg = -0.8317", abs(corr.loc["weight", "mpg"] - (-0.8317)) < 5e-4,
      f"{corr.loc['weight','mpg']:.4f}")
check("horsepower~mpg = -0.7756", abs(corr.loc["horsepower", "mpg"] - (-0.7756)) < 5e-4,
      f"{corr.loc['horsepower','mpg']:.4f}")
check("weight~horsepower = 0.8636", abs(corr.loc["weight", "horsepower"] - 0.8636) < 5e-4,
      f"{corr.loc['weight','horsepower']:.4f}")

std, scaler = standardize_features(imp)
X = std[["mpg", "displacement", "horsepower", "weight", "acceleration"]].to_numpy()
check("std mean ~0", np.all(np.abs(X.mean(axis=0)) < 1e-9))
check("std pop std ~1", np.all(np.abs(X.std(axis=0) - 1.0) < 1e-9))

exp_sil = {2: 0.5249, 3: 0.3819, 4: 0.3580, 5: 0.3493, 6: 0.3237}
exp_inertia = {2: 848.40, 3: 570.68, 4: 460.10, 5: 393.37, 6: 338.74}
sweep = evaluate_kmeans_range(X)
check("K sweep covers 2..6", list(sweep["K"]) == [2, 3, 4, 5, 6], str(list(sweep["K"])))
for _, row in sweep.iterrows():
    k = int(row["K"])
    check(f"K={k} inertia ~ {exp_inertia[k]}", abs(row["inertia"] - exp_inertia[k]) < 0.05,
          f"{row['inertia']:.2f}")
    check(f"K={k} silhouette ~ {exp_sil[k]}", abs(row["silhouette"] - exp_sil[k]) < 1e-4,
          f"{row['silhouette']:.4f}")

km3 = evaluate_kmeans(X, 3)
check("K=3 counts = 150/97/151", list(km3["counts"].sort_index()) == [150, 97, 151],
      str(list(km3["counts"].sort_index())))
check("K=3 silhouette 0.3819", abs(km3["silhouette"] - 0.3819) < 1e-4, f"{km3['silhouette']:.4f}")
check("K=3 DB 0.9835", abs(km3["db"] - 0.9835) < 1e-4, f"{km3['db']:.4f}")

w3 = fit_ward(X, 3)
check("Ward K=3 counts = 217/99/82", list(w3["counts"].sort_index()) == [217, 99, 82],
      str(list(w3["counts"].sort_index())))
check("Ward K=3 silhouette 0.3710", abs(w3["silhouette"] - 0.3710) < 1e-4, f"{w3['silhouette']:.4f}")
check("Ward K=3 DB 0.8993", abs(w3["db"] - 0.8993) < 1e-4, f"{w3['db']:.4f}")

dend = build_dendrogram(X)
dlab = extract_dendrogram_clusters(dend["Z_full"], 3)
ari = adjusted_rand_score(w3["labels"], dlab)
check("dendrogram fcluster == Ward (ARI 1.0000)", abs(ari - 1.0) < 1e-9, f"{ari:.4f}")

pca = run_pca(X, 2)
v = pca["variance_ratio"]
check("PC1 = 78.47%", abs(100 * v[0] - 78.47) < 0.01, f"{100*v[0]:.2f}")
check("PC2 = 14.26%", abs(100 * v[1] - 14.26) < 0.01, f"{100*v[1]:.2f}")
check("PC1+PC2 = 92.73%", abs(100 * float(v.sum()) - 92.73) < 0.01, f"{100*float(v.sum()):.2f}")

print()
if FAILS:
    print(f"{len(FAILS)} SCIENCE FAILURES:")
    for f in FAILS:
        print(" -", f)
    sys.exit(1)
print("ALL SCIENCE CHECKS PASSED")
