#!/usr/bin/env python
"""
analysis/03_global_hists.py
Read analysis/per_label_metrics.csv and draw:
  • histograms for every layer × label       (already in df)
  • histogram with *one row per predicate*   (df_mean)
  • print the hard cases (AUPRC < 0.30)
"""

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

# ---------- load the big per-layer CSV --------------------------------------
df = pd.read_csv("analysis/per_label_metrics.csv")   # this has >13 k rows

# ---------------------------------------------------------------------------
# 0️⃣  (optional) original per-layer histograms
# ---------------------------------------------------------------------------
for metric, label in [("f1",   "F1 (pos-class)"),
                      ("auprc","AUPRC"),
                      ("mcc",  "Matthews corr.")]:
    plt.figure(figsize=(6,4))
    sns.histplot(df[metric].dropna(), bins=40, edgecolor="k")
    plt.xlabel(label); plt.ylabel("# layer-label pairs")
    plt.title(f"Distribution of {label} (33 layers × 439 labels)")
    plt.tight_layout(); plt.savefig(f"analysis/hist_{metric}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# 1️⃣  collapse across layers  → one row per *semantic* label
# ---------------------------------------------------------------------------
df_mean = (df
           .groupby("label", as_index=False)
           .agg({"auprc":"mean",
                 "f1"   :"mean",
                 "mcc"  :"mean",
                 "family":"first"}))          # keep the family name

# ---- new histogram: unique-label AUPRC ------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df_mean["auprc"], bins=40, edgecolor='k')
plt.xlabel("AUPRC  (averaged over 33 layers)")
plt.ylabel("# unique predicates")
plt.title("Distribution of AUPRC • one entry per predicate")
plt.tight_layout()
plt.savefig("analysis/hist_auprc_unique_labels.png", dpi=300)
print("✓ hist_auprc_unique_labels.png written")

# ---------------------------------------------------------------------------
# 2️⃣  list the truly hard predicates
# ---------------------------------------------------------------------------
threshold = 0.30          # tweak as you like
hard = df_mean[df_mean["auprc"] < threshold]

if hard.empty:
    print(f"\nNo predicate has AUPRC < {threshold:.2f} 🎉")
else:
    print(f"\nPredicates with AUPRC < {threshold:.2f}:")
    for fam, lbls in hard.groupby("family")["label"]:
        print(f"  {fam:20s} : {len(lbls):2d}  →  {list(lbls)}")
