#!/usr/bin/env python
"""
Visualise per-label statistics for LIBERO-Object roll-outs
— positive / negative / missing all share the same denominator.
"""

# --------------------------------------------------------------------------- #
import argparse, ast, glob, os, re
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns

# ----------------------------- CLI ----------------------------------------- #
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir",      default="experiments/logs")
cli.add_argument("--success_only", action="store_true")
args = cli.parse_args()

# ------------------------ label keys --------------------------------------- #
OBJ = ast.literal_eval(Path("experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT = ast.literal_eval(Path("experiments/robot/libero/object_action_states_keys.txt").read_text())
LABELS   = OBJ + ACT
FAMILIES = sorted({lbl.split()[0] for lbl in LABELS})
fam_of   = lambda lbl: lbl.split()[0]

# ------------------ success-only filter (optional) ------------------------- #
succ_map = {}
csv = Path(args.log_dir) / "libero_object_summary_per_episode.csv"
if csv.exists() and args.success_only:
    import pandas as pd
    df = pd.read_csv(csv)
    succ_map = {int(r.EpisodeAbs): bool(r.Success) for _, r in df.iterrows()}
    print(f"✓ success-only mode — loaded {len(succ_map)} flags")

# ------------------ load all roll-outs ------------------------------------- #
Ys = []
for fp in sorted(glob.glob(os.path.join(args.log_dir, "episode_*.pt"))):
    ep = int(re.search(r"episode_(\d+)\.pt", fp).group(1))
    if args.success_only and succ_map and not succ_map.get(ep, True):
        continue
    d = torch.load(fp, map_location="cpu")
    Ys.append(torch.cat([d["symbolic_state_object_relations"],
                         d["symbolic_state_action_subgoals"]], 1))
if not Ys:
    raise RuntimeError("no episodes found after filtering")
Y = torch.cat(Ys)                                      # [N_frames, 481]
N_all = Y.shape[0]                                     # same for every label

# ------------------ per-label frequencies (统一分母) ------------------------ #
n_pos     = (Y ==  1).sum(0).float()                   # #frames ==1
n_neg     = (Y ==  0).sum(0).float()                   # #frames ==0
n_missing = (Y == -1).sum(0).float()                   # #frames ==-1

pos_rate  = (n_pos     / N_all).numpy()               # P(y==1)
neg_rate  = (n_neg     / N_all).numpy()               # P(y==0)
miss_rate = (n_missing / N_all).numpy()               # P(y==-1)

# 如果某列全是 -1，pos/neg 都是 0；我们把它们设成 NaN 便于平均
pos_rate[n_missing == N_all] = np.nan
neg_rate[n_missing == N_all] = np.nan

# --------------------------------------------------------------------------- #
# >>> NEW: dump the raw numbers to a CSV so nothing is lost in binning
import pandas as pd
df = pd.DataFrame({
    "label"     : LABELS,
    "family"    : [fam_of(l) for l in LABELS],
    "n_pos"     : n_pos.numpy(),
    "n_neg"     : n_neg.numpy(),
    "n_missing" : n_missing.numpy(),
    "p_pos"     : pos_rate,
    "p_neg"     : neg_rate,
    "p_missing" : miss_rate,
})
df.to_csv("label_stats.csv", index=False)
print("📄  Saved exact per-label numbers to label_stats.csv")

# >>> optionally print a quick summary in the console
print("\n┌─ Least frequent positives (p_pos ascending) ─────────")
print(df.sort_values("p_pos").head(10)[["label","p_pos"]].to_string(index=False))

print("\n┌─ Most often missing (p_missing descending) ──────────")
print(df.sort_values("p_missing", ascending=False).head(10)[["label","p_missing"]].to_string(index=False))
# --------------------------------------------------------------------------- #

# ------------------ family-level平均 --------------------------------------- #
fam_stats = {f: {"pos": [], "neg": [], "miss": []} for f in FAMILIES}
for i, lbl in enumerate(LABELS):
    f = fam_of(lbl)
    fam_stats[f]["pos"].append(pos_rate[i])
    fam_stats[f]["neg"].append(neg_rate[i])
    fam_stats[f]["miss"].append(miss_rate[i])

fam_pos  = {f: np.nanmean(v["pos"])  for f, v in fam_stats.items()}
fam_neg  = {f: np.nanmean(v["neg"])  for f, v in fam_stats.items()}
fam_miss = {f: np.nanmean(v["miss"]) for f, v in fam_stats.items()}

# ------------------ PLOTS --------------------------------------------------- #
sns.set(style="whitegrid")

# 1️⃣ hist - positive
plt.figure(figsize=(6,4))
plt.hist(pos_rate[~np.isnan(pos_rate)], bins=50, edgecolor="k")
plt.xlabel("P(y==1) per label"); plt.ylabel("#labels")
plt.title("Distribution of positive rate")
plt.tight_layout(); plt.savefig("hist_positive.png", dpi=300)

# 2️⃣ hist - missing
plt.figure(figsize=(6,4))
plt.hist(miss_rate, bins=50, color="tomato", edgecolor="k")
plt.xlabel("P(y==-1) per label"); plt.ylabel("#labels")
plt.title("Distribution of missing rate")
plt.tight_layout(); plt.savefig("hist_missing.png", dpi=300)

# 3️⃣ grouped bar-chart
x = np.arange(len(FAMILIES)); w = 0.22
fig, ax = plt.subplots(figsize=(12,5))
bp = ax.bar(x - w, [fam_pos [f] for f in FAMILIES], w, label="P(+)")
bn = ax.bar(x      , [fam_neg [f] for f in FAMILIES], w, label="P(0)")
bm = ax.bar(x + w, [fam_miss[f] for f in FAMILIES], w, label="P(–1)", color="tomato")

ax.set_xticks(x); ax.set_xticklabels(FAMILIES, rotation=45, ha="right")
ax.set_ylim(0,1); ax.set_ylabel("fraction of frames")
ax.set_title("Per-family positive / negative / missing rate")
ax.legend()

# annotate
def annotate(bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7)
for bars in (bp,bn,bm): annotate(bars)

fig.tight_layout(); fig.savefig("family_pos_neg_miss.png", dpi=300)

print("🖼  wrote  hist_positive.png, hist_missing.png, family_pos_neg_miss.png")
