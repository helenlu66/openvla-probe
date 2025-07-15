#!/usr/bin/env python
"""
Visualise per-label statistics for LIBERO-Object roll-outs.

• Histogram of positive (==1) frequency
• Histogram of missing (==-1) frequency
• Grouped bar-chart (predicate family × {positive, missing})
"""

# --------------------------------------------------------------------------- #
import argparse, ast, glob, os, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- CLI ----------------------------------------- #
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir", default="experiments/logs",
                 help="folder with episode_*.pt and *_summary_*.csv")
cli.add_argument("--success_only", action="store_true",
                 help="use only successful episodes (needs the CSV)")
args = cli.parse_args()

# ------------------------ label key files ---------------------------------- #
OBJ = ast.literal_eval(Path(
        "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT = ast.literal_eval(Path(
        "experiments/robot/libero/object_action_states_keys.txt").read_text())
LABELS = OBJ + ACT                      # length 481

def family(lbl: str) -> str:            # "left-of bowl plate" → "left-of"
    return lbl.split()[0]

FAMILIES = sorted({family(k) for k in LABELS})

# --------------------- optional success-only filter ------------------------ #
succ_map = {}
succ_csv = Path(args.log_dir) / "libero_object_summary_per_episode.csv"
if succ_csv.exists():
    import pandas as pd
    df = pd.read_csv(succ_csv)
    succ_map = {int(r.EpisodeAbs): bool(r.Success) for _, r in df.iterrows()}
    if args.success_only:
        print(f"✓ success-only mode – loaded {len(succ_map)} flags")

# --------------------- load all roll-outs ---------------------------------- #
Ys = []
for fp in sorted(glob.glob(os.path.join(args.log_dir, "episode_*.pt"))):
    epi = int(re.search(r"episode_(\d+)\.pt", fp).group(1))
    if args.success_only and succ_map and not succ_map.get(epi, True):
        continue
    d = torch.load(fp, map_location="cpu")
    Ys.append(torch.cat([d["symbolic_state_object_relations"],
                         d["symbolic_state_action_subgoals"]], 1))
if not Ys:
    raise RuntimeError("no episodes found after filtering")
Y = torch.cat(Ys)                       # [N_frames, 481]

# --------------------- per-label statistics -------------------------------- #
valid      = (Y != -1)                  # mask where label defined
pos_rate   = (Y == 1).float().sum(0) / valid.float().sum(0)      # P(y==1 | valid)
miss_rate  = (Y == -1).float().sum(0) / Y.shape[0]               # P(y==-1)

# guard against divide-by-zero (columns that were all –1)
pos_rate[valid.sum(0) == 0] = torch.nan

# --------------------- aggregate by family --------------------------------- #
pos_by_fam  = defaultdict(list)
miss_by_fam = defaultdict(list)

for i in range(len(LABELS)):
    fam = family(LABELS[i])
    pos_by_fam[fam].append(float(pos_rate[i]))
    miss_by_fam[fam].append(float(miss_rate[i]))

fam_pos_mean  = {f: np.nanmean(v) for f, v in pos_by_fam.items()}
fam_miss_mean = {f: np.nanmean(v) for f, v in miss_by_fam.items()}

# --------------------- PLOTS ------------------------------------------------ #
sns.set(style="whitegrid")

# 1️⃣  histogram of positive rates
plt.figure(figsize=(6,4))
plt.hist(pos_rate.numpy(), bins=50, edgecolor="k")
plt.xlabel("positive frequency per label"); plt.ylabel("#labels")
plt.title("Distribution of positive rates")
plt.tight_layout(); plt.savefig("freq_histogram.png", dpi=300)

# 2️⃣  histogram of missing rates
plt.figure(figsize=(6,4))
plt.hist(miss_rate.numpy(), bins=50, edgecolor="k", color="tomato")
plt.xlabel("missing (–1) frequency per label"); plt.ylabel("#labels")
plt.title("Distribution of missing rates")
plt.tight_layout(); plt.savefig("missing_histogram.png", dpi=300)

# 3️⃣  grouped bar chart
x      = np.arange(len(FAMILIES))
width  = 0.35
pos_v  = [fam_pos_mean[f]  for f in FAMILIES]
miss_v = [fam_miss_mean[f] for f in FAMILIES]

plt.figure(figsize=(11,5))
bp = plt.bar(x - width/2, pos_v,  width, label="mean positive rate")
bm = plt.bar(x + width/2, miss_v, width, label="mean missing rate", color="tomato")

plt.xticks(x, FAMILIES, rotation=45, ha="right")
plt.ylim(0,1); plt.ylabel("fraction of frames")
plt.title("Positive vs. missing rate by predicate family")
plt.legend()

# --- annotate bars with numeric value
for bars in (bp, bm):
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8)

plt.tight_layout(); plt.savefig("freq_by_family_pos+miss.png", dpi=300)

print("🖼  wrote freq_histogram.png, missing_histogram.png, freq_by_family_pos+miss.png")
