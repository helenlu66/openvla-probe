#  plot_label_frequencies.py
import argparse, ast, glob, os, re, torch, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt, seaborn as sns

# ------------- CLI ----------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--log_dir", default="experiments/logs")
p.add_argument("--success_only", action="store_true")
args = p.parse_args()

# ------------- label keys & helper ------------------------------------------
OBJ = ast.literal_eval(Path(
        "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT = ast.literal_eval(Path(
        "experiments/robot/libero/object_action_states_keys.txt").read_text())
LABELS = OBJ + ACT                                       # length 481

def category(lbl):      # e.g. 'left-of bowl_1 plate_0' → 'left-of'
    return lbl.split()[0]

# ------------- optional success flags ---------------------------------------
succ_map = {}
succ_csv = Path(args.log_dir) / "libero_object_summary_per_episode.csv"
if succ_csv.exists():
    import pandas as pd
    df = pd.read_csv(succ_csv)
    succ_map = {int(r.EpisodeAbs): bool(r.Success) for _, r in df.iterrows()}

# ------------- load all roll-outs -------------------------------------------
Ys = []
for pt in sorted(glob.glob(os.path.join(args.log_dir,"episode_*.pt"))):
    epi = int(re.search(r"episode_(\d+)\.pt", pt).group(1))
    if args.success_only and succ_map and not succ_map.get(epi, True):
        continue
    d = torch.load(pt, map_location="cpu")
    Ys.append(torch.cat([d["symbolic_state_object_relations"],
                         d["symbolic_state_action_subgoals"]], 1))
Y = torch.cat(Ys)                     # [N, 481]

# ----------- per-label positive frequency -----------------------------------
mask       = (Y != -1).bool()                 # True where label is defined
pos        = (Y == 1) & mask                  # True where label == 1
num_pos    = pos.sum(0).float()               # [481]  count of 1’s
num_total  = mask.sum(0).float()              # [481]  count of valid frames

freq = num_pos / num_total                    # fraction of 1’s
freq[num_total == 0] = float('nan')           # columns that were all −1

# ----------- aggregate by predicate family ----------------------------------
cats = sorted({category(k) for k in LABELS})
cat_freq = {c:[] for c in cats}
for i,f in enumerate(freq):
    cat_freq[category(LABELS[i])].append(float(f))
cat_mean = {c: np.mean(v) for c,v in cat_freq.items()}

# ----------- PLOTS ----------------------------------------------------------
sns.set(style="whitegrid")

plt.figure(figsize=(6,4))
plt.hist(freq.numpy(), bins=50, edgecolor='k')
plt.xlabel("positive frequency per label"); plt.ylabel("#labels")
plt.title("Distribution of per-label positive rates")
plt.tight_layout(); plt.savefig("freq_histogram.png", dpi=300)

plt.figure(figsize=(10,4))
bars = plt.bar(cat_mean.keys(), cat_mean.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("mean positive rate"); plt.ylim(0,1)
plt.title("Average positive frequency by predicate category")

# --- Add numeric labels on top of each bar ---
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig("freq_by_category.png", dpi=300)

print("🖼  wrote freq_histogram.png  and  freq_by_category.png")
