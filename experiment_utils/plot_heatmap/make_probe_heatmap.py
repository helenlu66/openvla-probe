#!/usr/bin/env python
"""
Create a layer × predicate-family heat-map from the *fixed* probes.
Usage:
    python make_probe_heatmap.py --log_dir experiments/logs
"""

# --------------------------------------------------------------------------- #
import argparse, ast, glob, os, re, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns
# --------------------------------------------------------------------------- #
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir", default="experiments/logs")
cli.add_argument("--success_only", action="store_true")
cli.add_argument("--outfile", default="probe_heatmap.png")
args = cli.parse_args()
# --------------------------------------------------------------------------- #
OBJ = ast.literal_eval(Path(
        "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT = ast.literal_eval(Path(
        "experiments/robot/libero/object_action_states_keys.txt").read_text())
ALL = OBJ + ACT
def cat(lbl): return lbl.split()[0]
CAT_NAMES = sorted({cat(k) for k in ALL})
# --------------------------------------------------------------------------- #
succ_map = {}
succ_csv = Path(args.log_dir) / "libero_object_summary_per_episode.csv"
if succ_csv.exists():
    import pandas as pd
    df = pd.read_csv(succ_csv)
    succ_map = {int(r.EpisodeAbs): bool(r.Success) for _, r in df.iterrows()}
# --------------------------------------------------------------------------- #
eps = {}
for fp in tqdm(sorted(glob.glob(os.path.join(args.log_dir,"episode_*.pt"))),
               desc="caching episodes", unit="ep"):
    idx = int(re.search(r"episode_(\d+)\.pt", fp).group(1))
    if args.success_only and succ_map and not succ_map.get(idx, True):
        continue
    eps[idx] = torch.load(fp, map_location="cpu")
print(f"cached {len(eps)} episodes")
# --------------------------------------------------------------------------- #
L_MAX  = 33
heat   = np.full((L_MAX, len(CAT_NAMES)), np.nan, dtype=float)

for L in range(L_MAX):
    ck_fp = Path(f"linear_probe_L{L}.pth")
    if not ck_fp.exists():                       # layer not trained / missing
        continue
    ck    = torch.load(ck_fp, map_location="cpu")
    keep  = ck.get("kept")
    probe = torch.nn.Linear(
        eps[next(iter(eps))]["visual_semantic_encoding"][L].shape[-1],
        len(keep))
    probe.load_state_dict(ck["state_dict"]); probe.eval()

    per_cat = defaultdict(list)

    for d in eps.values():
        if L not in d["visual_semantic_encoding"]:
            continue
        x = d["visual_semantic_encoding"][L].float()                     # [T,d]
        y = torch.cat([d["symbolic_state_object_relations"],
                       d["symbolic_state_action_subgoals"]], 1)[:, keep] # [T,|keep|]

        with torch.no_grad():
            pred = (probe(x).sigmoid() > 0.5).long()    # {0,1}

        mask   = (y != -1)
        target = (y == 1).long()
        correct = (pred == target) & mask

        for col, gidx in enumerate(keep):
            fam = cat(ALL[gidx])
            if mask[:, col].any():
                acc = correct[:, col][mask[:, col]].float().mean().item()
                per_cat[fam].append(acc)

    for ci, fam in enumerate(CAT_NAMES):
        if per_cat[fam]:
            heat[L, ci] = float(np.mean(per_cat[fam]))

# --------------------------------------------------------------------------- #
plt.figure(figsize=(12, 10))
sns.heatmap(heat, vmin=0, vmax=1, cmap="YlGnBu",
            xticklabels=CAT_NAMES, yticklabels=list(range(L_MAX)),
            annot=True, fmt=".2f")

plt.xticks(rotation=30, ha="right")  # tilt x by 30°
plt.yticks(rotation=45)              # tilt y by 45°
plt.xlabel("Predicate category"); plt.ylabel("Llama layer")
plt.title("Validation accuracy per layer × category")
plt.tight_layout()
plt.savefig(args.outfile, dpi=300)

