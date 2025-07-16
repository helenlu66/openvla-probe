#!/usr/bin/env python
"""
Support-weighted binary-F1 heat-map
for probes  linear_probe_3class_direct_L*.pth
"""

import argparse, glob, os, re, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import f1_score
import ast

# ---------------- CLI -------------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir",   default="experiments/logs")          # where episode_*.pt live
cli.add_argument("--outfile",   default="heatmap_3class_supportF1.png")
args = cli.parse_args()

# ---------------- label helpers --------------------------------------------
OBJ = ast.literal_eval(Path(
        "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT = ast.literal_eval(Path(
        "experiments/robot/libero/object_action_states_keys.txt").read_text())
ALL = OBJ + ACT
fam = lambda k: k.split()[0]
FAM_NAMES = sorted({fam(k) for k in ALL})

# ---------------- cache episodes -------------------------------------------
eps = {}
for fp in tqdm(sorted(glob.glob(os.path.join(args.log_dir, "episode_*.pt"))),
               desc="cache episodes", unit="ep"):
    idx = int(re.search(r"episode_(\d+)\.pt", fp).group(1))
    eps[idx] = torch.load(fp, map_location="cpu")
print(f"✓ cached {len(eps)} episodes")

# ---------------- build heat-map -------------------------------------------
L_MAX, heat = 33, np.full((33, len(FAM_NAMES)), np.nan)

for L in range(L_MAX):
    ck_fp = Path(f"linear_probe_3class_direct_L{L:02d}.pth")
    if not ck_fp.exists(): continue
    ck   = torch.load(ck_fp, map_location="cpu")
    keep = ck["kept_indices"]

    hidden = eps[next(iter(eps))]["visual_semantic_encoding"][L].shape[-1]
    probe  = torch.nn.Linear(hidden, len(keep)*3)
    probe.load_state_dict(ck["state_dict"]); probe.eval()

    fam2vals = defaultdict(list)      # collect (f1, support)

    for d in eps.values():
        if L not in d["visual_semantic_encoding"]: continue
        x = d["visual_semantic_encoding"][L].float()                   # [T,d]
        y = torch.cat([d["symbolic_state_object_relations"],
                       d["symbolic_state_action_subgoals"]], 1)[:,keep]# [T,K]

        with torch.no_grad():
            pred = probe(x).view(-1, len(keep), 3).argmax(-1)          # 0/1/2

        targ = torch.zeros_like(y)
        targ[y == -1] = 0; targ[y == 0] = 1; targ[y == 1] = 2

        for col, gidx in enumerate(keep):
            mask = targ[:,col] != 0
            supp = int(mask.sum())
            if supp == 0: continue
            y_true = (targ[mask,col]==2).cpu(); y_pred = (pred[mask,col]==2).cpu()
            score  = f1_score(y_true, y_pred, average="binary", zero_division=0)
            fam2vals[fam(ALL[gidx])].append((score, supp))

    for ci,f in enumerate(FAM_NAMES):
        if fam2vals[f]:
            scores, supps = zip(*fam2vals[f])
            heat[L,ci] = np.average(scores, weights=supps)

# ---------------- plot ------------------------------------------------------
plt.figure(figsize=(20,15))
sns.heatmap(heat, vmin=0, vmax=1, cmap="YlGnBu",
            xticklabels=FAM_NAMES, yticklabels=list(range(L_MAX)),
            annot=True, fmt=".3f")
plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=45)
plt.xlabel("Predicate family"); plt.ylabel("Llama layer")
plt.title("Support-weighted binary-F1 per layer × family")
plt.tight_layout(); plt.savefig(args.outfile, dpi=900, bbox_inches="tight")
plt.savefig("heatmap_3class_supportF1.pdf", bbox_inches="tight")     # or .svg
print("✓ saved", args.outfile)
