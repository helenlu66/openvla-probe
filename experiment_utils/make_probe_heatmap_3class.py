#!/usr/bin/env python
"""
make_probe_heatmap_3class.py   —   for checkpoints trained with
linear_probe_3class_direct_LXX.pth  (output dim = |keep| * 3)
Usage:
    python make_probe_heatmap_3class.py \
           --log_dir experiments/logs \
           --outfile heatmap_3class.png
"""
import argparse, ast, glob, os, re, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import f1_score

# ---------- CLI -------------------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir", default="experiments/logs")
cli.add_argument("--outfile", default="heatmap_3class.png")
args = cli.parse_args()

# ---------- label helpers ---------------------------------------------------
OBJ = ast.literal_eval(Path(
        "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT = ast.literal_eval(Path(
        "experiments/robot/libero/object_action_states_keys.txt").read_text())
ALL = OBJ + ACT
fam = lambda lbl: lbl.split()[0]
FAM_NAMES = sorted({fam(k) for k in ALL})

# ---------- cache episodes --------------------------------------------------
eps = {}
for fp in tqdm(sorted(glob.glob(os.path.join(args.log_dir, "episode_*.pt"))),
               desc="cache episodes", unit="ep"):
    idx = int(re.search(r"episode_(\d+)\.pt", fp).group(1))
    eps[idx] = torch.load(fp, map_location="cpu")
print(f"cached {len(eps)} episodes")

# ---------- build heat-map --------------------------------------------------
L_MAX = 33
heat  = np.full((L_MAX, len(FAM_NAMES)), np.nan)

for L in range(L_MAX):
    ck_fp = Path(f"linear_probe_3class_direct_L{L:02d}.pth")
    if not ck_fp.exists():
        continue
    ck   = torch.load(ck_fp, map_location="cpu")
    keep = ck["kept_indices"]
    Wout = len(keep) * 3

    probe = torch.nn.Linear(
        eps[next(iter(eps))]["visual_semantic_encoding"][L].shape[-1], Wout)
    probe.load_state_dict(ck["state_dict"]); probe.eval()

    per_fam = defaultdict(list)

    for d in eps.values():
        if L not in d["visual_semantic_encoding"]: continue
        x = d["visual_semantic_encoding"][L].float()             # [T,d]
        y = torch.cat([d["symbolic_state_object_relations"],
                       d["symbolic_state_action_subgoals"]], 1)  # [T,481]
        y = y[:, keep]                                           # [T,K]

        with torch.no_grad():
            logits = probe(x)                          # [T, 3*K]
            logits = logits.view(-1, len(keep), 3)     # [T, K, 3]
            pred   = logits.argmax(-1)                 # 0 / 1 / 2


        targ = torch.zeros_like(y)
        targ[y == -1] = 0; targ[y == 0] = 1; targ[y == 1] = 2    # map to 0,1,2

        
        # ---------- per-label accuracy ----------------------------------------
        # metric = (pred == targ).float()                          # accuracy
        # valid  = torch.ones_like(metric).bool()                  # every frame counts
        # for col, gidx in enumerate(keep):
        #     f = fam(ALL[gidx])
        #     if valid[:,col].any():
        #         per_fam[f].append(metric[:,col][valid[:,col]].mean().item())
        # ----------------------------------------------------------------------

        
        # ---------- per-label binary-F1  (NA frames are ignored) -------------------
        metric = []                                            # one value per kept label
        for col in range(len(keep)):
            # mask out NA (class-0) frames
            mask = targ[:, col] != 0          # keep only 0/1 frames in the *original* scale
            if not mask.any():                # label never defined in this episode
                metric.append(np.nan)
                continue
        
            y_true_bin = (targ[mask, col] == 2).cpu().numpy().astype(int)    # 1=True, 0=False
            y_pred_bin = (pred[mask, col] == 2).cpu().numpy().astype(int)
        
            f1 = f1_score(y_true_bin, y_pred_bin,
                          average="binary", zero_division=0)
            metric.append(f1)                 # list length == K
        
        # store the F1 for family averaging
        for col, gidx in enumerate(keep):
            f = fam(ALL[gidx])
            if not np.isnan(metric[col]):
                per_fam[f].append(metric[col])
        # --------------------------------------------------------------------------


        
    for ci,f in enumerate(FAM_NAMES):
        if per_fam[f]:
            heat[L,ci] = float(np.mean(per_fam[f]))

# ---------- plot ------------------------------------------------------------
# plt.figure(figsize=(12,10))
plt.figure(figsize=(20, 15))       # width x height in inches
sns.heatmap(heat, vmin=0, vmax=1, cmap="YlGnBu",
            xticklabels=FAM_NAMES, yticklabels=list(range(L_MAX)),
            annot=True, fmt=".3f")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=45)
plt.xlabel("Predicate family"); plt.ylabel("Llama layer")

# plt.title("3-class accuracy per layer × family")
plt.title("3-class macro-F1 per layer × family")

plt.tight_layout()
plt.savefig(args.outfile, dpi=900, bbox_inches="tight")
plt.savefig("heatmap_3class_F1.pdf", bbox_inches="tight")     # or .svg
print("✓ saved", args.outfile)
