#!/usr/bin/env python
"""
Per-label diagnostics for the probes trained with train_all_probes.py

• precision/recall/F1 for the POSITIVE class (y==1)
• balanced accuracy & MCC (optional)
• optional heat-map (layer × predicate-family) with positive-recall
"""

import argparse, ast, glob, os, re, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import (precision_recall_fscore_support,
                             matthews_corrcoef, balanced_accuracy_score)
import pandas as pd

# ---------------------------------------------------------------- CLI
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir",   default="experiments/logs")
cli.add_argument("--layer",     default="all",
                 help="'all' or a single int (0‥32) to evaluate")
cli.add_argument("--success_only", action="store_true")
cli.add_argument("--csv_out",   default="per_label_metrics.csv")
cli.add_argument("--plot_heat", action="store_true",
                 help="save positive-recall heat-map as pos_recall_heatmap.png")
args   = cli.parse_args()
layers = range(33) if args.layer=="all" else [int(args.layer)]

# ------------------------------------------------ label keys & helpers
OBJ = ast.literal_eval(Path(
        "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT = ast.literal_eval(Path(
        "experiments/robot/libero/object_action_states_keys.txt").read_text())
LABELS = OBJ + ACT
def fam(lbl): return lbl.split()[0]
FAM = sorted({fam(k) for k in LABELS})

# ------------------------------------------------ optional success filter
succ = {}
succ_csv = Path(args.log_dir)/"libero_object_summary_per_episode.csv"
if succ_csv.exists():
    succ = pd.read_csv(succ_csv).set_index("EpisodeAbs")["Success"].to_dict()

# ------------------------------------------------ cache episodes once
eps = {}
for fp in tqdm(sorted(glob.glob(os.path.join(args.log_dir,"episode_*.pt"))),
               desc="cache episodes", unit="ep"):
    idx = int(re.search(r"episode_(\d+)\.pt", fp).group(1))
    if args.success_only and succ and not succ.get(idx, True):
        continue
    eps[idx] = torch.load(fp, map_location="cpu")
print(f"cached {len(eps)} episodes")

# ------------------------------------------------ evaluation
records = []
heat = np.full((33, len(FAM)), np.nan)

for L in layers:
    ck = torch.load(f"linear_probe_L{L}.pth", map_location="cpu")
    keep = ck["kept"]; K = len(keep)
    probe = torch.nn.Linear(
        eps[next(iter(eps))]["visual_semantic_encoding"][L].shape[-1], K)
    probe.load_state_dict(ck["state_dict"]); probe.eval()

    # accumulate logits / targets for this layer
    logits_all, y_all = [], []
    for d in eps.values():
        if L not in d["visual_semantic_encoding"]: continue
        x = d["visual_semantic_encoding"][L].float()
        y = torch.cat([d["symbolic_state_object_relations"],
                       d["symbolic_state_action_subgoals"]],1)[:, keep]
        with torch.no_grad(): logits = probe(x).sigmoid()
        logits_all.append(logits);  y_all.append(y)
    logits = torch.cat(logits_all)      # [N,K]
    y      = torch.cat(y_all)           # [N,K]  {-1,0,1}

    for k in range(K):
        mask = (y[:,k] != -1)
        if mask.sum()==0: continue
        targ = (y[mask,k] == 1).long().numpy()
        pred = (logits[mask,k] > 0.5).long().numpy()

        p,r,f,_ = precision_recall_fscore_support(
                        targ, pred, average="binary", pos_label=1,
                        zero_division=0)
        mcc = matthews_corrcoef(targ, pred) if len(np.unique(targ))>1 else np.nan
        bal = balanced_accuracy_score(targ, pred) if len(np.unique(targ))>1 else np.nan
        global_idx = keep[k]
        fam_k = fam(LABELS[global_idx])

        records.append(dict(layer=L, label_idx=global_idx,
                            label=LABELS[global_idx],
                            family=fam_k,
                            prec=p, recall=r, f1=f,
                            mcc=mcc, bal_acc=bal))

        # for heat-map: average positive-recall per family
        if args.plot_heat:
            fam_id = FAM.index(fam_k)
            if np.isnan(heat[L,fam_id]): heat[L,fam_id]=r
            else:                        heat[L,fam_id]=(heat[L,fam_id]+r)/2

# ------------------------------------------------ save CSV
pd.DataFrame(records).to_csv(args.csv_out, index=False)
print(f"✓ per-label metrics written to {args.csv_out}")

# ------------------------------------------------ optional heat-map
if args.plot_heat:
    import matplotlib.pyplot as plt, seaborn as sns
    plt.figure(figsize=(12,10))
    sns.heatmap(heat, vmin=0, vmax=1, cmap="YlGnBu",
                xticklabels=FAM, yticklabels=list(range(33)),
                annot=True, fmt=".2f")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=45)
    plt.xlabel("Predicate family"); plt.ylabel("Llama layer")
    plt.title("Positive-class recall per layer × family")
    plt.tight_layout(); plt.savefig("pos_recall_heatmap.png", dpi=300)
    print("✓ pos_recall_heatmap.png saved")
