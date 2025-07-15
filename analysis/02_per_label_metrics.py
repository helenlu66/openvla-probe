#!/usr/bin/env python
"""
Compute per-label:
    • precision / recall / F1 (positive class)
    • AUPRC
    • MCC & balanced-accuracy
Outputs one CSV per layer and a big combined CSV.
"""
import argparse, pickle, numpy as np, pandas as pd, tqdm, torch
from sklearn.metrics import (precision_recall_fscore_support,
                             average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score)
from util_data import LABELS, family

cli = argparse.ArgumentParser()
cli.add_argument("--out_csv", default="analysis/per_label_metrics.csv")
args = cli.parse_args()

all_rows = []
for L in range(33):
    try:
        logits = pickle.load(open(f"analysis/logits_{L:02d}.pkl","rb"))
        targets= pickle.load(open(f"analysis/targets_{L:02d}.pkl","rb"))
    except FileNotFoundError: continue
    keep_any_ep = next(iter(targets.values())).shape[1]

    # concat over episodes
    log  = torch.cat(list(logits.values()))           # [N,K]
    targ = torch.cat(list(targets.values()))          # [N,K] {-1,0,1}

    for k in range(keep_any_ep):
        m = targ[:,k]!=-1
        if m.sum()==0: continue
        y = (targ[m,k]==1).numpy().astype(int)
        p = (log [m,k]>0.5).numpy().astype(int)
        prec,rec,f1,_ = precision_recall_fscore_support(
                            y,p,average="binary",zero_division=0)
        auprc = average_precision_score(y,log[m,k].numpy())
        mcc   = matthews_corrcoef       (y,p) if len(np.unique(y))>1 else np.nan
        bal   = balanced_accuracy_score (y,p) if len(np.unique(y))>1 else np.nan
        ckpt = torch.load(f"linear_probe_L{L}.pth", map_location="cpu")
        keep = ckpt["kept"]                # or "kept_indices", depending on your script
        gidx = keep[k]
        all_rows.append(dict(layer=L, label_idx=gidx,
                             label=LABELS[gidx],
                             family=family(LABELS[gidx]),
                             prec=prec, recall=rec, f1=f1,
                             auprc=auprc, mcc=mcc, bal_acc=bal))
pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
print("✓ wrote", args.out_csv)
