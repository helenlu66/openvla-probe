#!/usr/bin/env python
"""
04_family_auprc.py

Pool logits & targets over all labels in the same predicate **family**
and compute family-level AUPRC.  Requires the pickles produced by
01_collect_logits.py.

Run:
    python analysis/04_family_auprc.py
"""
import pickle, argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import average_precision_score
from util_data import LABELS, family          # same helper as before

cli = argparse.ArgumentParser()
cli.add_argument("--pickle_dir", default="analysis",
                 help="where logits_XX.pkl and targets_XX.pkl live")
cli.add_argument("--out_csv",   default="analysis/family_auprc.csv")
args = cli.parse_args()

# -------------------------------------------------------------------------
# 1️⃣  load & concatenate logits / targets for ALL layers
# -------------------------------------------------------------------------
logits_layers  = []
targets_layers = []
for L in range(33):
    p_log = Path(args.pickle_dir)/f"logits_{L:02d}.pkl"
    p_tar = Path(args.pickle_dir)/f"targets_{L:02d}.pkl"
    if not p_log.exists():   continue        # layer skipped during probing
    logits_layers.append ( pickle.load(open(p_log,"rb")) )  # dict[ep]→Tensor
    targets_layers.append( pickle.load(open(p_tar,"rb")) )

# Same keep-order in every pickle → get it once
K = next(iter(logits_layers[0].values())).shape[1]

# -------------------------------------------------------------------------
# 2️⃣  build mapping  column-index → predicate-family
# -------------------------------------------------------------------------
ck_file = sorted(Path(args.pickle_dir).parent.glob("linear_probe_L*.pth"))[0]
keep    = torch.load(ck_file, map_location="cpu")["kept"]      # length K

fam_for_k     = {k: family(LABELS[keep[k]]) for k in range(K)}
family_names  = sorted(set(fam_for_k.values()))                # ← NEW
# -------------------------------------------------------------------------
# 3️⃣  accumulate per-family logits / targets
# -------------------------------------------------------------------------
pool = {f: {"logits": [], "targs": []} for f in family_names}


for logits_ep, targs_ep in zip(logits_layers, targets_layers):
    for ep_id in logits_ep.keys():                 # loop episodes
        log = logits_ep [ep_id]         # [T,K]
        tar = targs_ep[ep_id]           # [T,K]  {-1,0,1}
        for k in range(K):
            fam = fam_for_k[k]
            pool[fam]["logits"].append( log[:,k] )
            pool[fam]["targs"].append( tar[:,k] )

rows = []
for fam, d in pool.items():
    if not d["logits"]:         # family absent in kept set (unlikely)
        continue
    log_f = torch.cat(d["logits"]).numpy()         # 1-D
    tar_f = torch.cat(d["targs"]).numpy()          # 1-D {-1,0,1}

    valid = tar_f != -1
    if valid.sum() == 0:        # no defined frames
        continue
    y_true = (tar_f[valid] == 1).astype(int)
    y_prob = log_f[valid]

    prior   = y_true.mean()                         # P(+)_fam
    auprc   = average_precision_score(y_true, y_prob)
    lift    = auprc / (prior+1e-9)

    rows.append(dict(family=fam,
                     prior=prior,
                     auprc=auprc,
                     lift=lift))

# -------------------------------------------------------------------------
# 4️⃣  save & pretty-print
# -------------------------------------------------------------------------
df = pd.DataFrame(rows).sort_values("auprc")
df.to_csv(args.out_csv, index=False)
print("✓ family-level metrics written to", args.out_csv, "\n")

pd.set_option("display.float_format", "{:6.3f}".format)
print(df.to_string(index=False))
