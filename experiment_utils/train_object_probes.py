#!/usr/bin/env python
"""
Train 33 multi-label linear probes on LIBERO-Object embeddings
— bug-free accuracy & no data-leakage version.
"""

# --------------- imports ----------------------------------------------------
import argparse, ast, glob, os, random
from pathlib import Path

import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# --------------- CLI --------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--log_dir", default="experiments/logs")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch",  type=int, default=4096)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p = p.parse_args()

# --------------- label lists ------------------------------------------------
OBJ_KEYS = ast.literal_eval(Path(
    "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT_KEYS = ast.literal_eval(Path(
    "experiments/robot/libero/object_action_states_keys.txt").read_text())
ALL_KEYS, NUM_LABELS = OBJ_KEYS + ACT_KEYS, len(OBJ_KEYS) + len(ACT_KEYS)

# --------------- load episodes ---------------------------------------------
cache, eps = {}, sorted(glob.glob(os.path.join(p.log_dir, "episode_*.pt")))
for i, fp in enumerate(tqdm(eps, desc="caching .pt", unit="ep")):
    cache[i] = torch.load(fp, map_location="cpu")

# --------------- episode-level split ---------------------------------------
rng = random.Random(0); ep_ids = list(cache.keys()); rng.shuffle(ep_ids)
val_len = max(1, int(0.1 * len(ep_ids)))
train_ids, val_ids = ep_ids[val_len:], ep_ids[:val_len]
print(f"Train episodes: {len(train_ids)} • Val episodes: {len(val_ids)}")

# --------------- freq filter (computed on TRAIN ONLY) -----------------------
def cat_labels(ids):
    ys = []
    for i in ids:
        d = cache[i]
        ys.append(torch.cat([d["symbolic_state_object_relations"],
                             d["symbolic_state_action_subgoals"]], 1))
    return torch.cat(ys, 0)

Y_tr = cat_labels(train_ids)
mask = (Y_tr != -1); freq = ((Y_tr == 1) & mask).sum(0).float() / mask.sum(0)
freq[mask.sum(0) == 0] = -1         # columns that are all –1
keep = ((freq > .01) & (freq < .99)).nonzero(as_tuple=True)[0]
print(f"Keeping {len(keep)}/{NUM_LABELS} non-constant labels")

# --------------- dataset ----------------------------------------------------
class StepDS(Dataset):
    def __init__(self, layer, ep_list):
        self.samples = [(i, t) for i in ep_list
                        for t in range(cache[i]
                         ["symbolic_state_object_relations"].shape[0])
                        if layer in cache[i]["visual_semantic_encoding"]]
        self.layer = layer
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        i,t = self.samples[idx]; d = cache[i]
        x = d["visual_semantic_encoding"][self.layer][t].float()
        y = torch.cat([d["symbolic_state_object_relations"][t],
                       d["symbolic_state_action_subgoals"][t]])[keep]
        return x, y                     # y ∈ {-1,0,1}^{|keep|}

# --------------- loss / metric helpers -------------------------------------
bce = nn.BCEWithLogitsLoss(reduction="none")

def run(model, loader, train=False, opt=None):
    model.train(train)
    ok = tot = 0; preds_all = []; targs_all = []
    for x,y in loader:
        x,y = x.to(p.device), y.to(p.device)
        logits = model(x); mask = (y != -1)
        target = (y == 1).float()
        if train:
            loss = (bce(logits,target)*mask.float()).sum() / mask.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = (logits.sigmoid() > .5).long()     # {0,1}
            ok  += (pred[mask] == target[mask]).sum().item()
            tot += mask.sum().item()
            preds_all.append(pred[mask].cpu())
            targs_all.append(target[mask].cpu())
    acc = ok / tot if tot else 0.0
    f1  = f1_score(torch.cat(targs_all).numpy(),
                   torch.cat(preds_all).numpy(),
                   average="macro", zero_division=0) if tot else 0.0
    return acc, f1

# --------------- training per layer ----------------------------------------
records = []
for L in range(33):
    ds_tr = StepDS(L, train_ids); ds_va = StepDS(L, val_ids)
    if len(ds_tr)==0 or len(ds_va)==0:
        records.append(dict(layer=L, val_acc=np.nan, val_f1=np.nan)); continue
    dl_tr = DataLoader(ds_tr, p.batch, True,  num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, p.batch, False, num_workers=4, pin_memory=True)

    probe = nn.Linear(ds_tr[0][0].shape[0], len(keep)).to(p.device)
    opt   = optim.AdamW(probe.parameters(), 1e-3, weight_decay=1e-4)

    for e in range(1, p.epochs+1):
        run(probe, dl_tr, True, opt)
    acc_va, f1_va = run(probe, dl_va, False)

    torch.save({"state_dict":probe.state_dict(),
                "layer":L, "kept":keep.tolist()}, f"linear_probe_L{L}.pth")
    print(f"Layer {L:02d}  acc={acc_va:.3f}  F1={f1_va:.3f}")
    records.append(dict(layer=L, val_acc=acc_va, val_f1=f1_va))

pd.DataFrame(records).to_csv("probe_metrics_clean.csv", index=False)
print("\n✓  probe_metrics_clean.csv written")
