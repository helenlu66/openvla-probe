#!/usr/bin/env python
"""
Train 33 multi-label linear probes on LIBERO-Spatial embeddings
— bug-free version (train-split freq filter, per-episode split, no data leakage).

Output:
    • linear_probe_L00.pth … linear_probe_L32.pth
    • probe_metrics_spatial.csv   (one row per layer)
"""

# -------------------------------------------------------------------- imports
import argparse, ast, glob, os, random
from pathlib import Path

import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# -------------------------------------------------------------------- CLI
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir", default="experiments/logs",
                 help="folder containing episode_*.pt")
cli.add_argument("--epochs",  type=int, default=20)
cli.add_argument("--batch",   type=int, default=4096)
cli.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
args = cli.parse_args()

rng = random.Random(0)                       # reproducible episode shuffle

# -------------------------------------------------------------------- label keys
OBJ_PATH = Path("experiments/robot/libero/spatial_object_relations_keys.txt")
ACT_PATH = Path("experiments/robot/libero/spatial_action_states_keys.txt")
OBJ_KEYS = ast.literal_eval(OBJ_PATH.read_text())
ACT_KEYS = ast.literal_eval(ACT_PATH.read_text())
ALL_KEYS = OBJ_KEYS + ACT_KEYS
NUM_LABELS = len(ALL_KEYS)
print(f"Spatial suite: {len(OBJ_KEYS)} object-relations + "
      f"{len(ACT_KEYS)} action-states = {NUM_LABELS} labels")

# -------------------------------------------------------------------- cache roll-outs
cache = {}
pt_files = sorted(glob.glob(os.path.join(args.log_dir, "episode_*.pt")))
if not pt_files:
    raise FileNotFoundError(f"No episode_*.pt in {args.log_dir}")
print("→ caching .pt files …")
for idx, fp in enumerate(tqdm(pt_files, unit="ep")):
    cache[idx] = torch.load(fp, map_location="cpu")

# -------------------------------------------------------------------- episode split
ep_ids = list(cache.keys()); rng.shuffle(ep_ids)
val_len  = max(1, int(0.10 * len(ep_ids)))
train_ids, val_ids = ep_ids[val_len:], ep_ids[:val_len]
print(f"Train episodes: {len(train_ids)}   Val episodes: {len(val_ids)}")

# -------------------------------------------------------------------- freq filter (TRAIN ONLY)
def concat_labels(ids):
    ys = [ torch.cat([cache[i]["symbolic_state_object_relations"],
                      cache[i]["symbolic_state_action_subgoals"]], 1)
           for i in ids ]
    return torch.cat(ys, 0)

Y_train = concat_labels(train_ids)                     # [N_train_frames,  K]
freq    = Y_train.float().mean(0)                      # P(label == 1)
keep    = ((freq > 0.01) & (freq < 0.99)).nonzero(as_tuple=True)[0]
print(f"Keeping {len(keep)}/{NUM_LABELS} non-constant labels")

# -------------------------------------------------------------------- dataset
class StepDS(Dataset):
    """(x, y) where y ∈ {0,1}^{|keep|}  — no -1 in Spatial suite."""
    def __init__(self, layer: int, ep_list):
        self.layer = layer
        self.samples = [(i, t)
                        for i in ep_list
                        if layer in cache[i]["visual_semantic_encoding"]
                        for t in range(cache[i]["visual_semantic_encoding"][layer].shape[0])]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        i, t = self.samples[idx]; d = cache[i]
        x = d["visual_semantic_encoding"][self.layer][t].float()
        y = torch.cat([d["symbolic_state_object_relations"][t],
                       d["symbolic_state_action_subgoals"][t]])[keep].float()
        return x, y                     # y already 0/1

# -------------------------------------------------------------------- helpers
bce = nn.BCEWithLogitsLoss()

def run_epoch(model, loader, train=False, opt=None):
    model.train(train)
    ok = tot = 0; preds_all = []; targs_all = []
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        logits = model(x)
        if train:
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = (logits.sigmoid() > 0.5).float()
            ok  += (pred == y).sum().item()
            tot += y.numel()
            preds_all.append(pred.cpu()); targs_all.append(y.cpu())
    acc = ok / tot
    f1  = f1_score(torch.cat(targs_all).numpy(),
                   torch.cat(preds_all).numpy(),
                   average="macro", zero_division=0)
    return acc, f1

# -------------------------------------------------------------------- training loop
records = []
for L in range(33):
    ds_tr = StepDS(L, train_ids)
    ds_va = StepDS(L, val_ids)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        records.append(dict(layer=L, val_acc=np.nan, val_f1=np.nan))
        continue

    dl_tr = DataLoader(ds_tr, args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, args.batch, shuffle=False, num_workers=4, pin_memory=True)

    probe = nn.Linear(ds_tr[0][0].shape[0], len(keep)).to(args.device)
    opt   = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(1, args.epochs+1):
        run_epoch(probe, dl_tr, True, opt)
    acc_va, f1_va = run_epoch(probe, dl_va, False)

    torch.save({"state_dict": probe.state_dict(),
                "layer": L,
                "kept":  keep.tolist()},
               f"linear_probe_L{L:02d}.pth")
    print(f"L{L:02d}  acc={acc_va:.3f}  F1={f1_va:.3f}")

    records.append(dict(layer=L, val_acc=acc_va, val_f1=f1_va))

pd.DataFrame(records).to_csv("probe_metrics_spatial.csv", index=False)
print("\n✓ probe_metrics_spatial.csv written")
