#!/usr/bin/env python
"""
Train multi-label linear probes on LIBERO-Spatial embeddings
===========================================================

Usage examples
--------------
# probe every layer (0-32)
python train_spatial_probes.py

# probe only layer 32
python train_spatial_probes.py --layers 32

# probe a subset and exclude failed roll-outs
python train_spatial_probes.py --layers 0,8,16,32 \
    --exclude_eps 11,13,18,21,26,27,28,43,51,52,53,60

Outputs
-------
linear_probe_L00.pth … linear_probe_L32.pth
probe_metrics_spatial.csv   (one row per probed layer)
"""

# ---------------------------------------------------------------- imports
import argparse, ast, glob, os, random
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

# --------------------------------------------------- CLI
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir", default="experiments/logs",
                 help="folder containing episode_*.pt")
cli.add_argument("--epochs", type=int, default=20)
cli.add_argument("--batch",  type=int, default=4096)
cli.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
cli.add_argument("--layers", default="all",
                 help="comma-sep list (e.g. 32 or 0,8,16,32); "
                      "'all' = every layer 0-32")
cli.add_argument("--exclude_eps", default="",
                 help="comma-sep episode ids to drop, e.g. 11,13 or range 50-60")
args = cli.parse_args()

# ------------------------- layer list
LAYERS = (list(range(33)) if args.layers.strip().lower() == "all"
          else [int(s) for s in args.layers.split(",")])
assert all(0 <= L <= 32 for L in LAYERS), "layer index must be 0-32"
print("Probing layers:", LAYERS)

# ------------------------- episode exclusion helper
def parse_exclusions(spec: str):
    out = set()
    if spec.strip():
        for tok in spec.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = map(int, tok.split("-")); out.update(range(a, b + 1))
            else:
                out.add(int(tok))
    return out

EXCLUDE = parse_exclusions(args.exclude_eps)
print("Excluding episodes:", sorted(EXCLUDE) if EXCLUDE else "none")

rng = random.Random(0)                           # reproducible shuffle

# ------------------------- label keys
OBJ_PATH = Path("experiments/robot/libero/spatial_object_relations_keys.txt")
ACT_PATH = Path("experiments/robot/libero/spatial_action_states_keys.txt")
OBJ_KEYS = ast.literal_eval(OBJ_PATH.read_text())
ACT_KEYS = ast.literal_eval(ACT_PATH.read_text())
NUM_LABELS = len(OBJ_KEYS) + len(ACT_KEYS)
print(f"Spatial suite: {len(OBJ_KEYS)} object-relations + "
      f"{len(ACT_KEYS)} action-states = {NUM_LABELS} labels")

# ------------------------- load .pt roll-outs
cache = {}
all_files = sorted(glob.glob(os.path.join(args.log_dir, "episode_*.pt")))
pt_files  = [fp for fp in all_files
             if int(Path(fp).stem.split("_")[1]) not in EXCLUDE]
if not pt_files:
    raise FileNotFoundError("No episode_*.pt after applying exclusions")
print("→ caching .pt files …")
for idx, fp in enumerate(tqdm(pt_files, unit="ep")):
    cache[idx] = torch.load(fp, map_location="cpu")

# ------------------------- train/val split (episode-level)
ep_ids = list(cache.keys()); rng.shuffle(ep_ids)
val_len = max(1, int(0.10 * len(ep_ids)))
train_ids, val_ids = ep_ids[val_len:], ep_ids[:val_len]
print(f"Train episodes: {len(train_ids)}   Val episodes: {len(val_ids)}")

# ------------------------- freq filter & pos_weight
# -------- decide which labels vary ANYWHERE (good + failed) -------
def quick_stack(fps):
    return torch.cat([
        torch.cat([
            torch.load(fp, map_location="cpu")["symbolic_state_object_relations"],
            torch.load(fp, map_location="cpu")["symbolic_state_action_subgoals"]
        ], dim=1)
        for fp in fps
    ], dim=0)

# ------------ decide keep by logical any() rather than mean ----------
Y_full = quick_stack(all_files)

pos_any = (Y_full == 1).any(0)          # True if a 1 ever occurs
neg_any = (Y_full == 0).any(0)          # True if a 0 ever occurs
keep    = (pos_any & neg_any).nonzero(as_tuple=True)[0]

print(f"Labels that show both 0 and 1: {len(keep)}/{NUM_LABELS}")
if len(keep) == 0:
    raise RuntimeError("Even across all episodes no label flips value.")


def stack_labels(ids):
    return torch.cat([
        torch.cat([cache[i]["symbolic_state_object_relations"],
                   cache[i]["symbolic_state_action_subgoals"]], 1)
        for i in ids
    ], 0)
    
Y_train = stack_labels(train_ids)

pos_cnt = Y_train.sum(0).float(); neg_cnt = Y_train.shape[0] - pos_cnt
eps = 1.0
pos_weight_full = (neg_cnt + eps) / (pos_cnt + eps)   # finite even if pos_cnt==0
POS_W = torch.as_tensor(pos_weight_full[keep]).clamp(max=20)

# ------------------------- dataset
class StepDS(Dataset):
    """Return (x, y) with y∈{0,1}^{|keep|} (Spatial suite has no -1)."""
    def __init__(self, layer, eps):
        self.layer = layer
        self.samples = [(i, t)
                        for i in eps
                        if layer in cache[i]["visual_semantic_encoding"]
                        for t in range(cache[i]["visual_semantic_encoding"][layer].shape[0])]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        i, t = self.samples[idx]; d = cache[i]
        x = d["visual_semantic_encoding"][self.layer][t].float()
        y = torch.cat([d["symbolic_state_object_relations"][t],
                       d["symbolic_state_action_subgoals"][t]])[keep].float()
        return x, y

# ------------------------- helpers
bce = nn.BCEWithLogitsLoss(pos_weight=POS_W.to(args.device))

def run_epoch(model, loader, train=False, opt=None, thresh=.5):
    model.train(train)
    ok = tot = 0
    probs_all, targs_all, preds_all = [], [], []
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        logits = model(x)
        if train:
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            probs = logits.sigmoid()
            pred  = (probs > thresh).float()
            ok  += (pred == y).sum().item(); tot += y.numel()
            probs_all.append(probs.cpu()); preds_all.append(pred.cpu()); targs_all.append(y.cpu())
    y_true = torch.cat(targs_all).numpy()
    y_pred = torch.cat(preds_all).numpy()
    y_prob = torch.cat(probs_all).numpy()
    acc = ok / tot
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ap  = average_precision_score(y_true, y_prob, average="macro")
    return acc, f1, ap

# ------------------------- training loop
records = []
for L in LAYERS:
    ds_tr, ds_va = StepDS(L, train_ids), StepDS(L, val_ids)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        records.append(dict(layer=L, val_acc=np.nan, val_f1=np.nan, val_ap=np.nan))
        continue

    dl_tr = DataLoader(ds_tr, args.batch, True,  num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, args.batch, False, num_workers=4, pin_memory=True)

    probe = nn.Linear(ds_tr[0][0].shape[0], len(keep)).to(args.device)
    opt   = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(args.epochs):
        run_epoch(probe, dl_tr, True, opt)
    acc_va, f1_va, ap_va = run_epoch(probe, dl_va, False)

    torch.save({"state_dict": probe.state_dict(),
                "layer": L, "kept": keep.tolist()},
               f"linear_probe_L{L:02d}.pth")

    print(f"L{L:02d}  acc={acc_va:.3f}  F1={f1_va:.3f}  AP={ap_va:.3f}")
    records.append(dict(layer=L, val_acc=acc_va, val_f1=f1_va, val_ap=ap_va))

pd.DataFrame(records).to_csv("probe_metrics_spatial.csv", index=False)
print("\n✓ probe_metrics_spatial.csv written")
