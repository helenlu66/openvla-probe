#!/usr/bin/env python
"""
Train multi-label linear probes on LIBERO-Object embeddings
----------------------------------------------------------
"""

# ---------- imports ----------------------------------------------------------
import argparse, ast, glob, os, random
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

# ---------- CLI --------------------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--log_dir", default="experiments/logs",
                 help="folder containing episode_*.pt")
cli.add_argument("--epochs", type=int, default=20)
cli.add_argument("--batch",  type=int, default=4096)
cli.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
cli.add_argument("--layers", default="all",
                 help="comma-sep list, e.g. 32 or 0,8,16,32; 'all' = 0-32")
cli.add_argument("--exclude_eps", default="",
                 help="comma-sep episode ids or ranges to drop, e.g. 11,13 or 50-60")
args = cli.parse_args()

# ---------- helper: parse layer list ----------------------------------------
LAYERS = (list(range(33)) if args.layers.strip().lower() == "all"
          else [int(x) for x in args.layers.split(",")])
assert all(0 <= L <= 32 for L in LAYERS), "layer index must be 0-32"
print("Probing layers:", LAYERS)

# ---------- helper: parse exclusions ----------------------------------------
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

rng = random.Random(0)        # reproducible shuffle

# ---------- label lists ------------------------------------------------------
OBJ_KEYS = ast.literal_eval(
    Path("experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT_KEYS = ast.literal_eval(
    Path("experiments/robot/libero/object_action_states_keys.txt").read_text())
NUM_LABELS = len(OBJ_KEYS) + len(ACT_KEYS)
print(f"Object suite: {len(OBJ_KEYS)} relations + {len(ACT_KEYS)} action-states = {NUM_LABELS}")

# ---------- load episodes ----------------------------------------------------
cache = {}
all_files = sorted(glob.glob(os.path.join(args.log_dir, "episode_*.pt")))
pt_files  = [fp for fp in all_files
             if int(Path(fp).stem.split("_")[1]) not in EXCLUDE]
if not pt_files:
    raise FileNotFoundError("No episode_*.pt after applying exclusions")
print("→ caching .pt files …")
for idx, fp in enumerate(tqdm(pt_files, unit="ep")):
    cache[idx] = torch.load(fp, map_location="cpu")

# ---------- split episodes ---------------------------------------------------
ep_ids = list(cache.keys()); rng.shuffle(ep_ids)
val_len = max(1, int(0.10 * len(ep_ids)))
train_ids, val_ids = ep_ids[val_len:], ep_ids[:val_len]
print(f"Train episodes: {len(train_ids)} • Val episodes: {len(val_ids)}")

# ---------- helper to stack labels ------------------------------------------
def stack_labels(ids):
    return torch.cat([
        torch.cat([cache[i]["symbolic_state_object_relations"],
                   cache[i]["symbolic_state_action_subgoals"]], 1)
        for i in ids
    ], 0)  # shape (N_frames, NUM_LABELS)

# ---------- choose keep columns (train ∪ val) --------------------------------
Y_full = stack_labels(ep_ids)
mask_full = (Y_full != -1)
pos_any   = ((Y_full == 1) & mask_full).any(0)
neg_any   = ((Y_full == 0) & mask_full).any(0)
keep      = (pos_any & neg_any).nonzero(as_tuple=True)[0]
print(f"Labels with both 0 and 1 somewhere: {len(keep)}/{NUM_LABELS}")
if len(keep) == 0:
    raise RuntimeError("No label flips value across remaining episodes.")

# ---------- pos_weight from TRAIN split --------------------------------------
Y_tr = stack_labels(train_ids)
mask_tr = (Y_tr != -1)
pos_cnt = ((Y_tr == 1) & mask_tr).sum(0).float()
neg_cnt = ((Y_tr == 0) & mask_tr).sum(0).float()
eps = 1.0
pos_weight_full = (neg_cnt + eps) / (pos_cnt + eps)  # finite
POS_W = torch.as_tensor(pos_weight_full[keep]).clamp(max=20)


# ---------- majority-class baseline on TRAIN → evaluate on VAL -------------
with torch.no_grad():
    # 众数（0/1）按列计算，只看 TRAIN + keep 列
    maj = ((Y_tr == 1) & mask_tr).sum(0) > ((Y_tr == 0) & mask_tr).sum(0)  # Bool[K]
    maj = maj[keep].float()                                                # 对应 keep 列

    # 在 VAL split 上评估
    Y_val = stack_labels(val_ids)
    mask_val = (Y_val != -1)[:, keep]
    tgt_val  = (Y_val[:, keep] == 1).float()

    pred_val = maj.unsqueeze(0).expand_as(tgt_val)  # 所有帧都预测众数

    acc_maj = ((pred_val == tgt_val)[mask_val]).float().mean().item()
    f1_maj  = f1_score(tgt_val[mask_val].numpy(),
                       pred_val[mask_val].numpy(),
                       average="macro", zero_division=0)

print(f"Majority-class baseline  acc={acc_maj:.3f}  F1={f1_maj:.3f}")
# ---------------------------------------------------------------------------

# ---------- dataset ----------------------------------------------------------
# ---------------------------------------------------------------------------
# 🔧 1.  keep the ORIGINAL StepDS (no self.split / row_idx logic needed)
class StepDS(Dataset):
    """Return (x, y) where y ∈ {-1,0,1}^{|keep|}"""
    def __init__(self, layer, eps):
        self.layer = layer
        self.samples = [
            (i, t)
            for i in eps
            if layer in cache[i]["visual_semantic_encoding"]
            for t in range(cache[i]["visual_semantic_encoding"][layer].shape[0])
        ]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        i, t = self.samples[idx]; d = cache[i]
        x = d["visual_semantic_encoding"][self.layer][t].float()
        y = torch.cat([d["symbolic_state_object_relations"][t],
                       d["symbolic_state_action_subgoals"][t]])[keep]
        return x, y
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 🔧 2.  column-wise shuffle by MODIFYING THE LABEL MATRICES IN CACHE
def shuffled_like(mat, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = mat.clone()
    for c in range(out.shape[1]):
        idx = torch.randperm(out.shape[0], generator=g)
        out[:, c] = out[idx, c]
    return out

USE_SHUFFLE = False
if USE_SHUFFLE:
    print("⚠️  Using column-shuffled labels for sanity-check")
    # shuffle once, then overwrite each episode tensors
    Y_full_shuf = shuffled_like(Y_full)            # shape (all_frames, K)
    cursor = 0
    for i in ep_ids:                               # restore per-episode
        n = cache[i]["symbolic_state_object_relations"].shape[0]
        shuf_slice = Y_full_shuf[cursor:cursor+n]
        # write back separated into the two blocks
        cache[i]["symbolic_state_object_relations"][:, :] = \
            shuf_slice[:, :len(OBJ_KEYS)]
        cache[i]["symbolic_state_action_subgoals"][:, :]  = \
            shuf_slice[:, len(OBJ_KEYS):]
        cursor += n
# ---------------------------------------------------------------------------


# ---------- helpers ----------------------------------------------------------
bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=POS_W.to(args.device))

def run_epoch(model, loader, train=False, opt=None, thresh=0.5):
    model.train(train)
    ok = tot = 0
    p_all, y_all, pred_all = [], [], []
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        logits = model(x); mask = (y != -1)
        target = (y == 1).float()
        if train:
            loss = ((bce(logits, target) * mask.float()).sum() / mask.sum())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            probs = logits.sigmoid()
            pred  = (probs > thresh).long()
            ok  += (pred[mask] == target[mask]).sum().item()
            tot += mask.sum().item()
            p_all.append(probs[mask].cpu())
            pred_all.append(pred[mask].cpu())
            y_all.append(target[mask].cpu())
    if tot == 0:
        return 0.0, 0.0, 0.0
    acc = ok / tot
    y_true = torch.cat(y_all).numpy()
    y_pred = torch.cat(pred_all).numpy()
    y_prob = torch.cat(p_all).numpy()
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ap = average_precision_score(y_true, y_prob, average="macro")
    return acc, f1, ap

# ---------- training ---------------------------------------------------------
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

pd.DataFrame(records).to_csv("probe_metrics_object.csv", index=False)
print("\n✓ probe_metrics_object.csv written")
