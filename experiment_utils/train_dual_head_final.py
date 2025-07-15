#!/usr/bin/env python
"""
Train 33 Dual-Head multi-label linear probes on LIBERO-Object embeddings.
- Head 1 (Presence): Predicts Applicable (0/1) vs Not Applicable (-1). (Uses pos_weight)
- Head 2 (Truth): Predicts True (1) vs False (0), only when applicable.
- Reports Accuracy and F1 for both heads.
- Uses bug-free accuracy & no data-leakage split.
"""

# --------------- imports ----------------------------------------------------
import argparse, ast, glob, os, random
from pathlib import Path
import warnings # To suppress sklearn warnings if needed

import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Import F1 score
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm # Ensure tqdm is installed: pip install tqdm

# Suppress undefined metric warnings from sklearn (when a class has no predictions/labels)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# --------------- CLI --------------------------------------------------------
p = argparse.ArgumentParser(description="Train Dual-Head linear probes (Presence + Truth) w/ F1 & pos_weight")
p.add_argument("--log_dir", default="experiments/logs", help="Directory containing episode_*.pt files")
p.add_argument("--epochs", type=int, default=20, help="Number of training epochs per layer")
p.add_argument("--batch",  type=int, default=4096, help="Batch size")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
p.add_argument("--label_dir", default="experiments/robot/libero", help="Directory for label key files")
p.add_argument("--output_csv", default="probe_metrics_dual_head_final.csv", help="Output CSV filename")
p.add_argument("--output_model_prefix", default="linear_probe_dual_head_final_L", help="Prefix for saved model files")
args = p.parse_args()

# --------------- Set Seed ---------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
g_cpu = torch.Generator().manual_seed(args.seed)

# --------------- label lists ------------------------------------------------
OBJ_KEYS_PATH = Path(args.label_dir) / "object_object_relations_keys.txt"
ACT_KEYS_PATH = Path(args.label_dir) / "object_action_states_keys.txt"
if not OBJ_KEYS_PATH.exists() or not ACT_KEYS_PATH.exists():
     raise FileNotFoundError(f"Label key files not found in {args.label_dir}")
OBJ_KEYS = ast.literal_eval(OBJ_KEYS_PATH.read_text())
ACT_KEYS = ast.literal_eval(ACT_KEYS_PATH.read_text())
ALL_KEYS, NUM_LABELS = OBJ_KEYS + ACT_KEYS, len(OBJ_KEYS) + len(ACT_KEYS)
print(f"Labels: {len(OBJ_KEYS)} object + {len(ACT_KEYS)} action = {NUM_LABELS} total.")

# --------------- load episodes ---------------------------------------------
# (Loading code remains the same)
cache = {}
pt_pattern = os.path.join(args.log_dir, "episode_*.pt")
eps_files = sorted(glob.glob(pt_pattern))
if not eps_files: raise FileNotFoundError(f"No episode .pt files found matching: {pt_pattern}")
print(f"Found {len(eps_files)} episodes.")
for i, fp in enumerate(tqdm(eps_files, desc="Caching .pt files", unit="ep")):
    try:
        cache[i] = torch.load(fp, map_location="cpu")
        if not isinstance(cache[i], dict) or "visual_semantic_encoding" not in cache[i]:
            print(f"\nWarning: Invalid data structure in {fp}. Skipping.")
            del cache[i]; continue
    except Exception as e:
        print(f"\nWarning: Failed to load or validate {fp}: {e}. Skipping.")
        if i in cache: del cache[i]
if not cache: raise ValueError("No valid episodes could be loaded.")
print(f"✓ Cached {len(cache)} valid episodes.")

# --------------- episode-level split (NO DATA LEAKAGE)----------------------
# (Episode split code remains the same)
available_ep_indices = list(cache.keys())
rng = random.Random(args.seed); rng.shuffle(available_ep_indices)
num_total_eps = len(available_ep_indices)
if num_total_eps < 2: raise ValueError(f"Need >= 2 valid episodes, found {num_total_eps}")
val_len = max(1, int(0.1 * num_total_eps)); train_len = num_total_eps - val_len
if train_len < 1: raise ValueError("Train split resulted in 0 episodes.")
train_ids = available_ep_indices[val_len:]; val_ids = available_ep_indices[:val_len]
print(f"Train episodes: {len(train_ids)} • Val episodes: {len(val_ids)}")

# --------------- freq filter & presence pos_weight (TRAIN ONLY) -------------
def cat_labels(ids_list):
    ys = []
    for i in ids_list:
        if i not in cache: continue
        d = cache[i]; ys.append(torch.cat([d["symbolic_state_object_relations"], d["symbolic_state_action_subgoals"]], 1))
    return torch.cat(ys, 0) if ys else torch.empty((0, NUM_LABELS))

print("Calculating label frequencies & presence stats on TRAINING data only...")
Y_tr = cat_labels(train_ids)
print(f"  Training labels shape: {Y_tr.shape}")

# --- Filter labels based on 0/1 frequency ---
mask_tr_01 = (Y_tr != -1) # Mask for valid 0/1 labels
sum_mask_tr_01 = mask_tr_01.sum(0)
freq_01 = torch.zeros_like(sum_mask_tr_01, dtype=torch.float32)
valid_cols_tr_01 = sum_mask_tr_01 > 0
freq_01[valid_cols_tr_01] = ((Y_tr == 1) & mask_tr_01).sum(0)[valid_cols_tr_01].float() / sum_mask_tr_01[valid_cols_tr_01]
freq_01[~valid_cols_tr_01] = -1.0
keep_indices = ((freq_01 > 0.01) & (freq_01 < 0.99)).nonzero(as_tuple=True)[0]
num_kept_labels = len(keep_indices)

if num_kept_labels == 0:
     print("⚠ Warning: No labels kept after 0/1 frequency filtering. Probing all labels.")
     keep_indices = torch.arange(NUM_LABELS); num_kept_labels = NUM_LABELS
else:
    print(f"✓ Keeping {num_kept_labels}/{NUM_LABELS} labels to probe (based on train 0/1 freq 1%-99%).")

# --- Calculate pos_weight for Presence Head using ONLY kept labels from TRAIN data ---
Y_tr_kept = Y_tr[:, keep_indices] # Shape [N_train_steps, K]
presence_target_tr = (Y_tr_kept != -1) # [N_train_steps, K], True where present (0 or 1)

num_present = presence_target_tr.sum().item()
num_absent = presence_target_tr.numel() - num_present
pos_weight_p = torch.tensor(num_absent / (num_present + 1e-9), device=args.device) # Avoid division by zero
print(f"  Presence Head: #Present={num_present}, #Absent={num_absent} (Train, Kept Labels)")
print(f"  Presence Head: Calculated pos_weight = {pos_weight_p.item():.4f}")


# --------------- dataset ----------------------------------------------------
# Dataset class remains the same
class StepDS(Dataset):
    def __init__(self, layer: int, ep_list: list):
        self.layer = layer; self.samples = []
        for i in ep_list:
            if i not in cache: continue
            d = cache[i]
            if layer not in d.get("visual_semantic_encoding", {}): continue
            try: T = d["symbolic_state_object_relations"].shape[0]
            except: continue
            self.samples.extend([(i, t) for t in range(T) if t < len(d["visual_semantic_encoding"][layer])])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        i, t = self.samples[idx]; d = cache[i]
        x = d["visual_semantic_encoding"][self.layer][t].float()
        y_full = torch.cat([d["symbolic_state_object_relations"][t], d["symbolic_state_action_subgoals"][t]])
        y_kept = y_full[keep_indices]
        return x, y_kept # y_kept ∈ {-1, 0, 1}^{num_kept_labels}

# --------------- Dual-Head Probe Definition -------------------------------
class DualHeadProbe(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.presence_head = nn.Linear(input_dim, num_labels)
        self.truth_head = nn.Linear(input_dim, num_labels)
    def forward(self, x):
        return self.presence_head(x), self.truth_head(x)

# --------------- loss / metric helpers (DUAL HEAD w/ F1 & pos_weight) ------
# Note: pos_weight is passed during loss calculation for presence head
bce_loss_no_reduction = nn.BCEWithLogitsLoss(reduction='none')
# Create specific loss for presence head WITH pos_weight
bce_loss_pres = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight_p)

def run_dual_head_final(model, loader, train=False, opt=None):
    """Runs one epoch. Returns (pres_acc, truth_acc, pres_f1, truth_f1)."""
    model.train(train)
    ok_p, tot_p, ok_t, tot_t = 0, 0, 0, 0
    # Lists to store *all* predictions and targets for epoch F1 calculation
    all_pres_preds, all_pres_targets = [], []
    all_truth_preds_masked, all_truth_targets_masked = [], [] # Only store valid ones

    desc = f"Epoch {'Train' if train else 'Eval'} (Dual Head)"
    batch_iterator = tqdm(loader, desc=desc, leave=False)

    for x, y in batch_iterator: # y is [B, K] with {-1, 0, 1}
        x, y = x.to(args.device), y.to(args.device)

        presence_logits, truth_logits = model(x) # [B, K] each

        # --- Targets ---
        presence_target = (y != -1).float() # 1=present, 0=absent
        truth_target = (y == 1).float()    # 1=true, 0=false (value for -1 doesn't matter)
        truth_mask = (y != -1)             # Boolean mask where truth target is valid

        # --- Loss Calculation ---
        # Presence Loss: Use pre-calculated pos_weight via bce_loss_pres
        loss_p = bce_loss_pres(presence_logits, presence_target)

        # Truth Loss: Use masked mean
        loss_t_elementwise = bce_loss_no_reduction(truth_logits, truth_target)
        truth_valid_count = truth_mask.sum().item()
        loss_t = (loss_t_elementwise * truth_mask.float()).sum() / truth_valid_count if truth_valid_count > 0 else torch.tensor(0.0, device=args.device)

        loss = loss_p + loss_t

        if train: opt.zero_grad(); loss.backward(); opt.step()

        # --- Evaluation Metrics ---
        with torch.no_grad():
            presence_pred = (presence_logits.sigmoid() > 0.5).long() # {0, 1}
            truth_pred = (truth_logits.sigmoid() > 0.5).long()       # {0, 1}

            # Accuracy calculation (same as before)
            ok_p += (presence_pred == presence_target.long()).sum().item()
            tot_p += presence_target.numel()
            ok_t += ((truth_pred == truth_target.long()) & truth_mask).sum().item()
            tot_t += truth_valid_count

            # Store ALL predictions and targets for F1 calculation
            all_pres_preds.append(presence_pred.view(-1).cpu())
            all_pres_targets.append(presence_target.long().view(-1).cpu())
            # Store ONLY VALID truth predictions and targets
            all_truth_preds_masked.append(truth_pred[truth_mask].cpu())
            all_truth_targets_masked.append(truth_target.long()[truth_mask].cpu())


    # --- Final Metrics for Epoch ---
    final_pres_acc = ok_p / tot_p if tot_p > 0 else 0.0
    final_truth_acc = ok_t / tot_t if tot_t > 0 else 0.0

    # Calculate F1 Scores
    final_pres_f1, final_truth_f1 = 0.0, 0.0
    if tot_p > 0: # If any predictions were made
        y_true_p = torch.cat(all_pres_targets).numpy()
        y_pred_p = torch.cat(all_pres_preds).numpy()
        # Use 'binary' average for presence (present vs absent)
        final_pres_f1 = f1_score(y_true_p, y_pred_p, average='binary', pos_label=1, zero_division=0)

    if tot_t > 0: # If any valid truth predictions were made
        y_true_t = torch.cat(all_truth_targets_masked).numpy()
        y_pred_t = torch.cat(all_truth_preds_masked).numpy()
        # Use 'macro' average over labels {0, 1} for truth, like the original masking script
        final_truth_f1 = f1_score(y_true_t, y_pred_t, labels=[0, 1], average='macro', zero_division=0)

    return final_pres_acc, final_truth_acc, final_pres_f1, final_truth_f1

# --------------- training per layer ----------------------------------------
records = []
num_layers_to_train = 33

print(f"\n--- Starting Dual-Head Training Loop w/ F1 & pos_weight for {num_layers_to_train} Layers ---")

for L in range(num_layers_to_train):
    print(f"\n▶ LAYER {L:02d}")
    ds_tr = StepDS(L, train_ids); ds_va = StepDS(L, val_ids)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        print(f"  Skipping layer {L} due to empty datasets.")
        records.append(dict(layer=L, pres_acc_va=np.nan, truth_acc_va=np.nan, pres_f1_va=np.nan, truth_f1_va=np.nan, status="skipped_empty_data"))
        continue

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    input_dim = ds_tr[0][0].shape[0]
    probe = DualHeadProbe(input_dim, num_kept_labels).to(args.device)
    opt = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"  Training Dual-Head probe: InputDim={input_dim}, OutputLabels(Kept)={num_kept_labels}")

    best_val_truth_f1 = -1.0 # Track based on Truth F1 now

    for e in range(1, args.epochs + 1):
        pres_acc_tr, truth_acc_tr, pres_f1_tr, truth_f1_tr = run_dual_head_final(probe, dl_tr, train=True, opt=opt)
        pres_acc_va, truth_acc_va, pres_f1_va, truth_f1_va = run_dual_head_final(probe, dl_va, train=False)

        if e == 1 or e % 5 == 0 or e == args.epochs:
             print(f"  Ep{e:02d}| Tr Pres(Acc/F1):{pres_acc_tr:.3f}/{pres_f1_tr:.3f} Tr Truth(Acc/F1):{truth_acc_tr:.3f}/{truth_f1_tr:.3f} "
                   f"| Va Pres(Acc/F1):{pres_acc_va:.3f}/{pres_f1_va:.3f} Va Truth(Acc/F1):{truth_acc_va:.3f}/{truth_f1_va:.3f}")

        if truth_f1_va > best_val_truth_f1: best_val_truth_f1 = truth_f1_va

    final_pres_acc_va, final_truth_acc_va, final_pres_f1_va, final_truth_f1_va = pres_acc_va, truth_acc_va, pres_f1_va, truth_f1_va
    print(f"  Finished Layer {L:02d}. Final Val Pres(Acc/F1): {final_pres_acc_va:.4f}/{final_pres_f1_va:.4f}, Truth(Acc/F1): {final_truth_acc_va:.4f}/{final_truth_f1_va:.4f}")

    model_save_path = f"{args.output_model_prefix}{L:02d}.pth"
    torch.save({
        "model_type": "DualHeadProbe", "state_dict": probe.state_dict(), "layer": L,
        "kept_indices": keep_indices.tolist(), "input_dim": input_dim, "num_output_labels": num_kept_labels,
        "presence_pos_weight_used": pos_weight_p.item() # Record weight used
    }, model_save_path)
    print(f"  ✔ Saved final dual-head probe model to {model_save_path}")

    records.append(dict(layer=L,
                        pres_acc_va=final_pres_acc_va, truth_acc_va=final_truth_acc_va,
                        pres_f1_va=final_pres_f1_va, truth_f1_va=final_truth_f1_va, # Added F1
                        status="completed"))

# --------------- CSV summary -----------------------------------------------
results_df = pd.DataFrame(records)
try: results_df.to_csv(args.output_csv, index=False)
except Exception as e: print(f"\nError saving results to CSV: {e}")
print(f"\n✓ Final dual-head results (w/ F1 & pos_weight) saved to {args.output_csv}")

print("\n✅ Dual-Head Probing Script (w/ F1 & pos_weight) finished.")