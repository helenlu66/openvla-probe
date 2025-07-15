#!/usr/bin/env python
"""
Train 33 multi-label linear probes on LIBERO-Object embeddings.
This version attempts to predict 3 classes per label: -1 (NA), 0 (False), 1 (True).
- Uses bug-free accuracy & no data-leakage split.
- Includes basic class weighting for CrossEntropyLoss.
"""

# --------------- imports ----------------------------------------------------
import argparse, ast, glob, os, random
from pathlib import Path
import warnings

import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report # For detailed metrics
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --------------- CLI --------------------------------------------------------
p = argparse.ArgumentParser(description="Train Direct 3-Class linear probes (-1, 0, 1)")
p.add_argument("--log_dir", default="experiments/logs", help="Directory containing episode_*.pt files")
p.add_argument("--epochs", type=int, default=20, help="Number of training epochs per layer")
p.add_argument("--batch",  type=int, default=2048, help="Batch size (might need smaller for 3-class)") # Adjusted default
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
p.add_argument("--label_dir", default="experiments/robot/libero", help="Directory for label key files")
p.add_argument("--output_csv", default="probe_metrics_3class_direct.csv", help="Output CSV filename")
p.add_argument("--output_model_prefix", default="linear_probe_3class_direct_L", help="Prefix for saved model files")
args = p.parse_args()

# --------------- Set Seed ---------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

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
cache = {}
pt_pattern = os.path.join(args.log_dir, "episode_*.pt")
eps_files = sorted(glob.glob(pt_pattern))
if not eps_files: raise FileNotFoundError(f"No episode .pt files found matching: {pt_pattern}")
print(f"Found {len(eps_files)} episodes.")
for i, fp in enumerate(tqdm(eps_files, desc="Caching .pt files", unit="ep")):
    try:
        cache[i] = torch.load(fp, map_location="cpu")
        if not isinstance(cache[i], dict) or "visual_semantic_encoding" not in cache[i]:
            del cache[i]; continue
    except Exception:
        if i in cache: del cache[i]; continue
if not cache: raise ValueError("No valid episodes could be loaded.")
print(f"✓ Cached {len(cache)} valid episodes.")

# --------------- episode-level split (NO DATA LEAKAGE)----------------------
available_ep_indices = list(cache.keys())
rng = random.Random(args.seed); rng.shuffle(available_ep_indices)
num_total_eps = len(available_ep_indices)
if num_total_eps < 2: raise ValueError(f"Need >= 2 valid episodes, found {num_total_eps}")
val_len = max(1, int(0.1 * num_total_eps)); train_len = num_total_eps - val_len
if train_len < 1: raise ValueError("Train split resulted in 0 episodes.")
train_ids = available_ep_indices[val_len:]; val_ids = available_ep_indices[:val_len]
print(f"Train episodes: {len(train_ids)} • Val episodes: {len(val_ids)}")

# --------------- freq filter & CLASS WEIGHTS (TRAIN ONLY) -------------------
def cat_labels(ids_list, kept_indices_to_use=None):
    """Concatenates labels. If kept_indices_to_use, it filters."""
    ys = []
    for i in ids_list:
        if i not in cache: continue
        d = cache[i]
        try:
            y_full = torch.cat([d["symbolic_state_object_relations"], d["symbolic_state_action_subgoals"]], 1)
            if kept_indices_to_use is not None:
                ys.append(y_full[:, kept_indices_to_use])
            else:
                ys.append(y_full)
        except: continue # Skip episode if labels are malformed
    return torch.cat(ys, 0) if ys else torch.empty((0, NUM_LABELS if kept_indices_to_use is None else len(kept_indices_to_use)))

print("Calculating 0/1 label frequencies for filtering (TRAIN data only)...")
Y_tr_full = cat_labels(train_ids) # Get all labels first for filtering decision
print(f"  Full training labels shape: {Y_tr_full.shape}")

mask_tr_01 = (Y_tr_full != -1)
sum_mask_tr_01 = mask_tr_01.sum(0)
freq_01 = torch.zeros_like(sum_mask_tr_01, dtype=torch.float32)
valid_cols_tr_01 = sum_mask_tr_01 > 0
freq_01[valid_cols_tr_01] = ((Y_tr_full == 1) & mask_tr_01).sum(0)[valid_cols_tr_01].float() / sum_mask_tr_01[valid_cols_tr_01]
freq_01[~valid_cols_tr_01] = -1.0 # if only -1, mark to drop for 0/1
keep_indices = ((freq_01 > 0.01) & (freq_01 < 0.99)).nonzero(as_tuple=True)[0]
num_kept_labels = len(keep_indices)

if num_kept_labels == 0:
     print("⚠ Warning: No labels kept after 0/1 frequency filtering. Probing all labels.")
     keep_indices = torch.arange(NUM_LABELS); num_kept_labels = NUM_LABELS
else:
    print(f"✓ Keeping {num_kept_labels}/{NUM_LABELS} labels to probe (based on train 0/1 freq 1%-99%).")

# --- Calculate CLASS WEIGHTS for CrossEntropyLoss (using kept labels from TRAIN data) ---
print("Calculating class weights for the 3-class problem (TRAIN data, kept labels)...")
Y_tr_kept = cat_labels(train_ids, kept_indices_to_use=keep_indices) # Shape [N_train_steps, K]
# Map to 0, 1, 2
class_0_count = (Y_tr_kept == -1).sum().item() # N/A
class_1_count = (Y_tr_kept == 0).sum().item()  # False
class_2_count = (Y_tr_kept == 1).sum().item()  # True
total_count = Y_tr_kept.numel()

print(f"  Class Counts (Train, Kept): N/A(-1) -> {class_0_count}, False(0) -> {class_1_count}, True(1) -> {class_2_count}")

if total_count == 0:
    print("Warning: No data to calculate class weights. Using equal weights.")
    class_weights = torch.tensor([1.0, 1.0, 1.0], device=args.device)
else:
    # Inverse frequency weighting: weight = total_samples / (num_classes * count_per_class)
    # Smoothed for stability if a class is very rare or absent (shouldn't happen for kept labels)
    weights = [total_count / (3 * (c + 1e-6)) for c in [class_0_count, class_1_count, class_2_count]]
    class_weights = torch.tensor(weights, device=args.device, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 3 # Normalize to sum to num_classes for stability
print(f"  Calculated CE class_weights: {class_weights.cpu().numpy()}")

# --------------- dataset ----------------------------------------------------
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

# --------------- loss / metric helpers (DIRECT 3-CLASS VERSION) ------------
# Use CrossEntropyLoss WITH the calculated class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Mapping from original labels {-1, 0, 1} to class indices {0, 1, 2}
def map_labels_to_indices(y_batch_kept): # y_batch_kept is [B, K]
    target_indices = torch.zeros_like(y_batch_kept, dtype=torch.long)
    target_indices[y_batch_kept == -1] = 0 # N/A
    target_indices[y_batch_kept == 0]  = 1 # False
    target_indices[y_batch_kept == 1]  = 2 # True
    return target_indices

def run_3class_direct(model, loader, train=False, opt=None):
    """Runs one epoch for Direct 3-Class probe. Returns (accuracy, macro_f1)."""
    model.train(train)
    total_correct = 0; total_items = 0
    all_pred_indices_flat = []; all_target_indices_flat = []

    desc = f"Epoch {'Train' if train else 'Eval'} (3-Class Direct)"
    batch_iterator = tqdm(loader, desc=desc, leave=False)

    for x, y_kept in batch_iterator: # y_kept is [B, K] with {-1, 0, 1}
        x, y_kept = x.to(args.device), y_kept.to(args.device)

        # Model outputs logits: [B, K * 3]
        logits_flat = model(x)
        # Reshape for CrossEntropyLoss: [B, K*3] -> [B*K, 3]
        # (or [B, 3, K] if using permute later, see PyTorch docs for CE)
        # Let's go with [B*K, 3] as it's more direct for argmax too.
        logits_reshaped = logits_flat.view(-1, 3) # Each row is logits for one (label, timestep) instance

        # Map targets {-1, 0, 1} to indices {0, 1, 2} and flatten
        target_indices_flat = map_labels_to_indices(y_kept).view(-1) # [B*K]

        if train:
            loss = criterion(logits_reshaped, target_indices_flat)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            pred_indices_flat = logits_reshaped.argmax(dim=1) # [B*K], contains {0, 1, 2}
            total_correct += (pred_indices_flat == target_indices_flat).sum().item()
            total_items += target_indices_flat.numel()
            all_pred_indices_flat.append(pred_indices_flat.cpu())
            all_target_indices_flat.append(target_indices_flat.cpu())

    final_acc = total_correct / total_items if total_items > 0 else 0.0
    final_f1 = 0.0
    if total_items > 0:
        y_true_all = torch.cat(all_target_indices_flat).numpy()
        y_pred_all = torch.cat(all_pred_indices_flat).numpy()
        final_f1 = f1_score(y_true_all, y_pred_all, labels=[0, 1, 2], average="macro", zero_division=0)
        if not train: # Print classification report for validation
            try:
                report = classification_report(y_true_all, y_pred_all, labels=[0, 1, 2],
                                              target_names=['NA(-1)', 'False(0)', 'True(1)'], zero_division=0)
                print(f"\nClassification Report (Val Layer):\n{report}")
            except Exception as e: print(f"Could not generate classification report: {e}")


    return final_acc, final_f1

# --------------- training per layer ----------------------------------------
records = []
num_layers_to_train = 33 # e.g., Llama2 has 32 blocks (0-31) + input (32 total) = 33

print(f"\n--- Starting Direct 3-Class Training Loop for {num_layers_to_train} Layers ---")

for L in range(num_layers_to_train): # 0 to 32
    print(f"\n▶ LAYER {L:02d}")
    ds_tr = StepDS(L, train_ids); ds_va = StepDS(L, val_ids)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        print(f"  Skipping layer {L} due to empty datasets.")
        records.append(dict(layer=L, val_acc=np.nan, val_f1=np.nan, status="skipped_empty_data"))
        continue

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    input_dim = ds_tr[0][0].shape[0]
    # Output dimension is K * 3 classes
    probe = nn.Linear(input_dim, num_kept_labels * 3).to(args.device)
    opt = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"  Training Direct 3-Class probe: InputDim={input_dim}, OutputDim={num_kept_labels * 3} ({num_kept_labels} labels x 3 classes)")

    best_val_f1 = -1.0
    for e in range(1, args.epochs + 1):
        acc_tr, f1_tr = run_3class_direct(probe, dl_tr, train=True, opt=opt)
        acc_va, f1_va = run_3class_direct(probe, dl_va, train=False)
        if e == 1 or e % 5 == 0 or e == args.epochs:
             print(f"  Ep{e:02d}| Tr Acc:{acc_tr:.3f} F1:{f1_tr:.3f} | Va Acc:{acc_va:.3f} F1:{f1_va:.3f}")
        if f1_va > best_val_f1: best_val_f1 = f1_va

    final_acc_va, final_f1_va = acc_va, f1_va
    print(f"  Finished Layer {L:02d}. Final Val Acc: {final_acc_va:.4f}, F1: {final_f1_va:.4f} (Best F1: {best_val_f1:.4f})")

    model_save_path = f"{args.output_model_prefix}{L:02d}.pth"
    torch.save({
        "model_type": "Direct3ClassProbe", "state_dict": probe.state_dict(), "layer": L,
        "kept_indices": keep_indices.tolist(), "input_dim": input_dim,
        "num_output_labels_probed": num_kept_labels, "num_classes_per_label": 3,
        "class_weights_used": class_weights.cpu().numpy().tolist(), # Record weights
        "class_mapping": {"NA(-1)": 0, "False(0)": 1, "True(1)": 2}
    }, model_save_path)
    print(f"  ✔ Saved final Direct 3-Class probe model to {model_save_path}")

    records.append(dict(layer=L, val_acc=final_acc_va, val_f1=final_f1_va, status="completed"))

# --------------- CSV summary -----------------------------------------------
results_df = pd.DataFrame(records)
try: results_df.to_csv(args.output_csv, index=False)
except Exception as e: print(f"\nError saving results to CSV: {e}")
print(f"\n✓ Final Direct 3-Class results saved to {args.output_csv}")

print("\n✅ Direct 3-Class Probing Script finished.")