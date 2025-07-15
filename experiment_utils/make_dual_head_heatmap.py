#!/usr/bin/env python
"""
Create a layer × predicate-family heat-map from DUAL-HEAD probes.
Calculates accuracy for each individual label (based on chosen metric)
and then averages these accuracies per category for the final plot.

Usage:
    python make_dual_head_heatmap.py --model_prefix linear_probe_dual_head_final_L --metric truth_acc --outfile truth_acc_heatmap.png
    python make_dual_head_heatmap.py --model_prefix linear_probe_dual_head_final_L --metric pres_acc --outfile pres_acc_heatmap.png
"""

# --------------------------------------------------------------------------- #
import argparse, ast, glob, os, re, numpy as np, torch, torch.nn as nn # Need nn
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns
import warnings

# Suppress annoying UserWarnings from plt/seaborn with many ticks
warnings.filterwarnings("ignore", category=UserWarning)
# --------------------------------------------------------------------------- #

# --------------- Dual-Head Probe Definition (Must match training script) ---
class DualHeadProbe(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.presence_head = nn.Linear(input_dim, num_labels)
        self.truth_head = nn.Linear(input_dim, num_labels)
    def forward(self, x):
        return self.presence_head(x), self.truth_head(x)
# --------------------------------------------------------------------------- #

cli = argparse.ArgumentParser(description="Generate heatmap from Dual-Head probe checkpoints by averaging per-label accuracies within categories.")
# (CLI arguments remain the same)
cli.add_argument("--log_dir", default="experiments/logs", help="Directory containing episode_*.pt files")
cli.add_argument("--model_prefix", default="linear_probe_dual_head_final_L", help="Prefix for saved model files (e.g., linear_probe_dual_head_final_L)")
cli.add_argument("--metric", required=True, choices=['truth_acc', 'pres_acc'], help="Which metric's accuracy to plot ('truth_acc' or 'pres_acc')")
cli.add_argument("--success_only", action="store_true", help="Use only successful episodes for evaluation")
cli.add_argument("--outfile", default="probe_heatmap_dual_head.png", help="Output filename for the heatmap image")
cli.add_argument("--label_dir", default="experiments/robot/libero", help="Directory for label key files")
args = cli.parse_args()

# --------------------------------------------------------------------------- #
# Load Label Keys and Define Categories
# (Code remains the same)
OBJ_KEYS = ast.literal_eval(Path(
        "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT_KEYS = ast.literal_eval(Path(
        "experiments/robot/libero/object_action_states_keys.txt").read_text())
ALL = OBJ_KEYS + ACT_KEYS
def cat(lbl: str) -> str:
    """Return predicate family, e.g. 'left-of', 'open', …"""
    return lbl.split()[0]

CAT_NAMES = sorted({cat(k) for k in ALL})
print(f"Found {len(CAT_NAMES)} predicate categories.")

# --------------------------------------------------------------------------- #
# Load Success Map (Optional)
# (Code remains the same)
succ_map = {}
succ_csv = Path(args.log_dir) / "libero_object_summary_per_episode.csv"
if succ_csv.exists():
    try:
        import pandas as pd
        df = pd.read_csv(succ_csv)
        succ_map = {int(r.EpisodeAbs): bool(r.Success) for _, r in df.iterrows()}
        print(f"Loaded success map for {len(succ_map)} episodes.")
    except ImportError: print("Warning: pandas not found, cannot load success map.")
    except Exception as e: print(f"Warning: Failed to load success map: {e}")

# --------------------------------------------------------------------------- #
# Load Episode Cache
# (Code remains the same)
eps = {}
pt_pattern = os.path.join(args.log_dir,"episode_*.pt")
eps_files = sorted(glob.glob(pt_pattern))
if not eps_files: raise FileNotFoundError(f"No episode files found at {pt_pattern}")
print(f"Found {len(eps_files)} episode files.")
for fp in tqdm(eps_files, desc="Caching episodes", unit="ep"):
    match = re.search(r"episode_(\d+)\.pt", os.path.basename(fp)); idx = int(match.group(1)) if match else -1
    if idx == -1: continue
    if args.success_only and succ_map and not succ_map.get(idx, True): continue
    try:
        eps[idx] = torch.load(fp, map_location="cpu")
        if not isinstance(eps[idx], dict) or "visual_semantic_encoding" not in eps[idx]: del eps[idx]; continue
    except Exception as e: print(f"\nWarning: Failed to load/validate episode {idx} from {fp}: {e}"); continue
if not eps: raise ValueError("No valid episodes loaded after filtering.")
print(f"Cached {len(eps)} episodes { '(success only)' if args.success_only else '' }.")
# --------------------------------------------------------------------------- #

# --- Main Loop ---
L_MAX  = 33
heat = np.full((L_MAX, len(CAT_NAMES)), np.nan, dtype=float)
# (Get representative input dimension remains the same)
first_ep_key = next(iter(eps)); input_dim_ref = None
if eps[first_ep_key].get("visual_semantic_encoding"):
     # Find first layer actually present in the first episode to get dim
     for l_idx in range(L_MAX):
         if l_idx in eps[first_ep_key]["visual_semantic_encoding"]:
             input_dim_ref = eps[first_ep_key]["visual_semantic_encoding"][l_idx].shape[-1]
             break
if input_dim_ref is None:
     raise ValueError(f"Could not determine input embedding dimension from first episode {first_ep_key}.")


print(f"\nGenerating heatmap for metric: {args.metric}")

for L in range(L_MAX):
    print(f"Processing Layer {L:02d}...", end='\r')
    ck_fp = Path(f"{args.model_prefix}{L:02d}.pth")
    if not ck_fp.exists(): continue

    try:
        ck = torch.load(ck_fp, map_location="cpu")
        if ck.get("model_type") != "DualHeadProbe": continue
        keep_indices = ck.get("kept_indices"); input_dim = ck.get("input_dim"); num_kept_labels = ck.get("num_output_labels")
        if keep_indices is None or input_dim is None or num_kept_labels is None: continue
        if len(keep_indices) != num_kept_labels: continue
        if input_dim != input_dim_ref: # Check consistency
             print(f"\nWarning: Input dim mismatch in ckpt {L} ({input_dim}) vs reference ({input_dim_ref}). Skipping.")
             continue

        probe = DualHeadProbe(input_dim, num_kept_labels)
        probe.load_state_dict(ck["state_dict"]); probe.eval()
    except Exception as e:
        print(f"\nError loading or validating checkpoint {ck_fp}: {e}. Skipping layer.")
        continue

    # Stores list of accuracies for each label within the category
    per_cat_accuracies = defaultdict(list)
    num_valid_ep_steps = 0

    for ep_idx, d in eps.items():
        if L not in d.get("visual_semantic_encoding", {}): continue
        try:
            x = d["visual_semantic_encoding"][L].float(); y_full = torch.cat([d["symbolic_state_object_relations"], d["symbolic_state_action_subgoals"]], 1)
            y = y_full[:, keep_indices]
            if x.shape[0] != y.shape[0]: continue

            with torch.no_grad():
                presence_logits, truth_logits = probe(x)
                presence_pred = (presence_logits.sigmoid() > 0.5).long()
                truth_pred = (truth_logits.sigmoid() > 0.5).long()

            # --- Select Target, Pred, Mask based on chosen METRIC ---
            if args.metric == 'pres_acc':
                target = (y != -1).long(); pred = presence_pred; mask = torch.ones_like(target, dtype=torch.bool)
            elif args.metric == 'truth_acc':
                target = (y == 1).long(); pred = truth_pred; mask = (y != -1)
            else: raise ValueError(f"Unknown metric: {args.metric}")

            correct = (pred == target) & mask

            # --- Accumulate per-label accuracies ---
            for col_idx, global_label_idx in enumerate(keep_indices):
                category_name = cat(ALL[global_label_idx])
                valid_mask_col = mask[:, col_idx]
                if valid_mask_col.any():
                    # Calculate accuracy for this label (col_idx) over valid steps in this episode
                    acc = correct[:, col_idx][valid_mask_col].float().mean().item()
                    per_cat_accuracies[category_name].append(acc) # Add to list for averaging later
            num_valid_ep_steps += y.shape[0]

        except Exception as e:
            print(f"\nError processing episode {ep_idx} for layer {L}: {e}. Skipping.")
            continue

    # --- Average per-label accuracies within each category for Layer L ---
    if num_valid_ep_steps == 0: print(f"\nWarning: No valid steps processed for layer {L}. Setting NaNs.")
    for cat_idx, category_name in enumerate(CAT_NAMES):
        if per_cat_accuracies[category_name]:
            # THE AGGREGATION STEP: Average the collected per-label accuracies
            heat[L, cat_idx] = float(np.mean(per_cat_accuracies[category_name]))

print("\nFinished processing all layers.")
# --------------------------------------------------------------------------- #

# --- Plotting ---
# (Plotting code remains the same)
valid_layers = np.where(~np.isnan(heat).all(axis=1))[0]
if len(valid_layers) == 0: print("Error: No valid data found to plot. Exiting."); exit()
heat_filtered = heat[valid_layers, :]; yticklabels_filtered = valid_layers
plt.figure(figsize=(12, max(6, 0.3 * len(valid_layers)))); sns.heatmap(heat_filtered, vmin=0, vmax=1, cmap="YlGnBu", xticklabels=CAT_NAMES, yticklabels=yticklabels_filtered, annot=True, fmt=".2f", annot_kws={"size": 8})
plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0); plt.xlabel("Predicate category"); plt.ylabel("Llama layer")
title = f"Validation {args.metric.replace('_',' ').title()} per layer × category"; title += " (Success Only)" if args.success_only else ""; plt.title(title)
plt.tight_layout(pad=0.5)
try: plt.savefig(args.outfile, dpi=300); print(f"\nHeatmap saved to {args.outfile}")
except Exception as e: print(f"\nError saving heatmap: {e}")
print("\n✅ Heatmap script finished.")