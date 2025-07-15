#!/usr/bin/env python
"""
make_probe_heatmap_binary_F1.py

Create a layer × predicate-family heat-map from the MASKING probes
(trained with linear_probe_LXX.pth).
Plots binary F1-score (for class 1) averaged per category.
Assumes category is the first word in the label string.

Usage:
    python make_probe_heatmap_binary_F1.py --metric unweighted_f1 --outfile heatmap_masking_unweighted_f1.png
    python make_probe_heatmap_binary_F1.py --metric support_weighted_f1 --outfile heatmap_masking_support_weighted_f1.png
"""

# --------------------------------------------------------------------------- #
import argparse, ast, glob, os, re, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# --------------------------------------------------------------------------- #
cli = argparse.ArgumentParser(description="Generate Binary F1 heatmap from masking probe checkpoints.")
cli.add_argument("--log_dir", default="experiments/logs", help="Directory containing episode_*.pt files")
cli.add_argument("--model_prefix", default="linear_probe_L", help="Prefix for saved model files (e.g., linear_probe_L)")
cli.add_argument("--metric", required=True, choices=['unweighted_f1', 'support_weighted_f1'],
                 help="Which F1 averaging to plot ('unweighted_f1' or 'support_weighted_f1')")
cli.add_argument("--success_only", action="store_true", help="Use only successful episodes for evaluation")
cli.add_argument("--outfile", default="probe_heatmap_binary_f1.png", help="Output filename for the heatmap image")
cli.add_argument("--label_dir", default="experiments/robot/libero", help="Directory for label key files")
args = cli.parse_args()

# --------------------------------------------------------------------------- #
# Load Label Keys and Define Categories
OBJ_KEYS_PATH = Path(args.label_dir) / "object_object_relations_keys.txt"
ACT_KEYS_PATH = Path(args.label_dir) / "object_action_states_keys.txt"

print(f"Loading OBJ_KEYS from: {OBJ_KEYS_PATH}")
print(f"Loading ACT_KEYS from: {ACT_KEYS_PATH}")

if not OBJ_KEYS_PATH.exists():
     raise FileNotFoundError(f"Object relation keys file not found at: {OBJ_KEYS_PATH}")
if not ACT_KEYS_PATH.exists():
     raise FileNotFoundError(f"Action state keys file not found at: {ACT_KEYS_PATH}")

OBJ_KEYS = ast.literal_eval(OBJ_KEYS_PATH.read_text())
ACT_KEYS = ast.literal_eval(ACT_KEYS_PATH.read_text())
ALL = OBJ_KEYS + ACT_KEYS

# !!! USING THE ORIGINAL SIMPLER cat FUNCTION !!!
def cat(lbl: str) -> str:
    """Extracts the category part of a label string (assumed to be the first word)."""
    if not isinstance(lbl, str):
        print(f"Warning: Non-string label encountered: {lbl}. Returning as is.")
        return str(lbl)
    try:
        return lbl.split()[0] # Takes the first word after splitting by whitespace
    except IndexError: # Handles empty strings or strings with no whitespace
        print(f"Warning: Label string '{lbl}' could not be split. Returning as is.")
        return lbl


CAT_NAMES = sorted({cat(k) for k in ALL})
print("-" * 20)
print("DEBUG: First 10 labels from ALL list:", ALL[:10])
print("DEBUG: Categories found (CAT_NAMES):", CAT_NAMES)
print(f"DEBUG: Number of categories found: {len(CAT_NAMES)}")
print(f"DEBUG: Number of total labels from key files: {len(ALL)}")
print("-" * 20)
if len(CAT_NAMES) > 50 and len(CAT_NAMES) > len(ALL) * 0.8 : # Heuristic: if too many categories close to total labels
    print("WARNING: Number of categories is very high, close to the total number of labels.")
    print("This might indicate the `cat` function is not correctly parsing your label strings into broader families.")
    print("Current cat function expects category to be the first word: `category detail detail`")
    print("Please check the format of labels in your .txt key files and the cat() function.")


# --------------------------------------------------------------------------- #
# Load Success Map (Optional)
succ_map = {}
succ_csv = Path(args.log_dir) / "libero_object_summary_per_episode.csv"
if succ_csv.exists():
    try:
        import pandas as pd
        df = pd.read_csv(succ_csv); succ_map = {int(r.EpisodeAbs): bool(r.Success) for _, r in df.iterrows()}
        print(f"Loaded success map for {len(succ_map)} episodes.")
    except ImportError: print("Warning: pandas not found, cannot load success map.")
    except Exception as e: print(f"Warning: Failed to load success map: {e}")
# --------------------------------------------------------------------------- #
# Load Episode Cache
eps = {}
pt_pattern = os.path.join(args.log_dir,"episode_*.pt")
eps_files = sorted(glob.glob(pt_pattern))
if not eps_files: raise FileNotFoundError(f"No episode files found at {pt_pattern}")
print(f"Found {len(eps_files)} episode files.")
for fp in tqdm(eps_files, desc="Caching episodes", unit="ep"):
    match = re.search(r"episode_(\d+)\.pt", os.path.basename(fp)); idx = int(match.group(1)) if match else -1
    if idx == -1: print(f"Warning: Could not parse episode index from {fp}"); continue
    if args.success_only and succ_map and not succ_map.get(idx, True): continue
    try:
        eps[idx] = torch.load(fp, map_location="cpu") # Load to CPU
        if not isinstance(eps[idx], dict) or "visual_semantic_encoding" not in eps[idx]:
             print(f"\nWarning: Invalid data structure or missing embeddings in ep {idx} from {fp}. Skipping.")
             del eps[idx]; continue
    except Exception as e: print(f"\nWarning: Failed to load/validate episode {idx} from {fp}: {e}"); continue
if not eps: raise ValueError("No valid episodes loaded after filtering.")
print(f"Cached {len(eps)} episodes { '(success only)' if args.success_only else '' }.")
# --------------------------------------------------------------------------- #

# --- Main Loop ---
L_MAX  = 33
heat = np.full((L_MAX, len(CAT_NAMES)), np.nan, dtype=float)
first_ep_key = next(iter(eps)); input_dim_ref = None
if eps[first_ep_key].get("visual_semantic_encoding"):
     for l_idx in range(L_MAX):
         if l_idx in eps[first_ep_key]["visual_semantic_encoding"]:
             if eps[first_ep_key]["visual_semantic_encoding"][l_idx].ndim >=2:
                input_dim_ref = eps[first_ep_key]["visual_semantic_encoding"][l_idx].shape[-1]
                print(f"Reference input_dim from ep {first_ep_key}, layer {l_idx}: {input_dim_ref}")
                break
             else: print(f"Warning: Embedding for ep {first_ep_key}, layer {l_idx} is not a 2D+ tensor.")
if input_dim_ref is None: print("CRITICAL WARNING: Could not determine reference input_dim. Attempting to infer from checkpoints.")

print(f"\nGenerating heatmap for metric: {args.metric}")

for L in range(L_MAX):
    print(f"Processing Layer {L:02d}...", end='\r')
    ck_fp = Path(f"{args.model_prefix}{L:02d}.pth")
    if not ck_fp.exists(): continue

    try:
        ck = torch.load(ck_fp, map_location="cpu")
        keep_indices = ck.get("kept");
        if keep_indices is None: keep_indices = ck.get("kept_indices")
        if "state_dict" not in ck or not keep_indices: print(f"\nWarning: Ckpt {ck_fp} missing state_dict/kept_indices. Skipping."); continue

        num_kept_labels_ckpt = len(keep_indices); input_dim_ckpt = None
        for name, param in ck["state_dict"].items():
            if name == "linear.weight" or name == "weight":
                 if param.shape[0] == num_kept_labels_ckpt: input_dim_ckpt = param.shape[1]; break
        
        current_input_dim = input_dim_ckpt if input_dim_ckpt is not None else input_dim_ref
        if current_input_dim is None : print(f"\nCRITICAL: Cannot determine input_dim for probe {L}. Skipping."); continue
        if input_dim_ref is not None and input_dim_ckpt is not None and current_input_dim != input_dim_ref:
             print(f"\nWarning: Input dim from ckpt {L} ({current_input_dim}) differs from ref ({input_dim_ref}). Using ckpt dim.")
        
        probe = torch.nn.Linear(current_input_dim, num_kept_labels_ckpt)
        probe.load_state_dict(ck["state_dict"]); probe.eval()
        num_kept_labels = num_kept_labels_ckpt
    except Exception as e: print(f"\nError loading ckpt {ck_fp}: {e}. Skipping."); continue

    per_cat_metrics = defaultdict(list)
    num_valid_ep_steps_layer = 0

    for ep_idx, d in eps.items():
        if L not in d.get("visual_semantic_encoding", {}) or not d["visual_semantic_encoding"][L].numel(): continue
        try:
            x_ep = d["visual_semantic_encoding"][L].float()
            y_full_ep = torch.cat([d["symbolic_state_object_relations"], d["symbolic_state_action_subgoals"]], 1)
            y_ep = y_full_ep[:, keep_indices]
            if x_ep.shape[0] != y_ep.shape[0]: print(f"\nWarning: Time dim mismatch ep {ep_idx}, L {L}. Skipping."); continue

            with torch.no_grad(): logits_ep = probe(x_ep); pred_bin_ep = (logits_ep.sigmoid() > 0.5).long()
            mask_01_ep = (y_ep != -1); target_bin_ep = (y_ep == 1).long()

            for col_idx, global_label_idx in enumerate(keep_indices):
                category_name = cat(ALL[global_label_idx])
                valid_mask_for_label_ep = mask_01_ep[:, col_idx]
                if not valid_mask_for_label_ep.any(): continue

                y_true_col = target_bin_ep[:, col_idx][valid_mask_for_label_ep].numpy()
                y_pred_col = pred_bin_ep[:, col_idx][valid_mask_for_label_ep].numpy()
                f1 = f1_score(y_true_col, y_pred_col, labels=[0, 1], pos_label=1, average='binary', zero_division=0)
                
                if args.metric == 'support_weighted_f1':
                    per_cat_metrics[category_name].append((f1, len(y_true_col)))
                else: per_cat_metrics[category_name].append(f1)
            num_valid_ep_steps_layer += y_ep.shape[0]
        except Exception as e: print(f"\nError processing ep {ep_idx} for L {L}: {e}. Skipping."); continue
            
    if num_valid_ep_steps_layer > 0:
        for cat_idx, category_name in enumerate(CAT_NAMES):
            metrics_for_cat = per_cat_metrics[category_name]
            if metrics_for_cat:
                if args.metric == 'support_weighted_f1':
                    f1_scores, supports = zip(*metrics_for_cat)
                    if sum(supports) > 0: heat[L, cat_idx] = np.average(f1_scores, weights=supports)
                    else: heat[L, cat_idx] = np.mean(f1_scores) if f1_scores else np.nan
                else: heat[L, cat_idx] = float(np.mean(metrics_for_cat))
    else: print(f"\nWarning: No valid steps for L {L}. NaNs in heatmap.")

print("\nFinished processing all layers.")
# --------------------------------------------------------------------------- #
# Plotting code remains largely the same, ensure title reflects binary F1 for masking probes
valid_layers = np.where(~np.isnan(heat).all(axis=1))[0]
if len(valid_layers) == 0: print("CRITICAL Error: No valid data to plot. Exiting."); exit()
heat_filtered = heat[valid_layers, :]; yticklabels_filtered = valid_layers
plt.figure(figsize=(max(8, 0.5 * len(CAT_NAMES)), max(6, 0.3 * len(valid_layers))))
sns.heatmap(heat_filtered, vmin=0, vmax=1, cmap="YlGnBu", xticklabels=CAT_NAMES,
            yticklabels=yticklabels_filtered, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.xticks(rotation=35, ha="right", fontsize=8); plt.yticks(rotation=0, fontsize=8)
plt.xlabel("Predicate category", fontsize=10); plt.ylabel("Llama layer", fontsize=10)
metric_title = "Support-Weighted Binary F1 (True vs False)" if args.metric == 'support_weighted_f1' else "Unweighted Binary F1 (True vs False)"
title = f"Validation {metric_title} per layer × category (Masking -1 Probes)"
title += " (Success Only)" if args.success_only else ""; plt.title(title, fontsize=12)
plt.tight_layout(pad=1.0);
try: plt.savefig(args.outfile, dpi=300, bbox_inches='tight'); print(f"\nHeatmap saved to {args.outfile}")
except Exception as e: print(f"\nError saving heatmap: {e}")
print("\n✅ Heatmap script (Binary F1 for Masking Probes) finished.")