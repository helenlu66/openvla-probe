# util_data.py ---------------------------------------------------------------
import ast, glob, os, re, torch
from pathlib import Path

# -------- label keys --------------------------------------------------------
OBJ = ast.literal_eval(Path(
        "experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT = ast.literal_eval(Path(
        "experiments/robot/libero/object_action_states_keys.txt").read_text())
LABELS = OBJ + ACT
def family(lbl): return lbl.split()[0]

# -------- load all .pt roll-outs once --------------------------------------
def cache_episodes(log_dir, success_only=False):
    succ = {}
    csv = Path(log_dir)/"libero_object_summary_per_episode.csv"
    if csv.exists():
        import pandas as pd
        succ = pd.read_csv(csv).set_index("EpisodeAbs")["Success"].to_dict()

    cache = {}
    for fp in glob.glob(os.path.join(log_dir,"episode_*.pt")):
        idx = int(re.search(r"episode_(\d+)\.pt", fp).group(1))
        if success_only and succ and not succ.get(idx, True):
            continue
        cache[idx] = torch.load(fp, map_location="cpu")
    return cache
