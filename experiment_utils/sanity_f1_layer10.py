# sanity_f1_layer10.py
import torch, glob, ast
from sklearn.metrics import f1_score
from pathlib import Path

ck = torch.load("linear_probe_3class_direct_L10.pth", map_location="cpu")
keep = ck["kept_indices"]
probe = torch.nn.Linear(4096, len(keep)*3)
probe.load_state_dict(ck["state_dict"]); probe.eval()

OBJ = ast.literal_eval(Path("experiments/robot/libero/object_object_relations_keys.txt").read_text())
left_of_idx = [i for i,k in enumerate(OBJ) if k.startswith("left-of")][0]
col = keep.index(left_of_idx)                    # column inside our kept set

y_true, y_pred = [], []
for ep in glob.glob("/root/autodl-tmp/object-episodes/episode_*.pt"):
    d = torch.load(ep, map_location="cpu")
    if 10 not in d["visual_semantic_encoding"]: continue
    x = d["visual_semantic_encoding"][10].float()
    logits = probe(x).view(-1, len(keep), 3)     # [T,K,3]
    pred   = logits.argmax(-1)[:, col]           # [T]
    targ   = torch.zeros_like(pred)
    y = torch.cat([d["symbolic_state_object_relations"],
                   d["symbolic_state_action_subgoals"]], 1)[:, keep][:, col]
    targ[y==-1] = -1; targ[y==0] = 0; targ[y==1] = 1

    mask = targ != -1
    y_true.extend((targ[mask]==1).int().tolist())   # 1=True, 0=False
    y_pred.extend((pred[mask]==2).int().tolist())   # class 2 corresponds to True

print("binary-F1 for left-of @ layer10 =", f1_score(y_true, y_pred))
