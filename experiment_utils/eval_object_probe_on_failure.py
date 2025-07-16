#!/usr/bin/env python
"""
Evaluate a saved linear probe on a set of episodes (e.g. failures only).
Usage:
    python eval_object_probe.py \
        --probe linear_probe_L32.pth \
        --eps 2,7,9,10,12,20,22,26,45 \
        --log_dir /root/autodl-tmp/object-episodes
"""

import ast, glob, torch, argparse, random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm
import numpy as np

# ---------------------------------------------------------------- CLI
cli = argparse.ArgumentParser()
cli.add_argument("--probe", required=True)
cli.add_argument("--eps",   required=True,
                 help="comma-sep ids or ranges, e.g. 2,7,9 or 40-50")
cli.add_argument("--log_dir", required=True)
cli.add_argument("--batch", type=int, default=4096)
cli.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = cli.parse_args()

def parse_list(spec):
    out=set()
    for tok in spec.split(","):
        tok=tok.strip()
        if "-" in tok:
            a,b=map(int,tok.split("-")); out.update(range(a,b+1))
        else: out.add(int(tok))
    return out
EVAL_IDS=parse_list(args.eps)

# ---------------------------------------------------------------- labels
OBJ_KEYS=ast.literal_eval(Path("experiments/robot/libero/object_object_relations_keys.txt").read_text())
ACT_KEYS=ast.literal_eval(Path("experiments/robot/libero/object_action_states_keys.txt").read_text())
NUM_LABELS=len(OBJ_KEYS)+len(ACT_KEYS)

# ---------------------------------------------------------------- load probe
ckpt=torch.load(args.probe,map_location="cpu")
probe=torch.nn.Linear(ckpt["state_dict"]["weight"].shape[1],len(ckpt["kept"]))
probe.load_state_dict(ckpt["state_dict"]); probe.to(args.device); probe.eval()
KEEP=torch.as_tensor(ckpt["kept"])

# ---------------------------------------------------------------- cache episodes
cache={}
pattern = str(Path(args.log_dir) / "episode_*.pt")
for fp in glob.glob(pattern):
    idx=int(Path(fp).stem.split("_")[1])
    if idx in EVAL_IDS:
        cache[idx]=torch.load(fp,map_location="cpu")
if not cache: raise ValueError("No matching episodes found!")

# ---------------------------------------------------------------- dataset
class StepDS(Dataset):
    def __init__(self, layer):
        self.samples=[(i,t) for i in cache
                      if layer in cache[i]["visual_semantic_encoding"]
                      for t in range(cache[i]["visual_semantic_encoding"][layer].shape[0])]
        self.layer=layer
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        i,t=self.samples[idx]; d=cache[i]
        x=d["visual_semantic_encoding"][self.layer][t].float()
        y=torch.cat([d["symbolic_state_object_relations"][t],
                     d["symbolic_state_action_subgoals"][t]])[KEEP]
        return x,y

ds=StepDS(layer=ckpt["layer"]); dl=DataLoader(ds,args.batch,False)
print(f"Frames in evaluation set: {len(ds)}")

# ---------------------------------------------------------------- metric
bce=torch.nn.BCEWithLogitsLoss(reduction="none")  # not used, but handy if needed
ok=tot=0; probs_all=[]; y_all=[]; pred_all=[]
for x,y in tqdm(dl):
    x,y=x.to(args.device),y.to(args.device)
    with torch.no_grad():
        p=probe(x).sigmoid()
    m=(y!=-1); tgt=(y==1).float()
    pred=(p>0.5).long()
    ok+=(pred[m]==tgt[m]).sum().item(); tot+=m.sum().item()
    probs_all.append(p[m].cpu()); pred_all.append(pred[m].cpu()); y_all.append(tgt[m].cpu())

acc= ok/tot if tot else 0.0
y_true=np.concatenate([t.numpy() for t in y_all])
y_pred=np.concatenate([t.numpy() for t in pred_all])
y_prob=np.concatenate([t.numpy() for t in probs_all])
f1=f1_score(y_true,y_pred,average="macro",zero_division=0)
ap=average_precision_score(y_true,y_prob,average="macro")
print(f"\nEval on failures  acc={acc:.3f}  F1={f1:.3f}  AP={ap:.3f}")
