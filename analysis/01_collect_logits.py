#!/usr/bin/env python
"""
Forward all validation episodes through each trained probe and pickle:
    logits_L.pkl   — dict{ep_id: tensor [T,K_kept]}
    targets_L.pkl  — same shape, values ∈ {-1,0,1}
"""
import argparse, torch, pickle, tqdm, os
from util_data import cache_episodes, LABELS

cli = argparse.ArgumentParser()
cli.add_argument("--log_dir", default="experiments/logs")
cli.add_argument("--device",  default="cpu")
cli.add_argument("--success_only", action="store_true")
args = cli.parse_args()

eps = cache_episodes(args.log_dir, args.success_only)
print(f"cached {len(eps)} episodes")

for L in range(33):
    ck = torch.load(f"linear_probe_L{L}.pth", map_location=args.device)
    keep = ck["kept"]; K=len(keep)
    if K==0: continue
    probe = torch.nn.Linear(
        eps[next(iter(eps))]["visual_semantic_encoding"][L].shape[-1], K
    ).to(args.device)
    probe.load_state_dict(ck["state_dict"]); probe.eval()

    logits, targets = {}, {}
    for eid,d in eps.items():
        if L not in d["visual_semantic_encoding"]: continue
        x = d["visual_semantic_encoding"][L].to(args.device).float()
        y = torch.cat([d["symbolic_state_object_relations"],
                       d["symbolic_state_action_subgoals"]],1)[:,keep]
        with torch.no_grad(): log = probe(x).sigmoid().cpu()
        logits[eid]  = log
        targets[eid] = y.cpu()

    with open(f"analysis/logits_{L:02d}.pkl","wb") as f:  pickle.dump(logits,f)
    with open(f"analysis/targets_{L:02d}.pkl","wb") as f: pickle.dump(targets,f)
    print(f"L{L:02d}  saved ({len(logits)} eps, {K} labels)")
