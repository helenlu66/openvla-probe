import torch, glob
Y = []
for p in glob.glob("experiments/logs/episode_*.pt"):
    d = torch.load(p, map_location="cpu")
    Y.append(torch.cat([d["symbolic_state_object_relations"],
                        d["symbolic_state_action_subgoals"]], 1))
Y = torch.cat(Y)                      # [N, 481]
print( (Y == 1).sum(0), (Y == 0).sum(0) )   # counts of 1-s and 0-s per column