"""
python count_atoms.py
"""

import ast, collections, pprint, pathlib

OBJ_FILE = pathlib.Path("experiments/robot/libero/object_object_relations_keys.txt")
ACT_FILE = pathlib.Path("experiments/robot/libero/object_action_states_keys.txt")

obj_keys = ast.literal_eval(OBJ_FILE.read_text().strip())
act_keys = ast.literal_eval(ACT_FILE.read_text().strip())

# --- group by predicate name (= first token) -------------------------------
def by_pred(keys):
    d = collections.Counter()
    for k in keys:
        pred = k.split()[0]            # "behind basket_1 tomato_sauce_1" -> "behind"
        d[pred] += 1
    return d

print("\nObject–relation atoms:")
pprint.pp(by_pred(obj_keys))

print("\nAction-state atoms:")
pprint.pp(by_pred(act_keys))

print("\nTotals:")
print(f"  object atoms = {len(obj_keys)}")
print(f"  action atoms = {len(act_keys)}")
print(f"  grand total  = {len(obj_keys)+len(act_keys)}")
