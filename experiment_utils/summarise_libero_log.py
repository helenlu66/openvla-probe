#!/usr/bin/env python3
"""
summarise_libero_log.py
-----------------------

Parse LIBERO evaluation logs and summarise per-task and per-episode success.

Example
-------
python summarise_libero_log.py --log EVAL-*.txt --csv summary.csv
"""

import argparse, re, collections, pathlib, csv
import pandas as pd

# ---------- CLI --------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--log",  nargs="+", required=True,
               help="one or more EVAL-*.txt files")
p.add_argument("--csv",  default=None,
               help="optional path to save the task-level table; "
                    "a *_per_episode.csv file is also written")
p.add_argument("--sort", action="store_true",
               help="sort tasks alphabetically (default = keep first-seen order)")
args = p.parse_args()

# ---------- regex helpers ----------------------------------------------------
re_task   = re.compile(r"^Task:\s+(.*)$")
re_result = re.compile(r"^Success:\s+(True|False)$")

# ---------- data structures --------------------------------------------------
TaskStats   = collections.namedtuple("TaskStats", "episodes successes")
task_order  = []            # preserve first-seen order
stats       = collections.defaultdict(lambda: TaskStats(0, 0))
episode_rows= []            # for per-episode table
abs_index   = 0

# ---------- parse ------------------------------------------------------------
for log_path in args.log:
    current_task = None
    for line in pathlib.Path(log_path).read_text().splitlines():
        m_task = re_task.match(line)
        if m_task:
            current_task = m_task.group(1).strip()
            if current_task not in stats:
                task_order.append(current_task)
            continue

        m_res = re_result.match(line)
        if m_res and current_task is not None:
            success_bool = m_res.group(1) == "True"

            # update task-level counters
            epi, suc = stats[current_task]
            stats[current_task] = TaskStats(episodes=epi+1,
                                            successes=suc + success_bool)

            # store per-episode detail
            episode_rows.append({
                "Task":           current_task,
                "EpisodeAbs":     abs_index+1,
                "EpisodeWithin":  epi+1,            # 1-based per task
                "Success":        success_bool,
            })
            abs_index += 1

# ---------- build tables -----------------------------------------------------
if args.sort:
    task_order = sorted(task_order)

task_rows = []
for t in task_order:
    epi, suc = stats[t]
    task_rows.append({
        "Task":        t,
        "#Episodes":   epi,
        "#Successes":  suc,
        "SuccessRate": f"{(suc/epi*100):.2f}%",
    })

df_tasks   = pd.DataFrame(task_rows)
df_epi     = pd.DataFrame(episode_rows)

# ---------- display ----------------------------------------------------------
print("\nTask-level summary\n" + "-"*80)
print(df_tasks.to_string(index=False))

print("\nPer-episode detail (first 20 rows)\n" + "-"*80)
print(df_epi.head(20).to_string(index=False))

# ---------- optional CSV export ---------------------------------------------
if args.csv:
    csv_path = pathlib.Path(args.csv)
    df_tasks.to_csv(csv_path, index=False)
    df_epi.to_csv(csv_path.with_name(csv_path.stem+"_per_episode.csv"),
                  index=False)
    print(f"\n✓ CSVs written to {csv_path} and {csv_path.stem}_per_episode.csv")
