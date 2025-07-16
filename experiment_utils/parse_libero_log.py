#!/usr/bin/env python
"""parse_libero_log.py

Utility to parse LIBERO spatial‑episode logs and report:
  • total episode count
  • successes / failures and success‑rate
  • **global episode indices** that failed or succeeded (unique, 1‑based)

Why a fix?
==========
The raw LIBERO logs reset the phrase *"Starting episode 1"* every time the task changes, so
naïvely using that number leads to duplicate indices like `[1, 3, 2, 3, 4 …]`.  
We now track a *global* episode counter that increments every time we see
**Starting episode** — ensuring each run has a unique index.

Example
-------
$ python parse_libero_log.py rollout_log.txt
{
  "episodes": 60,
  "successes": 49,
  "failures": 11,
  "success_rate": 0.8166666667,
  "failed_indices": [11, 13, 18, 21, 26, 27, 28, 31, 32, 51, 52, 60],
  "successful_indices": [1, 2, 3, 4, 5, 6, ...]
}
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Dict, List, Union

__all__ = [
    "parse_episode_stats",
    "main",
]

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_START_RE = re.compile(r"Starting episode\s+(\d+)", re.IGNORECASE)
_SUCCESS_RE = re.compile(r"Success:\s+(True|False)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Core parsing logic
# ---------------------------------------------------------------------------

def parse_episode_stats(log_text: str) -> Dict[str, Union[int, float, List[int]]]:
    """Return structured statistics from a LIBERO log.

    Parameters
    ----------
    log_text : str
        Raw text produced by the LIBERO task runner.

    Returns
    -------
    dict
        Keys:
        ``episodes``            – total number of episodes detected
        ``successes``           – number of successful episodes
        ``failures``            – number of failed episodes
        ``success_rate``        – successes / episodes (float)
        ``failed_indices``      – list of *global* episode numbers that failed
        ``successful_indices``  – list of *global* episode numbers that succeeded
    """

    failed: List[int] = []
    succeeded: List[int] = []
    current_episode: int | None = None
    global_counter = 0  # unique episode id

    for line in log_text.splitlines():
        # Detect the start of a new episode and assign a unique global id.
        if _START_RE.search(line):
            global_counter += 1
            current_episode = global_counter
            continue

        # Detect the success/failure outcome for the *current* episode.
        if m := _SUCCESS_RE.search(line):
            if current_episode is None:
                # Outcome line found before "Starting episode"; skip.
                continue
            is_success = m.group(1).lower() == "true"
            (succeeded if is_success else failed).append(current_episode)
            current_episode = None  # reset until the next episode start.

    episodes = global_counter
    successes = len(succeeded)
    failures = len(failed)
    success_rate = successes / episodes if episodes else 0.0

    return {
        "episodes": episodes,
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
        "failed_indices": failed,
        "successful_indices": succeeded,
    }


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """CLI wrapper. Prints JSON stats to stdout.

    Usage
    -----
    python parse_libero_log.py rollout_log.txt
    """
    argv = argv or sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        sys.exit(0)

    path = pathlib.Path(argv[0])
    if not path.is_file():
        sys.stderr.write(f"Error: {path} is not a file.\n")
        sys.exit(1)

    stats = parse_episode_stats(path.read_text(errors="replace"))
    json.dump(stats, sys.stdout, indent=2)
    print()  # newline


if __name__ == "__main__":  # pragma: no cover
    main()
