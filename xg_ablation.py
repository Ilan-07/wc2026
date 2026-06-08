"""Does xG carry more predictive signal than goals? (the premise behind the xG layer)

The honest, feasible test given we have xG for 4 tournaments (WC2018/2022, Euro2024, Copa2024):
for every team-match where the team has prior matches in that tournament, compare two leading
indicators of this match's goal-difference — the team's prior average GOAL-difference vs its prior
average xG-difference. If xG-diff correlates better with future results, xG is the more repeatable
signal and worth integrating; if not, it isn't. Results cached so re-runs are instant.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wc2026.data.statsbomb import INTERNATIONAL, competition_match_records

CACHE = Path("data/raw/sb_match_records.json")


def load_records() -> dict[str, list[dict]]:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    data = {}
    for key in INTERNATIONAL:
        print(f"fetching per-match xG for {INTERNATIONAL[key][2]} ...")
        data[key] = competition_match_records(key)
    CACHE.write_text(json.dumps(data))
    return data


def main() -> None:
    data = load_records()
    prior_gd, prior_xgd, future_gd = [], [], []
    for recs in data.values():
        hist: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in recs:
            sides = [
                (r["home"], r["home_goals"] - r["away_goals"], r["home_xg"] - r["away_xg"]),
                (r["away"], r["away_goals"] - r["home_goals"], r["away_xg"] - r["home_xg"]),
            ]
            for team, gd, xgd in sides:
                past = hist[team]
                if past:  # need at least one prior match to form an indicator
                    prior_gd.append(float(np.mean([g for g, _ in past])))
                    prior_xgd.append(float(np.mean([x for _, x in past])))
                    future_gd.append(gd)
                past.append((gd, xgd))

    g, x, y = np.array(prior_gd), np.array(prior_xgd), np.array(future_gd)
    n = len(y)
    c_goals = float(np.corrcoef(g, y)[0, 1])
    c_xg = float(np.corrcoef(x, y)[0, 1])
    print(f"\nObservations: {n} team-matches (4 tournaments)")
    print("Predicting this match's goal-difference from a team's prior form:")
    print(f"  corr(prior GOAL-diff , result) = {c_goals:+.3f}")
    print(f"  corr(prior xG-diff   , result) = {c_xg:+.3f}")
    print(f"\nxG is the {'BETTER' if c_xg > c_goals else 'WORSE/EQUAL'} leading indicator "
          f"(Δ = {c_xg - c_goals:+.3f}).")
    print("Verdict:", "xG carries extra signal -> worth integrating." if c_xg > c_goals + 0.02
          else "xG does NOT clearly beat goals on this data -> do not ship.")


if __name__ == "__main__":
    main()
