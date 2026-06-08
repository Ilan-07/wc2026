"""Does blending an xG rating into the goals rating improve match RPS? (the gate)

Leave-one-tournament-out, fully out-of-sample:
  for each of the 4 StatsBomb tournaments T:
    * goals model  — Dixon-Coles fit on all real results BEFORE T (what the live forecast uses)
    * xG rating    — XGRating fit on the OTHER 3 tournaments' xG (never T → no leakage)
    * for every T match where BOTH teams have an xG rating, predict W/D/L with the goals rates and
      with goals-rates blended toward xG-rates at weight w, and score both by RPS.
Evaluation is restricted to xG-covered matches (where the blend actually changes anything), so the
comparison isn't diluted by matches the blend leaves untouched. A weight earns its place only if the
blended RPS beats the goals-only baseline on this held-out set.

Uses the cached data/raw/sb_match_records.json (no re-fetch).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.ratings.dixon_coles import DixonColesModel, _poisson_pmf, tau
from wc2026.ratings.elo import EloModel
from wc2026.ratings.xg_rating import XGRating, blend_rates

TOURN = {"wc2018": date(2018, 6, 14), "wc2022": date(2022, 11, 20),
         "euro2024": date(2024, 6, 14), "copa2024": date(2024, 6, 20)}
CACHE = Path("data/raw/sb_match_records.json")
WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]


def wdl(g, a):
    return 0 if g > a else (1 if g == a else 2)


def grid_wdl(lam, mu, rho):
    ks = np.arange(11)
    g = np.outer(_poisson_pmf(ks, lam), _poisson_pmf(ks, mu)) * tau(ks[:, None], ks[None, :], lam, mu, rho)
    g = np.clip(g, 0, None)
    g /= g.sum()
    return [float(np.tril(g, -1).sum()), float(np.trace(g)), float(np.triu(g, 1).sum())]


def main():
    records = json.loads(CACHE.read_text())
    allm = loaders.load_results(since="2006-01-01", min_team_matches=15)

    base_rps, n_eval = 0.0, 0
    blend_rps = {w: 0.0 for w in WEIGHTS}

    for test in TOURN:
        start = TOURN[test]
        train = [m for m in allm if m["date"] < start]
        elo = EloModel().fit(train)
        mm = float(np.mean(list(elo.ratings.values())))
        dc = DixonColesModel(half_life_days=1100.0)
        params = dc.fit(train, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})

        # xG rating from the OTHER three tournaments only
        xg_train = [r for k in TOURN if k != test for r in records[k]]
        xparams = XGRating().fit(xg_train)

        P_base, P_blend, outs = [], {w: [] for w in WEIGHTS}, []
        for r in records[test]:
            h, a = r["home"], r["away"]
            if h not in params.attack or a not in params.attack:
                continue
            if not (xparams.has(h) and xparams.has(a)):  # only xG-covered matches
                continue
            lam, mu = params.rates(h, a, neutral=True)
            outs.append(wdl(r["home_goals"], r["away_goals"]))
            P_base.append(grid_wdl(lam, mu, params.rho))
            for w in WEIGHTS:
                bl, bm = blend_rates((lam, mu), xparams, h, a, w)
                P_blend[w].append(grid_wdl(bl, bm, params.rho))

        if not outs:
            continue
        outs = np.array(outs)
        k = len(outs)
        base_rps += rps(np.array(P_base), outs) * k
        n_eval += k
        for w in WEIGHTS:
            blend_rps[w] += rps(np.array(P_blend[w]), outs) * k
        print(f"{test:9s}: {k:3d} xG-covered matches")

    print(f"\nEvaluated on {n_eval} xG-covered matches (leave-one-tournament-out)\n")
    base = base_rps / n_eval
    print(f"  goals-only baseline RPS : {base:.4f}")
    best_w, best = None, base
    for w in WEIGHTS:
        r = blend_rps[w] / n_eval
        flag = "  <-- best" if r < best else ""
        if r < best:
            best, best_w = r, w
        print(f"  + xG blend w={w:<3}      RPS : {r:.4f}   ({base - r:+.4f}){flag}")
    print()
    if best_w is None:
        print("VERDICT: xG blend does NOT beat goals-only out-of-sample → keep as diagnostic, not in core.")
    else:
        print(f"VERDICT: xG blend helps; best weight w={best_w} improves RPS by {base - best:+.4f} → ship gated.")


if __name__ == "__main__":
    main()
