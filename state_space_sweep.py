"""Tune and gate the dynamic state-space rating (#1, Lane 1).

The state-space rating has one real knob — the signal-to-noise ratio sigma_proc/sigma_obs that
sets how fast a team's latent strength is allowed to drift — plus the draw half-band. This script
sweeps them and reports out-of-sample W/D/L RPS on the only labelled tournaments we have, the 2018
and 2022 World Cups, exactly as ``evaluate/ablation.py`` does for the squad covariates. Two honest
readouts:

  * pooled sweep  — best params over both WCs (lower bound on error; mildly optimistic with 3 knobs);
  * leave-one-out — tune on one WC, score the other (genuinely out-of-sample on the hyperparameters).

The Dixon-Coles MatchModel W/D/L on the same matches is printed as the incumbent baseline so the
dynamic rating is gated, not just plotted. Run: ``PYTHONPATH=src python3 state_space_sweep.py``.
"""

from __future__ import annotations

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.ablation import WORLD_CUPS
from wc2026.evaluate.metrics import ranked_probability_score
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel
from wc2026.ratings.state_space import StateSpaceRating


def _wdl(gh: int, ga: int) -> int:
    return 0 if gh > ga else (1 if gh == ga else 2)


def _fold(year: int, all_matches: list[dict]):
    start, _ = WORLD_CUPS[year]
    train = [m for m in all_matches if m["date"] < start]
    seen = {m["home_team"] for m in train} | {m["away_team"] for m in train}
    test = [
        m
        for m in all_matches
        if m["date"] >= start
        and m["date"].year == year
        and m["importance"] == "world_cup"
        and m["home_team"] in seen
        and m["away_team"] in seen
    ]
    return train, test, start


def _ss_rps(train, test, start, sigma_proc, sigma_obs, draw_band, home=0.30) -> tuple[float, int]:
    ss = StateSpaceRating(
        sigma_proc=sigma_proc, sigma_obs=sigma_obs, home=home, draw_band=draw_band
    )
    p = ss.fit(train, ref_date=start)
    probs, outs = [], []
    for m in test:
        probs.append(list(p.wdl(m["home_team"], m["away_team"], neutral=True, as_of=start)))
        outs.append(_wdl(m["home_score"], m["away_score"]))
    return ranked_probability_score(np.array(probs), np.array(outs)), len(test)


def _dc_rps(train, test) -> float:
    elo = EloModel().fit(train)
    init = {t: (r - 1500) / 400.0 for t, r in elo.ratings.items()}
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(train, init_attack=init)
    model = MatchModel(dc)
    probs, outs = [], []
    for m in test:
        if m["home_team"] in dc.params.attack and m["away_team"] in dc.params.attack:
            probs.append(list(model.wdl(m["home_team"], m["away_team"], neutral=True)))
            outs.append(_wdl(m["home_score"], m["away_score"]))
    return ranked_probability_score(np.array(probs), np.array(outs))


# Sweep grid: process SD (drift/day), observation SD (match noise), draw half-band.
PROC_GRID = [0.0015, 0.0025, 0.0035, 0.0050, 0.0070]
OBS_GRID = [1.10, 1.30, 1.50]
BAND_GRID = [0.50, 0.60, 0.70]


def main() -> dict:
    all_matches = loaders.load_results(since="2010-01-01", min_team_matches=15)
    folds = {yr: _fold(yr, all_matches) for yr in WORLD_CUPS}

    # Per-fold cached RPS for every grid point, plus DC baseline.
    grid = [(sp, so, bd) for sp in PROC_GRID for so in OBS_GRID for bd in BAND_GRID]
    fold_rps = {yr: {} for yr in folds}
    fold_n = {}
    dc_base = {}
    for yr, (train, test, start) in folds.items():
        fold_n[yr] = len(test)
        dc_base[yr] = _dc_rps(train, test)
        for combo in grid:
            r, _ = _ss_rps(train, test, start, *combo)
            fold_rps[yr][combo] = r

    years = list(folds)
    n_tot = sum(fold_n[y] for y in years)

    # Pooled best (optimistic): one param set minimising n-weighted RPS over both WCs.
    def pooled(combo):
        return sum(fold_rps[y][combo] * fold_n[y] for y in years) / n_tot

    best = min(grid, key=pooled)
    pooled_dc = sum(dc_base[y] * fold_n[y] for y in years) / n_tot

    # Leave-one-out on the hyperparameters: tune on the other WC, score the held-out one.
    loo_ss, loo_dc = 0.0, 0.0
    loo_detail = {}
    for test_yr in years:
        tune_yr = [y for y in years if y != test_yr][0]
        pick = min(grid, key=lambda c: fold_rps[tune_yr][c])
        ss_r = fold_rps[test_yr][pick]
        loo_ss += ss_r * fold_n[test_yr]
        loo_dc += dc_base[test_yr] * fold_n[test_yr]
        loo_detail[test_yr] = {"params": pick, "ss_rps": ss_r, "dc_rps": dc_base[test_yr]}

    result = {
        "n_matches": n_tot,
        "pooled_best_params": {"sigma_proc": best[0], "sigma_obs": best[1], "draw_band": best[2]},
        "pooled_ss_rps": pooled(best),
        "pooled_dc_rps": pooled_dc,
        "loo_ss_rps": loo_ss / n_tot,
        "loo_dc_rps": loo_dc / n_tot,
        "per_year": {
            y: {
                "n": fold_n[y],
                "dc_rps": dc_base[y],
                "ss_best_rps": min(fold_rps[y].values()),
            }
            for y in years
        },
        "loo_detail": loo_detail,
    }

    print(f"Matches scored: {n_tot} ({', '.join(f'{y}:{fold_n[y]}' for y in years)})")
    print("\n-- Pooled (best params on both WCs; mildly optimistic) --")
    bp = result["pooled_best_params"]
    print(f"  best params : sigma_proc={bp['sigma_proc']} sigma_obs={bp['sigma_obs']} band={bp['draw_band']}")
    print(f"  state-space : RPS {result['pooled_ss_rps']:.4f}")
    print(f"  Dixon-Coles : RPS {result['pooled_dc_rps']:.4f}  (incumbent baseline)")
    print("\n-- Leave-one-WC-out (tune on the other WC; honest) --")
    for y, d in loo_detail.items():
        print(f"  {y}: state-space {d['ss_rps']:.4f} vs DC {d['dc_rps']:.4f}  params={d['params']}")
    print(f"  pooled LOO  : state-space {result['loo_ss_rps']:.4f} vs DC {result['loo_dc_rps']:.4f}")
    delta = result["loo_dc_rps"] - result["loo_ss_rps"]
    verdict = "BEATS" if delta > 0 else "does NOT beat"
    print(f"\nVerdict: dynamic state-space rating {verdict} the DC baseline out-of-sample "
          f"(delta {delta:+.4f} RPS; +ve = state-space better).")
    return result


if __name__ == "__main__":
    main()
