"""Validate the market-fusion layer at match level (plan Tier C, P5).

Demonstrates the fusion math on real data with real bookmaker odds: for five top leagues over
three seasons, fit a Dixon-Coles model on earlier matches, predict held-out matches, and compare
the Ranked Probability Score of three forecasts:

    * model   — the structural Dixon-Coles forecast (no market info)
    * market  — de-vigged bookmaker odds (the benchmark)
    * fused   — logarithmic opinion pool of the two, weight learned out-of-sample

The honest expectation (and the point): the market is hard to beat, so the learned model weight
is modest and the fused score lands at/near the market — confirming the spine is well-calibrated
and that combining does not hurt. The same machinery applies to WC2026 once WC odds are plugged in.

    PYTHONPATH=src python fusion_validate.py
"""

from __future__ import annotations

import numpy as np

from wc2026.collective.market import load_league
from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.fusion.divergence import flag_match
from wc2026.fusion.pool import fit_model_weight, pool_two
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel

LEAGUES = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga",
           "I1": "Serie A", "F1": "Ligue 1"}


def collect() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (model_probs, market_probs, outcomes) over held-out matches from all leagues."""
    model_p, market_p, outcomes = [], [], []
    for code, name in LEAGUES.items():
        matches = load_league(code)
        if len(matches) < 200:
            continue
        split = int(len(matches) * 0.67)  # train on first 2 seasons, test on the third
        train, test = matches[:split], matches[split:]
        dc = DixonColesModel(half_life_days=400.0)
        dc.fit(train)
        model = MatchModel(dc)
        kept = 0
        for m in test:
            if m["home_team"] not in dc.params.attack or m["away_team"] not in dc.params.attack:
                continue
            model_p.append(list(model.wdl(m["home_team"], m["away_team"], neutral=False)))
            market_p.append(list(m["market_prob"]))
            outcomes.append(m["outcome"])
            kept += 1
        print(f"  {name:<15} train {len(train):4d}  test {kept:4d}")
    return np.array(model_p), np.array(market_p), np.array(outcomes)


def main() -> None:
    print("Loading league odds + fitting Dixon-Coles per league...")
    model_p, market_p, outcomes = collect()
    n = len(outcomes)

    # Split held-out matches in two: fit the pool weight on one half, evaluate on the other.
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    fit_idx, ev_idx = perm[: n // 2], perm[n // 2 :]
    w = fit_model_weight(model_p[fit_idx], market_p[fit_idx], outcomes[fit_idx])
    fused_ev = pool_two(model_p[ev_idx], market_p[ev_idx], w)

    print(f"\nHeld-out evaluation on {len(ev_idx)} matches (lower RPS is better):")
    print(f"  model  RPS : {rps(model_p[ev_idx], outcomes[ev_idx]):.4f}")
    print(f"  market RPS : {rps(market_p[ev_idx], outcomes[ev_idx]):.4f}   (benchmark)")
    print(f"  fused  RPS : {rps(fused_ev, outcomes[ev_idx]):.4f}   (learned model weight = {w:.2f})")

    # An illustrative divergence: the match where model and market disagree most on the home side.
    d = market_p[:, 0] - model_p[:, 0]
    i = int(np.argmax(np.abs(d)))
    info = flag_match(model_p[i], market_p[i])
    print("\nLargest model/market disagreement (home result):")
    print(f"  model P(H/D/A) = {np.round(model_p[i],2)}  market = {np.round(market_p[i],2)}")
    print(f"  -> {', '.join(info['flags']) or 'no large divergence'}")


if __name__ == "__main__":
    main()
