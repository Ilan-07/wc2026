"""Hierarchical Bayesian vs frequentist MLE — does partial pooling improve the forecast? (gap #2)

Fits both on pre-WC2022 data, predicts the 2022 World Cup matches, and compares RPS. Also reports
sampler diagnostics (R-hat, divergences) and how the two rate low-data minnows (where pooling
should help most). One World Cup is a thin test, but it's the honest head-to-head.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.model.match_model import MatchModel
from wc2026.ratings.bayesian_dc import BayesianDixonColes
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel

START = date(2022, 11, 20)
def wdl(g, a): return 0 if g > a else (1 if g == a else 2)


def predict_rps(model, test, attack):
    P, O = [], []
    for m in test:
        if m["home_team"] in attack and m["away_team"] in attack:
            P.append(list(model.wdl(m["home_team"], m["away_team"], neutral=True)))
            O.append(wdl(m["home_score"], m["away_score"]))
    return rps(np.array(P), np.array(O)), len(O)


def main():
    allm = loaders.load_results(since="2014-01-01", min_team_matches=20)
    train = [m for m in allm if m["date"] < START]
    test = [m for m in allm if m["date"] >= START and m["date"].year == 2022
            and m["importance"] == "world_cup"]
    print(f"{len(train)} train / {len(test)} WC2022 matches")

    # frequentist MLE
    elo = EloModel().fit(train); mm = float(np.mean(list(elo.ratings.values())))
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(train, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})
    mle_rps, n = predict_rps(MatchModel(dc), test, dc.params.attack)

    # hierarchical Bayesian
    print("sampling hierarchical Bayesian model on real data (this is the slow part)...")
    b = BayesianDixonColes(draws=500, tune=500, chains=2, seed=0)
    bp = b.fit(train)
    print("diagnostics:", b.diagnostics())
    bdc = DixonColesModel(); bdc.params = bp
    bay_rps, _ = predict_rps(MatchModel(bdc), test, bp.attack)

    print(f"\nWC2022 RPS ({n} matches):")
    print(f"  MLE (current)        : {mle_rps:.4f}")
    print(f"  Hierarchical Bayesian: {bay_rps:.4f}   ({'better' if bay_rps < mle_rps else 'worse/equal'} by {abs(mle_rps-bay_rps):.4f})")

    # minnow shrinkage check: teams with the fewest matches
    from collections import Counter
    cnt = Counter()
    for m in train:
        cnt[m["home_team"]] += 1; cnt[m["away_team"]] += 1
    fewest = [t for t, _ in cnt.most_common()[:-8:-1] if t in dc.params.attack]
    print("\nminnow attack ratings (MLE vs Bayesian; Bayesian should shrink harder to 0):")
    for t in fewest:
        print(f"  {t:<18} n={cnt[t]:3d}  MLE {dc.params.attack[t]:+.2f}  Bayes {bp.attack.get(t,0):+.2f}")


if __name__ == "__main__":
    main()
