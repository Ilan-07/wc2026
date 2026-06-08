"""Does a learned tau + pooled home advantage beat the plain hierarchical Poisson? (#2, Lane 1)

Three-way head-to-head on the 2022 World Cup, fit on pre-tournament data:

  1. MLE Dixon-Coles                    — the frequentist incumbent (fixed rho, global home);
  2. Hierarchical Bayesian Poisson      — partial pooling, rho=0, global home (already shipped);
  3. Bayesian tau + pooled home         — adds a *learned* DC rho and *per-team* home advantage.

The question for the gate: do the two extra pieces (low-score dependence + venue deconfounding)
move out-of-sample WC RPS, or are they redundant once the hierarchy is in place? One World Cup is a
thin test — reported honestly. Also prints rho's posterior and the spread of per-team home effects
(what the pooling actually learned). Run: ``PYTHONPATH=src python3 bayesian_tau_ablation.py``.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.model.match_model import MatchModel
from wc2026.ratings.bayesian_dc import BayesianDixonColes, BayesianDixonColesTauHome
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel

START = date(2022, 11, 20)


def wdl(g, a):
    return 0 if g > a else (1 if g == a else 2)


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
    test = [
        m for m in allm
        if m["date"] >= START and m["date"].year == 2022 and m["importance"] == "world_cup"
    ]
    print(f"{len(train)} train / {len(test)} WC2022 matches")

    # 1. frequentist MLE
    elo = EloModel().fit(train)
    mm = float(np.mean(list(elo.ratings.values())))
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(train, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})
    mle_rps, n = predict_rps(MatchModel(dc), test, dc.params.attack)

    # 2. hierarchical Bayesian Poisson (rho=0, global home)
    print("sampling hierarchical Poisson (rho=0)...")
    b0 = BayesianDixonColes(draws=500, tune=500, chains=2, seed=0)
    p0 = b0.fit(train)
    pois = DixonColesModel(); pois.params = p0
    pois_rps, _ = predict_rps(MatchModel(pois), test, p0.attack)

    # 3. Bayesian tau + pooled home
    print("sampling Bayesian tau + pooled home (the new model)...")
    b1 = BayesianDixonColesTauHome(draws=500, tune=500, chains=2, seed=0)
    p1 = b1.fit(train)
    diag = b1.diagnostics()
    tau = DixonColesModel(); tau.params = p1
    tau_rps, _ = predict_rps(MatchModel(tau), test, p1.attack)

    print(f"\nWC2022 RPS ({n} matches):")
    print(f"  1. MLE Dixon-Coles            : {mle_rps:.4f}")
    print(f"  2. Hierarchical Poisson       : {pois_rps:.4f}")
    print(f"  3. Bayesian tau + pooled home : {tau_rps:.4f}")
    d_pois = pois_rps - tau_rps
    print(f"\n  tau+home vs Poisson : {d_pois:+.4f} RPS ({'better' if d_pois > 0 else 'worse/equal'})")
    print(f"  tau+home vs MLE     : {mle_rps - tau_rps:+.4f} RPS")

    print("\ndiagnostics:")
    print(f"  max R-hat {diag['max_rhat']:.3f} (all params), "
          f"{diag['forecast_max_rhat']:.3f} (forecast params att/def/rho), "
          f"divergences {diag['divergences']}")
    print(f"  learned rho = {diag['rho_mean']:+.4f} ± {diag['rho_sd']:.4f}  (0 => no low-score dependence)")
    print(f"  global home = {diag['home_mu_mean']:+.3f}, between-team home sigma = {diag['home_sigma_mean']:.3f}")

    home = b1.team_home_advantage()
    ranked = sorted(home.items(), key=lambda kv: kv[1], reverse=True)
    print("\n  strongest pooled home advantage:", ", ".join(f"{t} {v:+.2f}" for t, v in ranked[:5]))
    print("  weakest pooled home advantage  :", ", ".join(f"{t} {v:+.2f}" for t, v in ranked[-5:]))

    return {"mle": mle_rps, "poisson": pois_rps, "tau_home": tau_rps, "diagnostics": diag}


if __name__ == "__main__":
    main()
