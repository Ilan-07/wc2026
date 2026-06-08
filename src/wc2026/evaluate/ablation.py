"""Ablation gate for the squad-feature covariates (plan validation section).

The honest question: do the squad features add predictive value *beyond* the results-based
Dixon-Coles/Elo rating, or do they just re-encode strength the rating already has?

We answer it the only rigorous way available before WC2026 is played: backtest on the 2018 and
2022 World Cups, where we have both the squad pages and the actual results. For each tournament
we fit the base model on matches *before* kickoff, then measure RPS on the tournament's matches
with and without the squad adjustment. Weights are validated **leave-one-tournament-out** (fit
on one World Cup, evaluate on the other) so the reported improvement is genuinely out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import minimize

from ..data import loaders, squads
from ..evaluate.metrics import ranked_probability_score
from ..intelligence.covariates import SquadAdjustment, zscore_features
from ..intelligence.squads import all_squad_features
from ..model.match_model import MatchModel
from ..ratings.dixon_coles import DixonColesModel, tau
from ..ratings.elo import EloModel

# World Cup kickoff dates (training cutoff) and squad-page files.
WORLD_CUPS = {
    2018: (date(2018, 6, 14), "wc2018_squads.json"),
    2022: (date(2022, 11, 20), "wc2022_squads.json"),
}


def _wdl(gh: int, ga: int) -> int:
    return 0 if gh > ga else (1 if gh == ga else 2)


@dataclass
class _Fold:
    year: int
    dc: DixonColesModel
    names: list[str]
    z: dict[str, np.ndarray]
    matches: list[dict]  # tournament matches with base lam/mu and feature diff precomputed


def _build_fold(year: int, all_matches: list[dict], raw_dir) -> _Fold:
    start, squad_file = WORLD_CUPS[year]
    train = [m for m in all_matches if m["date"] < start]

    # base model: Elo warm-start + Dixon-Coles on pre-tournament matches
    elo = EloModel().fit(train)
    init = {t: (r - 1500) / 400.0 for t, r in elo.ratings.items()}
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(train, init_attack=init)

    feats = all_squad_features(squads.load_squads(f"{raw_dir}/{squad_file}"))
    feats = {t: f for t, f in feats.items() if t in dc.params.attack}
    names, z = zscore_features(feats)
    zero = np.zeros(len(names))

    tourn = [
        m
        for m in all_matches
        if m["date"] >= start
        and m["date"].year == year
        and m["importance"] == "world_cup"
        and m["home_team"] in dc.params.attack
        and m["away_team"] in dc.params.attack
    ]
    rows = []
    for m in tourn:
        lam, mu = dc.params.rates(m["home_team"], m["away_team"], neutral=True)
        d = z.get(m["home_team"], zero) - z.get(m["away_team"], zero)
        rows.append({**m, "lam": lam, "mu": mu, "dvec": d})
    return _Fold(year=year, dc=dc, names=names, z=z, matches=rows)


def _fit_theta(folds: list[_Fold], reg: float = 1.0) -> np.ndarray:
    """Maximum-likelihood squad weights across the given folds (L2-regularized)."""
    n_feat = len(folds[0].names)

    def neg_ll(theta):
        total = 0.0
        for fold in folds:
            rho = fold.dc.params.rho
            for m in fold.matches:
                d = float(theta @ m["dvec"])
                lam, mu = m["lam"] * np.exp(d), m["mu"] * np.exp(-d)
                gh, ga = m["home_score"], m["away_score"]
                ll = gh * np.log(lam) - lam + ga * np.log(mu) - mu
                ll += np.log(max(tau(gh, ga, lam, mu, rho), 1e-12))
                total -= ll
        return total + reg * float(theta @ theta)

    res = minimize(neg_ll, np.zeros(n_feat), method="L-BFGS-B")
    return res.x


def _predict_rps(fold: _Fold, theta: np.ndarray) -> tuple[float, float]:
    """Return (base RPS, adjusted RPS) on a fold's tournament matches."""
    adj = SquadAdjustment(fold.names, fold.z, theta)
    base_model = MatchModel(fold.dc)
    adj_model = MatchModel(fold.dc, adjustment=adj)
    base_p, adj_p, outcomes = [], [], []
    for m in fold.matches:
        base_p.append(list(base_model.wdl(m["home_team"], m["away_team"], neutral=True)))
        adj_p.append(list(adj_model.wdl(m["home_team"], m["away_team"], neutral=True)))
        outcomes.append(_wdl(m["home_score"], m["away_score"]))
    o = np.array(outcomes)
    return (
        ranked_probability_score(np.array(base_p), o),
        ranked_probability_score(np.array(adj_p), o),
    )


def run_ablation(reg: float = 1.0, raw_dir: str | None = None) -> dict:
    """Leave-one-tournament-out ablation. Returns per-year and pooled base vs +squad RPS."""
    from pathlib import Path

    raw = raw_dir or str(Path(__file__).resolve().parents[3] / "data" / "raw")
    all_matches = loaders.load_results(since="2010-01-01", min_team_matches=15)
    folds = {yr: _build_fold(yr, all_matches, raw) for yr in WORLD_CUPS}

    years = list(folds)
    per_year = {}
    pooled_base_w, pooled_adj_w, n_total = 0.0, 0.0, 0
    learned_theta = {}
    for test_yr in years:
        train_folds = [folds[y] for y in years if y != test_yr]
        theta = _fit_theta(train_folds, reg=reg)
        learned_theta[test_yr] = dict(zip(folds[test_yr].names, theta))
        base_rps, adj_rps = _predict_rps(folds[test_yr], theta)
        n = len(folds[test_yr].matches)
        per_year[test_yr] = {"n": n, "base_rps": base_rps, "adj_rps": adj_rps}
        pooled_base_w += base_rps * n
        pooled_adj_w += adj_rps * n
        n_total += n

    return {
        "per_year": per_year,
        "pooled_base_rps": pooled_base_w / n_total,
        "pooled_adj_rps": pooled_adj_w / n_total,
        "n_matches": n_total,
        "theta_out_of_sample": learned_theta,
    }
