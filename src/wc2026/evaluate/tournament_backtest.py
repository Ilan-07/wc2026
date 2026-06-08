"""Multi-tournament out-of-sample backtest (Lane 3 #5).

The headline skill claim used to rest on two tournaments (WC2018, WC2022) — the documented "can't
validate on N≈3" weakness. This widens the evidence base to every major reconstructable men's
tournament in the dataset: both World Cups **plus** the recent European Championships and Copa
Américas. For each edition we fit exactly the production rating (Elo warm-start → time-weighted
Dixon-Coles) on matches *before kickoff* and score W/D/L by RPS on that edition's matches, against
the uniform 1/3 baseline. Pooling ~8-10 tournaments and ~450 matches turns an anecdote into a
distribution: the per-tournament spread is itself the honest read on how much the single-WC numbers
can be trusted.

Run: ``PYTHONPATH=src python3 tournament_backtest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..data import loaders
from ..evaluate.metrics import log_loss, ranked_probability_score
from ..model.match_model import MatchModel
from ..ratings.dixon_coles import DixonColesModel
from ..ratings.elo import EloModel


@dataclass(frozen=True)
class TournamentSpec:
    key: str           # short id, e.g. "euro2024"
    name: str          # exact ``tournament`` string in results.csv
    year: int
    start: date        # kickoff = training cutoff
    label: str         # human label for output


# Reconstructable modern editions with enough pre-tournament history for every participant.
SPECS: list[TournamentSpec] = [
    TournamentSpec("wc2018", "FIFA World Cup", 2018, date(2018, 6, 14), "World Cup 2018"),
    TournamentSpec("wc2022", "FIFA World Cup", 2022, date(2022, 11, 20), "World Cup 2022"),
    TournamentSpec("euro2016", "UEFA Euro", 2016, date(2016, 6, 10), "Euro 2016"),
    TournamentSpec("euro2020", "UEFA Euro", 2021, date(2021, 6, 11), "Euro 2020"),
    TournamentSpec("euro2024", "UEFA Euro", 2024, date(2024, 6, 14), "Euro 2024"),
    TournamentSpec("copa2016", "Copa América", 2016, date(2016, 6, 3), "Copa América 2016"),
    TournamentSpec("copa2019", "Copa América", 2019, date(2019, 6, 14), "Copa América 2019"),
    TournamentSpec("copa2021", "Copa América", 2021, date(2021, 6, 13), "Copa América 2021"),
    TournamentSpec("copa2024", "Copa América", 2024, date(2024, 6, 20), "Copa América 2024"),
]


def _wdl(gh: int, ga: int) -> int:
    return 0 if gh > ga else (1 if gh == ga else 2)


def fit_rating(train: list[dict], half_life: float = 1100.0) -> MatchModel:
    """Production rating: Elo warm-start → time-weighted Dixon-Coles on the training matches."""
    elo = EloModel().fit(train)
    mm = float(np.mean(list(elo.ratings.values())))
    dc = DixonColesModel(half_life_days=half_life)
    dc.fit(train, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})
    return MatchModel(dc)


def backtest_one(
    spec: TournamentSpec, allm: list[dict], half_life: float = 1100.0
) -> dict | None:
    """Fit before kickoff, score W/D/L RPS on the edition's matches. None if too few are ratable."""
    train = [m for m in allm if m["date"] < spec.start]
    model = fit_rating(train, half_life=half_life)
    rated = set(model.params.attack)
    probs, outs, exp_total, act_total = [], [], [], []
    for m in allm:
        if (
            m["tournament"] == spec.name
            and m["date"].year == spec.year
            and m["date"] >= spec.start
            and m["home_team"] in rated
            and m["away_team"] in rated
        ):
            probs.append(list(model.wdl(m["home_team"], m["away_team"], neutral=True)))
            outs.append(_wdl(m["home_score"], m["away_score"]))
            lam, mu = model.rates(m["home_team"], m["away_team"], neutral=True)
            exp_total.append(lam + mu)  # model's expected total goals
            act_total.append(int(m["home_score"]) + int(m["away_score"]))
    if len(outs) < 10:
        return None
    p, o = np.array(probs), np.array(outs)
    uni = np.tile([1 / 3, 1 / 3, 1 / 3], (len(o), 1))
    return {
        "key": spec.key,
        "label": spec.label,
        "n": len(o),
        "rps": ranked_probability_score(p, o),
        "uniform_rps": ranked_probability_score(uni, o),
        "log_loss": log_loss(p, o),
        "probs": p,
        "outcomes": o,
        "exp_total": np.array(exp_total),
        "act_total": np.array(act_total),
    }


def run(specs: list[TournamentSpec] | None = None, half_life: float = 1100.0) -> dict:
    """Backtest every spec; return per-tournament rows and the match-pooled summary."""
    specs = specs or SPECS
    allm = loaders.load_results(since="2006-01-01", min_team_matches=15)
    rows = [r for s in specs if (r := backtest_one(s, allm, half_life)) is not None]

    n_tot = sum(r["n"] for r in rows)
    pooled_rps = sum(r["rps"] * r["n"] for r in rows) / n_tot
    pooled_uni = sum(r["uniform_rps"] * r["n"] for r in rows) / n_tot
    pooled_ll = sum(r["log_loss"] * r["n"] for r in rows) / n_tot
    per_tourn_rps = np.array([r["rps"] for r in rows])
    return {
        "rows": rows,
        "n_tournaments": len(rows),
        "n_matches": n_tot,
        "pooled_rps": pooled_rps,
        "pooled_uniform_rps": pooled_uni,
        "pooled_log_loss": pooled_ll,
        "skill_vs_uniform": pooled_uni - pooled_rps,
        "per_tournament_rps_mean": float(per_tourn_rps.mean()),
        "per_tournament_rps_std": float(per_tourn_rps.std(ddof=1)),
    }
