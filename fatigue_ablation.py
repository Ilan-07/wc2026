"""Ablation gate for the fatigue (rest-days) covariate (gap #9).

Rest-days congestion is genuinely *orthogonal* to a results-based rating — a strong team on a
3-day turnaround is not weaker on paper, only on the night — so unlike squad reputation / FIFA
rank it is worth testing rather than assuming redundant. We gate it the same way as every other
layer: fit the base Dixon-Coles on pre-tournament matches, then compare out-of-sample RPS on the
2018 and 2022 World Cups WITH vs WITHOUT the fatigue shift on each match's rates.

Scope (honest): only the **rest-days** term is gated here — it is computable from match dates in
results.csv for any tournament. The cumulative-travel term needs venue coordinates we only have for
WC2026, so it is not separately validated; if rest-days fatigue does not help here, travel is not
shipped either. Run:  PYTHONPATH=src python3 fatigue_ablation.py
"""

from __future__ import annotations

from datetime import date

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.metrics import ranked_probability_score
from wc2026.intelligence.conditions import fixture_fatigue_penalties
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel

WORLD_CUPS = {2018: date(2018, 6, 14), 2022: date(2022, 11, 20)}


def _wdl(gh: int, ga: int) -> int:
    return 0 if gh > ga else (1 if gh == ga else 2)


def _wdl_delta(mm: MatchModel, home: str, away: str, delta: float) -> list[float]:
    """W/D/L probs with a per-match log-rate shift (fatigue) applied to the home side."""
    lam, mu = mm.rates(home, away, neutral=True, extra_delta=delta)
    grid = mm.dc.score_matrix(lam, mu)
    p_home = float(np.tril(grid, -1).sum())
    p_away = float(np.triu(grid, 1).sum())
    p_draw = float(np.trace(grid))
    return [p_home, p_draw, p_away]


def _fit_base(train: list[dict]) -> MatchModel:
    elo = EloModel().fit(train)
    mm = float(np.mean(list(elo.ratings.values())))
    init = {t: (r - mm) / 400.0 for t, r in elo.ratings.items()}
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(train, init_attack=init)
    return MatchModel(dc)


def run(rest_k: float = 0.04) -> dict:
    allm = loaders.load_results(since="2006-01-01", min_team_matches=15)
    base_p, fat_p, outcomes = [], [], []
    per_year = {}
    for year, start in WORLD_CUPS.items():
        train = [m for m in allm if m["date"] < start]
        wc = [
            m for m in allm
            if m["date"] >= start and m.get("importance") == "world_cup"
            and getattr(m["date"], "year", None) == year
            and m.get("home_score") is not None and m.get("away_score") is not None
        ]
        if not wc:
            continue
        model = _fit_base(train)
        if any(t not in model.params.attack for m in wc for t in (m["home_team"], m["away_team"])):
            wc = [m for m in wc if m["home_team"] in model.params.attack
                  and m["away_team"] in model.params.attack]
        # rest-days fatigue from this tournament's own schedule (no venue coords → travel off)
        fixtures = [(m["date"], m["home_team"], m["away_team"], None) for m in wc]
        fatigue = fixture_fatigue_penalties(fixtures, rest_k=rest_k, use_travel=False)

        yb, yf, yo = [], [], []
        for m in wc:
            h, a = m["home_team"], m["away_team"]
            yb.append(list(model.wdl(h, a, neutral=True)))
            pens = fatigue.get(frozenset((h, a)), {})
            delta = pens.get(a, 0.0) - pens.get(h, 0.0)
            yf.append(_wdl_delta(model, h, a, delta))
            yo.append(_wdl(int(m["home_score"]), int(m["away_score"])))
        o = np.array(yo)
        per_year[year] = (
            ranked_probability_score(np.array(yb), o),
            ranked_probability_score(np.array(yf), o),
            len(wc),
        )
        base_p += yb; fat_p += yf; outcomes += yo

    o = np.array(outcomes)
    pooled = (
        ranked_probability_score(np.array(base_p), o),
        ranked_probability_score(np.array(fat_p), o),
        len(outcomes),
    )
    return {"per_year": per_year, "pooled": pooled}


if __name__ == "__main__":
    res = run()
    print("Fatigue (rest-days) ablation — base RPS vs +fatigue RPS (lower is better):")
    for year, (b, f, n) in res["per_year"].items():
        print(f"  WC{year}: base {b:.4f}  +fatigue {f:.4f}  (delta {f - b:+.4f}, n={n})")
    b, f, n = res["pooled"]
    print(f"  POOLED: base {b:.4f}  +fatigue {f:.4f}  (delta {f - b:+.4f}, n={n})")
    print("\nSHIP into the default forecast only if pooled delta < 0 (improves).")
