"""Grade the model against reality — the working-model proof.

Scores the model's W/D/L predictions for the 2018 and 2022 World Cups against the actual results
(its real out-of-sample track record), and demonstrates champion scoring. Run this now to see how
the model *actually did*; after WC2026, point score.score_archive at data/processed/forecasts/.

    PYTHONPATH=src python score_backtest.py
"""

from __future__ import annotations

from datetime import date

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.score import score_champion, score_matches
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel
from wc2026.ratings.xg_rating import XGRating

WCS = {2018: (date(2018, 6, 14), "France"), 2022: (date(2022, 11, 20), "Argentina")}
XG_W = 0.4  # shipped xG blend weight


def wdl(g, a):
    return 0 if g > a else (1 if g == a else 2)


def _wrap(params, xg_blend=None):
    dc = DixonColesModel(half_life_days=1100.0)
    dc.params = params
    return MatchModel(dc, xg_blend=xg_blend)


def _score_on_wc(model, allm, yr, start, teams):
    probs, outs = [], []
    for m in allm:
        if (m["date"] >= start and m["date"].year == yr and m["importance"] == "world_cup"
                and m["home_team"] in teams and m["away_team"] in teams):
            probs.append(list(model.wdl(m["home_team"], m["away_team"], neutral=True)))
            outs.append(wdl(m["home_score"], m["away_score"]))
    return score_matches(probs, outs), outs


def main():
    import json
    from pathlib import Path
    allm = loaders.load_results(since="2006-01-01", min_team_matches=15)
    xg_records = json.loads(Path("data/raw/sb_match_records.json").read_text())

    print("Grading the COMBINED system (Bayesian rating + temporally-clean xG blend) vs the MLE base:\n")
    for yr, (start, _winner) in WCS.items():
        train = [m for m in allm if m["date"] < start]
        elo = EloModel().fit(train)
        mm = float(np.mean(list(elo.ratings.values())))
        init = {t: (r - mm) / 400.0 for t, r in elo.ratings.items()}

        # baseline: MLE Dixon-Coles, no xG
        dc = DixonColesModel(half_life_days=1100.0)
        dc.fit(train, init_attack=init)
        teams = set(dc.params.attack)
        base_s, outs = _score_on_wc(MatchModel(dc), allm, yr, start, teams)

        # xG rating from ONLY tournaments played before this World Cup (no leakage)
        prior_xg = [r for recs in xg_records.values() for r in recs if r["date"] < start.isoformat()]
        xparams = XGRating().fit(prior_xg) if prior_xg else None
        xg_blend = (xparams, XG_W) if xparams else None
        n_cov = sum(xparams.has(t) for t in teams) if xparams else 0

        # combined: Bayesian rating + xG blend
        from wc2026.ratings.bayesian_dc import BayesianDixonColes
        bparams = BayesianDixonColes(draws=500, tune=500, chains=2).fit(train)
        comb_s, _ = _score_on_wc(_wrap(bparams, xg_blend), allm, yr, start, set(bparams.attack))

        uni = score_matches(np.tile([1 / 3, 1 / 3, 1 / 3], (len(outs), 1)), outs)
        xg_note = f"xG on {n_cov} teams from {len(prior_xg)} prior matches" if prior_xg else "no prior xG"
        print(f"WC{yr} ({base_s['n']:.0f} matches, uniform RPS {uni['rps']:.4f}):")
        print(f"    MLE base      RPS {base_s['rps']:.4f} | log-loss {base_s['log_loss']:.4f}")
        print(f"    COMBINED      RPS {comb_s['rps']:.4f} | log-loss {comb_s['log_loss']:.4f}"
              f"   ({base_s['rps'] - comb_s['rps']:+.4f})   [{xg_note}]")

    # champion-scoring demo on the archived live WC2026 forecast (winner TBD -> illustrative)
    print("\nChampion-scoring demo (illustrative — uses the live archive vs a hypothetical winner):")
    import json
    from pathlib import Path
    archs = sorted(Path("data/processed/forecasts").glob("forecast_*.json"))
    if archs:
        rec = json.loads(archs[-1].read_text())
        for hypothetical in (rec["pick"], "Spain"):
            print("  if winner =", hypothetical, "->", score_champion(rec["champion_odds"], hypothetical))


if __name__ == "__main__":
    main()
