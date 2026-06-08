"""Tune the Dixon-Coles recency half-life by backtest RPS (gap #4)."""
from __future__ import annotations

from datetime import date

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel

WCS = {2018: date(2018, 6, 14), 2022: date(2022, 11, 20)}
def wdl(g, a): return 0 if g > a else (1 if g == a else 2)

def main():
    allm = loaders.load_results(since="2006-01-01", min_team_matches=15)
    print(f"{'half_life':>10} | pooled WC RPS")
    for hl in (365.0, 540.0, 730.0, 1100.0, 1500.0, 2000.0, None):
        P, O = [], []
        for yr, start in WCS.items():
            train = [m for m in allm if m["date"] < start]
            elo = EloModel().fit(train); mm = float(np.mean(list(elo.ratings.values())))
            dc = DixonColesModel(half_life_days=hl)
            dc.fit(train, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})
            md = MatchModel(dc)
            for m in allm:
                if (m["date"] >= start and m["date"].year == yr and m["importance"] == "world_cup"
                        and m["home_team"] in dc.params.attack and m["away_team"] in dc.params.attack):
                    P.append(list(md.wdl(m["home_team"], m["away_team"], neutral=True)))
                    O.append(wdl(m["home_score"], m["away_score"]))
        print(f"{str(hl):>10} | {rps(np.array(P), np.array(O)):.4f}")

if __name__ == "__main__":
    main()
