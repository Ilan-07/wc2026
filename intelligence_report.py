"""Per-team intelligence report — the cathedral's valid output: a *traceable* explanation.

Builds the knowledge graph, runs the deterministic reasoning pipeline (Analyst / Market / Contrarian
/ Judge), and prints an auditable breakdown of a team's forecast. This does NOT change the number —
it explains it.

    PYTHONPATH=src python intelligence_report.py [Team ...]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from wc2026.collective import market
from wc2026.data import loaders, venues
from wc2026.data import squads as squads_mod
from wc2026.fusion.pool import pool_two
from wc2026.graph.kg import build_kg
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel
from wc2026.reports.explain import ReportContext, explain_team
from wc2026.simulate.tournament import TournamentSimulator


def altitude_cities(path=None) -> dict[str, set]:
    df = pd.read_csv(loaders.DEFAULT_RESULTS if path is None else path)
    df = df[(df["tournament"] == "FIFA World Cup")]
    out: dict[str, set] = {}
    groups = loaders.load_wc2026_groups()
    gof = {t: g for g, ts in groups.items() for t in ts}
    for r in df.itertuples(index=False):
        if str(r.date).startswith("2026") and gof.get(r.home_team) == gof.get(r.away_team):
            v = venues.venue_for_city(str(r.city))
            if v and v.altitude_m >= 1000:
                out.setdefault(r.home_team, set()).add(v.city)
                out.setdefault(r.away_team, set()).add(v.city)
    return out


def main(targets: list[str]) -> None:
    groups = loaders.load_wc2026_groups()
    teams = [t for g in groups.values() for t in g]
    matches = loaders.load_results(since="2014-01-01", min_team_matches=20, keep_teams=set(teams))
    sq = squads_mod.load_squads()
    caps = {t: {p.name: p.caps for p in s.players} for t, s in sq.items()}

    elo = EloModel().fit(matches)
    mm = float(np.mean(list(elo.ratings.values())))
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(matches, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})
    psi = loaders.load_shootout_psi()
    va = loaders.load_wc2026_group_venue_altitudes()
    res = TournamentSimulator(MatchModel(dc), groups, psi=psi, group_venue_altitudes=va).run(
        n_sims=20000, seed=0)
    model_p = res.reach_prob["champion"]
    market_p = market.devig_outright(market.load_outright_odds(), teams)
    mr = np.array([[model_p[t] for t in teams]])
    kr = np.array([[market_p[t] for t in teams]])
    blended_p = dict(zip(teams, pool_two(mr, kr, 0.35)[0]))

    fv = [(r.home_team, r.away_team, r.city) for r in
          pd.read_csv(loaders.DEFAULT_RESULTS).itertuples(index=False)
          if str(r.tournament) == "FIFA World Cup" and str(r.date).startswith("2026")]
    kg = build_kg(groups, sq, fv)
    print("Knowledge graph:", kg.stats())

    ctx = ReportContext(
        teams=teams, elo={t: elo.rating(t) for t in teams}, groups=groups,
        model_p=model_p, market_p=market_p, blended_p=blended_p, kg=kg, caps=caps,
        altitude_cities=altitude_cities(),
    )

    if not targets:
        targets = [max(blended_p, key=lambda t: blended_p[t]), "France", "Mexico"]
    for t in targets:
        if t not in teams:
            print(f"\n(unknown team: {t})")
            continue
        print("\n" + explain_team(ctx, t).render())


if __name__ == "__main__":
    main(sys.argv[1:])
