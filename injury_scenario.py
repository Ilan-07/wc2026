"""Injury / availability what-if tool (plan Phase B).

Answers questions like "how do the title odds move if a key player is ruled out?" — the kind of
factor that genuinely isn't in past results yet. Edit SCENARIO below.

    PYTHONPATH=src python injury_scenario.py

Honest note: this is a *scenario* tool, not a validated accuracy feature. It moves odds in the
obvious direction by a chosen magnitude (k); it is not claiming a fitted truth.
"""

from __future__ import annotations

import numpy as np

from wc2026.data import loaders, squads
from wc2026.intelligence.injuries import InjuryAdjustment, penalties_from_scenario, player_importance
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel
from wc2026.simulate.tournament import TournamentSimulator

# Who is unavailable, per team. Try your own.
SCENARIO = {
    "Argentina": ["Lionel Messi"],
    "France": ["Kylian Mbappé"],
}
K = 1.0  # severity scale


def main(n_sims: int = 20_000) -> None:
    groups = loaders.load_wc2026_groups()
    teams = [t for g in groups.values() for t in g]
    matches = loaders.load_results(since="2014-01-01", min_team_matches=20, keep_teams=set(teams))
    psi = loaders.load_shootout_psi()
    sq = squads.load_squads()

    elo = EloModel().fit(matches)
    mm = float(np.mean(list(elo.ratings.values())))
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(matches, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})

    base = TournamentSimulator(MatchModel(dc), groups, psi=psi).run(n_sims=n_sims, seed=0)

    penalties = penalties_from_scenario(sq, SCENARIO, k=K)
    adj_model = MatchModel(dc, adjustment=InjuryAdjustment(penalties))
    scen = TournamentSimulator(adj_model, groups, psi=psi).run(n_sims=n_sims, seed=0)

    print(f"Injury scenario (k={K}):")
    for team, players in SCENARIO.items():
        imp = player_importance(sq[team])
        share = sum(imp.get(p, 0.0) for p in players)
        print(f"  {team}: out = {', '.join(players)}  "
              f"(importance share {share:.0%}, penalty {penalties[team]:.2f})")

    print("\nChampion odds — baseline -> scenario:")
    affected = list(SCENARIO) + [t for t, _ in base.champion_odds()[:6] if t not in SCENARIO]
    for t in affected:
        b, s = base.reach_prob["champion"][t], scen.reach_prob["champion"][t]
        arrow = "↓" if s < b - 1e-4 else ("↑" if s > b + 1e-4 else "·")
        print(f"  {t:<13} {b:5.1%} -> {s:5.1%}  {arrow}{abs(s-b):.1%}")


if __name__ == "__main__":
    main()
