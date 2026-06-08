"""Monte Carlo World Cup 2026 simulator (plan Tier A5).

Runs the full tournament many times, sampling every match from the fitted match model, and
aggregates how often each team reaches each stage. Every headline number (champion odds,
finalist %, qualification %, dark horses) is just a different count over the same runs, with
a Monte Carlo standard error of sqrt(p(1-p)/N).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..model.match_model import MatchModel
from . import bracket as bracketmod
from . import format as fmt

# Knockout stage labels in order; index = how far a team advanced.
STAGES = ["group", "r32", "r16", "qf", "sf", "final", "champion"]


@dataclass
class SimulationResult:
    n_sims: int
    teams: list[str]
    # stage -> {team: probability of reaching at least this stage}
    reach_prob: dict[str, dict[str, float]]

    def champion_odds(self) -> list[tuple[str, float]]:
        return sorted(self.reach_prob["champion"].items(), key=lambda x: x[1], reverse=True)

    def standard_error(self, p: float) -> float:
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / self.n_sims))

    def table(self, top: int = 16) -> str:
        rows = ["team            champ    final     sf     qf   (±se champ)"]
        champ = self.reach_prob["champion"]
        for team, p in sorted(champ.items(), key=lambda x: x[1], reverse=True)[:top]:
            rows.append(
                f"{team:<14} {p:6.1%}  {self.reach_prob['final'][team]:6.1%} "
                f"{self.reach_prob['sf'][team]:6.1%} {self.reach_prob['qf'][team]:6.1%}"
                f"   ±{self.standard_error(p):.1%}"
            )
        return "\n".join(rows)


# Host nations get a home-advantage boost (matches are mostly played in these countries).
HOST_TEAMS = {"United States", "Canada", "Mexico"}


class TournamentSimulator:
    def __init__(
        self,
        model: MatchModel | list[MatchModel],
        groups: dict[str, list[str]],
        psi: dict[str, float] | None = None,
        host_teams: set[str] | None = None,
        known_group_results: dict | None = None,
        group_venue_altitudes: dict | None = None,
        altitude_per_1000m: float = 0.12,
        group_fixture_fatigue: dict | None = None,
        known_bracket: list | None = None,
        known_ko_results: dict | None = None,
        shootout_model=None,
        rotation_penalty: float = 0.0,
    ):
        if len(groups) != 12 or any(len(v) != 4 for v in groups.values()):
            raise ValueError("WC2026 needs exactly 12 groups of 4 teams")
        # Accept either a single model or a bootstrap ensemble (uncertainty propagation):
        # each simulation draws one ensemble member so champion odds carry parameter uncertainty.
        self.models = model if isinstance(model, list) else [model]
        self.groups = groups
        self.psi = psi
        # Optional fitted shootout win-propensity model (model/shootout.py). When supplied it sets
        # the penalty-shootout bias from the learned scale instead of the hand-set 0.4 (#3).
        self.shootout_model = shootout_model
        # Dead-rubber rotation (Gap 3): log-rate down-weight applied to a team already guaranteed top-2
        # (6 pts after two matchdays) in its final group game — the resting-starters effect. Default off
        # (gated like fatigue); a positive value reduces a clinched side's scoring rate in that fixture.
        self.rotation_penalty = rotation_penalty
        self.hosts = HOST_TEAMS if host_teams is None else host_teams
        self.teams = [t for g in groups.values() for t in g]
        # Conditioning (live forecasting): group matches already played are locked to their real
        # scores instead of being simulated, so the forecast updates as the tournament unfolds.
        # Keyed by frozenset({home, away}) -> (home_team, home_score, away_score).
        self.known = known_group_results or {}
        # Per-group-fixture venue altitude (m), for the altitude effect at the Mexican venues.
        self.venue_alt = group_venue_altitudes or {}
        self.alt_k = altitude_per_1000m
        # Per-group-fixture fatigue penalties {frozenset({h,a}): {team: penalty}} (rest-days +
        # cumulative travel). Orthogonal to the rating; ablation-gated (fatigue_ablation.py). Empty
        # by default (dormant) unless explicitly supplied, since the gate is the deciding evidence.
        self.fatigue = group_fixture_fatigue or {}
        # Knockout conditioning (live, second half of the tournament): once the group stage is
        # complete the real Round-of-32 bracket is fixed — supply it as ``known_bracket`` (32 team
        # names in bracket-position order) and the group stage is skipped. ``known_ko_results`` locks
        # already-played knockout games: {frozenset({a, b}): winner}. Both populate as fixtures land.
        if known_bracket is not None and len(known_bracket) != 32:
            raise ValueError("known_bracket must list exactly 32 teams in bracket order")
        self.known_bracket = list(known_bracket) if known_bracket is not None else None
        self.known_ko = known_ko_results or {}

    # --------------------------------------------------------------- one run
    def _alt_delta(self, home: str, away: str) -> float:
        """Altitude log-rate shift for the *home*-oriented side (delta = pen_away - pen_home)."""
        alt = self.venue_alt.get(frozenset((home, away)))
        if not alt:
            return 0.0
        from ..data.venues import home_altitude

        pen = lambda t: self.alt_k * max(0, alt - home_altitude(t)) / 1000.0
        return pen(away) - pen(home)

    def _fatigue_delta(self, home: str, away: str) -> float:
        """Fatigue log-rate shift for the *home*-oriented side (pen_away - pen_home)."""
        pens = self.fatigue.get(frozenset((home, away)))
        if not pens:
            return 0.0
        return pens.get(away, 0.0) - pens.get(home, 0.0)

    def _extra(self, home: str, away: str) -> float:
        """Combined per-match log-rate shift (altitude + fatigue), home-oriented."""
        return self._alt_delta(home, away) + self._fatigue_delta(home, away)

    def _sample(self, model: MatchModel, a: str, b: str, rng, extra: float = 0.0) -> tuple[int, int]:
        """Goals (a, b) with host advantage, altitude, fatigue, and an optional home-oriented
        ``extra`` log-rate shift (positive favours ``a``; used for dead-rubber rotation)."""
        if b in self.hosts and a not in self.hosts:
            gb, ga = model.sample_score(b, a, rng, neutral=False, extra_delta=self._extra(b, a) - extra)
            return ga, gb
        neutral = not (a in self.hosts and b not in self.hosts)
        return model.sample_score(a, b, rng, neutral=neutral, extra_delta=self._extra(a, b) + extra)

    def _ko(self, model: MatchModel, a: str, b: str, rng) -> str:
        """Knockout winner — locked to the real result if this tie has already been played."""
        known = self.known_ko.get(frozenset((a, b)))
        if known is not None:
            return known
        one_host = (a in self.hosts) != (b in self.hosts)
        if one_host and b in self.hosts:
            a, b = b, a  # put the host as the home side
        return model.sample_knockout(
            a, b, rng, neutral=not one_host, psi=self.psi, shootout_model=self.shootout_model
        )

    def _simulate_group(self, model: MatchModel, teams: list[str], rng) -> list[fmt.TeamRecord]:
        records = {t: fmt.TeamRecord(t) for t in teams}
        final_md = fmt.ROUND_ROBIN_PAIRS[-2:]  # matchday 3 (each team's dead-rubber-capable game)
        for a, b in fmt.ROUND_ROBIN_PAIRS:
            ta, tb = teams[a], teams[b]
            real = self.known.get(frozenset((ta, tb)))
            if real is not None:
                hteam, hs, as_ = real
                ga, gb = (hs, as_) if hteam == ta else (as_, hs)
            else:
                extra = 0.0
                if self.rotation_penalty and (a, b) in final_md:
                    # 6 pts after two matchdays = guaranteed top-2 in a 4-team group → rest starters.
                    da = self.rotation_penalty if records[ta].points >= 6 else 0.0
                    db = self.rotation_penalty if records[tb].points >= 6 else 0.0
                    extra = db - da  # home-oriented: a clinched home side scores less
                ga, gb = self._sample(model, ta, tb, rng, extra=extra)
            records[ta].add_match(tb, ga, gb)
            records[tb].add_match(ta, gb, ga)
        return fmt.rank_group(list(records.values()), rng)

    def _simulate_once(self, model: MatchModel, rng) -> dict[str, int]:
        """Return {team: stage_index_reached} for a single tournament."""
        reached = {t: 0 for t in self.teams}  # 0 = group stage

        if self.known_bracket is not None:
            # Group stage is over: use the real Round-of-32 bracket directly.
            bracket = list(self.known_bracket)
        else:
            # Simulate groups, then place qualifiers into the OFFICIAL bracket template (so the
            # bracket auto-assembles correctly from group identity, not a performance seeding).
            winners: dict[str, str] = {}
            runners: dict[str, str] = {}
            thirds_list = []
            for g, teams in self.groups.items():
                ranked = self._simulate_group(model, teams, rng)
                winners[g] = ranked[0].team
                runners[g] = ranked[1].team
                thirds_list.append((g, ranked[2].team, ranked[2]))
            best = sorted(thirds_list, key=lambda x: (x[2].points, x[2].gd, x[2].gf, rng.random()),
                          reverse=True)[:8]
            thirds_by_group = {g: team for g, team, _ in best}
            bracket = bracketmod.build_official_bracket(winners, runners, thirds_by_group)

        for t in bracket:
            reached[t] = 1  # reached R32

        # Single elimination. Each round halves the field; stage index grows.
        alive = bracket
        stage = 1  # currently at r32
        while len(alive) > 1:
            stage += 1
            next_round = []
            for i in range(0, len(alive), 2):
                w = self._ko(model, alive[i], alive[i + 1], rng)
                reached[w] = stage
                next_round.append(w)
            alive = next_round
        # Winner of the final reaches "champion" (index 6); final loser stays at 5.
        return reached

    # ----------------------------------------------------------------- run N
    def run(self, n_sims: int = 50_000, seed: int = 0) -> SimulationResult:
        rng = np.random.default_rng(seed)
        # counts[stage_index][team]
        counts = [defaultdict(int) for _ in STAGES]
        n_models = len(self.models)
        for k in range(n_sims):
            # draw an ensemble member for this run (uncertainty propagation)
            model = self.models[k % n_models] if n_models > 1 else self.models[0]
            reached = self._simulate_once(model, rng)
            for team, st in reached.items():
                # team that reached stage `st` also reached every earlier stage
                for s in range(st + 1):
                    counts[s][team] += 1

        reach_prob: dict[str, dict[str, float]] = {}
        for s, name in enumerate(STAGES):
            reach_prob[name] = {t: counts[s][t] / n_sims for t in self.teams}
        return SimulationResult(n_sims=n_sims, teams=self.teams, reach_prob=reach_prob)
