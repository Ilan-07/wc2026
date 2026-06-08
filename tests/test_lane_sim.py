"""Lane 4 (sim-engine) tests: fatigue covariate (#9) and uncertainty decomposition (#11)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from wc2026.data import loaders, synthetic
from wc2026.intelligence.conditions import (
    FatigueAdjustment,
    fatigue_penalty,
    fixture_fatigue_penalties,
)
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.simulate.format import ROUND_ROBIN_PAIRS
from wc2026.simulate.tournament import TournamentSimulator


# --------------------------------------------------------------------- fatigue (#9)
def test_fatigue_penalty_monotonic():
    # Shorter rest => bigger penalty; more travel => bigger penalty; ample rest, no travel => 0.
    assert fatigue_penalty(2.0) > fatigue_penalty(3.0) > fatigue_penalty(4.0)
    assert fatigue_penalty(4.0) == 0.0
    assert fatigue_penalty(3.0, 8000.0) > fatigue_penalty(3.0, 0.0)


def test_fatigue_adjustment_is_antisymmetric():
    adj = FatigueAdjustment({"A": 0.10, "B": 0.0})
    assert adj("A", "B") == pytest.approx(-0.10)   # fatigued home A scores less
    assert adj("B", "A") == pytest.approx(0.10)
    assert adj("A", "B") == -adj("B", "A")


def test_fixture_fatigue_short_rest_accrues_penalty():
    d = dt.date(2026, 6, 11)
    rows = [
        (d, "A", "B", None),                       # both teams' first match: no deficit
        (d + dt.timedelta(days=2), "A", "C", None),  # A on 2-day turnaround vs C's first match
    ]
    pens = fixture_fatigue_penalties(rows, use_travel=False)
    second = pens[frozenset(("A", "C"))]
    assert second["A"] > 0.0          # A is fatigued (2-day rest < baseline)
    assert second["C"] == 0.0         # C is fresh (first appearance)


def test_load_wc2026_group_fixture_fatigue_shape():
    fat = loaders.load_wc2026_group_fixture_fatigue()
    assert fat, "expected non-empty WC2026 group fatigue map"
    for key, pens in fat.items():
        assert len(key) == 2 and len(pens) == 2
        assert all(np.isfinite(v) and v >= 0.0 for v in pens.values())


# --------------------------------------------- fatigue plumbing through the simulator
@pytest.fixture(scope="module")
def fitted_model():
    teams = synthetic.make_teams(48, seed=0)
    history = synthetic.generate_history(teams, n_matches=4000, seed=1)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(history)
    return MatchModel(dc), teams


def test_simulator_accepts_fatigue_and_it_bites(fitted_model):
    model, teams = fitted_model
    groups = synthetic.make_group_draw(teams, seed=2)
    base = TournamentSimulator(model, groups).run(n_sims=4000, seed=7)

    # Heavily fatigue one team in each of its group fixtures and confirm it qualifies less.
    victim = groups["A"][0]
    fatigue = {}
    for ts in groups.values():
        for ia, ib in ROUND_ROBIN_PAIRS:
            h, a = ts[ia], ts[ib]
            pens = {h: 0.0, a: 0.0}
            if victim in (h, a):
                pens[victim] = 0.6  # large handicap
            fatigue[frozenset((h, a))] = pens
    tired = TournamentSimulator(model, groups, group_fixture_fatigue=fatigue).run(n_sims=4000, seed=7)

    assert tired.reach_prob["r32"][victim] < base.reach_prob["r32"][victim]
    # Probabilities stay valid.
    assert all(0.0 <= p <= 1.0 for p in tired.reach_prob["champion"].values())


# ----------------------------------------------------- uncertainty decomposition (#11)
def test_monte_carlo_standard_error_formula(fitted_model):
    model, teams = fitted_model
    groups = synthetic.make_group_draw(teams, seed=2)
    res = TournamentSimulator(model, groups).run(n_sims=5000, seed=1)
    p = res.reach_prob["champion"][groups["A"][0]]
    assert res.standard_error(p) == pytest.approx(np.sqrt(p * (1 - p) / res.n_sims))
    # MC SE shrinks with more sims (sampling noise), unlike parameter SE.
    assert res.standard_error(0.2) > 0.0
