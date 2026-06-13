"""Tests for the per-match scoreline display layer (reports/scores.py).

These are pure unit tests on a small fitted model — no network, no full simulation. They verify the
displayed numbers are internally consistent (valid distributions, modal = grid argmax), that the
home/away orientation matches the simulator's host-as-home convention, that played fixtures are
locked to their real score, and that knockouts only appear once a bracket is supplied.
"""

from __future__ import annotations

import numpy as np
import pytest

from wc2026.data import synthetic
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.reports import scores


@pytest.fixture(scope="module")
def model_and_teams():
    teams = synthetic.make_teams(8, seed=0)
    history = synthetic.generate_history(teams, n_matches=2000, seed=1)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(history)
    return MatchModel(dc), list(teams)


def test_score_fixture_is_a_valid_distribution(model_and_teams):
    model, teams = model_and_teams
    fs = scores.score_fixture(model, teams[0], teams[1], hosts=set())
    assert not fs.played
    # W/D/L is a proper distribution
    assert fs.p_home + fs.p_draw + fs.p_away == pytest.approx(1.0, abs=1e-6)
    for p in (fs.p_home, fs.p_draw, fs.p_away, fs.modal_prob):
        assert 0.0 <= p <= 1.0
    # expected goals are positive and finite
    assert fs.home_score > 0 and fs.away_score > 0
    assert np.isfinite(fs.home_score) and np.isfinite(fs.away_score)


def test_modal_scoreline_is_the_grid_argmax(model_and_teams):
    model, teams = model_and_teams
    a, b = teams[0], teams[1]
    fs = scores.score_fixture(model, a, b, hosts=set())  # neutral (no hosts)
    grid = model.scoreline_grid(a, b, neutral=True)
    ij = int(np.argmax(grid))
    assert (fs.modal_home, fs.modal_away) == (ij // grid.shape[1], ij % grid.shape[1])
    assert fs.modal_prob == pytest.approx(float(grid.max()), abs=1e-9)


def test_host_orientation_matches_simulator(model_and_teams):
    """An away-listed host must be scored as the home side, exactly like TournamentSimulator._sample,
    so the displayed numbers are consistent with the forecast."""
    model, teams = model_and_teams
    a, b = teams[0], teams[1]
    fs = scores.score_fixture(model, a, b, hosts={b})  # b is the host but listed away
    # Simulator scores this as model.scoreline_grid(b, a, neutral=False); reproduce and compare.
    grid = model.scoreline_grid(b, a, neutral=False)  # rows = b goals, cols = a goals
    p_b = float(np.tril(grid, -1).sum())
    p_a = float(np.triu(grid, 1).sum())
    assert fs.p_home == pytest.approx(p_a, abs=1e-9)   # a is the listed home team
    assert fs.p_away == pytest.approx(p_b, abs=1e-9)
    # host (b) gets home advantage => should be expected to score more than on neutral ground
    eg_b_home, _ = model.rates(b, a, neutral=False)
    assert fs.away_score == pytest.approx(eg_b_home, abs=1e-9)


def test_played_group_fixture_is_locked_to_real_score(model_and_teams):
    model, teams = model_and_teams
    groups = {"A": teams[:4], "B": teams[4:8]}
    fixtures = [(None, teams[0], teams[1], "City")]
    played = {frozenset((teams[0], teams[1])): (teams[0], 3, 1)}  # real 3-1 to teams[0]
    out = scores.build_group_scores(model, fixtures, played, groups, hosts=set())
    fs = out[0]
    assert fs.played
    assert (fs.home_score, fs.away_score) == (3.0, 1.0)
    assert (fs.modal_home, fs.modal_away, fs.modal_prob) == (3, 1, 1.0)
    assert (fs.p_home, fs.p_draw, fs.p_away) == (1.0, 0.0, 0.0)  # home win realised
    assert fs.group == "A"


def test_played_fixture_orientation_when_real_home_is_listed_away(model_and_teams):
    model, teams = model_and_teams
    groups = {"A": teams[:4], "B": teams[4:8]}
    # fixture lists teams[1] at home, but the real match was played with teams[0] at home, 2-0
    fixtures = [(None, teams[1], teams[0], "City")]
    played = {frozenset((teams[0], teams[1])): (teams[0], 2, 0)}
    fs = scores.build_group_scores(model, fixtures, played, groups, hosts=set())[0]
    # listed home = teams[1], who lost 0-2
    assert (fs.home_score, fs.away_score) == (0.0, 2.0)
    assert (fs.p_home, fs.p_draw, fs.p_away) == (0.0, 0.0, 1.0)


def test_knockouts_empty_without_bracket(model_and_teams):
    model, _ = model_and_teams
    assert scores.build_knockout_scores(model, None) == []
    assert scores.build_knockout_scores(model, ["X", "Y"]) == []  # wrong length -> ignored


def test_knockouts_from_bracket_and_payload_shape(model_and_teams):
    model, teams = model_and_teams
    bracket = [teams[i % len(teams)] for i in range(32)]
    ko = scores.build_knockout_scores(model, bracket, hosts=set())
    assert len(ko) == 16  # 32 teams -> 16 ties
    assert all(fs.stage == "r32" and fs.round_name == "Round of 32" for fs in ko)

    fixtures = [(None, teams[0], teams[1], "C")]
    groups = {"A": teams[:4], "B": teams[4:8]}
    sec = scores.ScoreSections(
        groups=scores.build_group_scores(model, fixtures, {}, groups, hosts=set()),
        knockouts=ko,
    )
    payload = sec.payload()
    assert payload["groups"][0]["group"] == "A"
    assert payload["knockouts"][0]["round"] == "Round of 32"
    fx = payload["groups"][0]["fixtures"][0]
    assert {"home", "away", "played", "h", "a", "mh", "ma", "mp", "pH", "pD", "pA"} <= set(fx)


def test_console_formatter_runs_and_mentions_groups(model_and_teams):
    model, teams = model_and_teams
    fixtures = [(None, teams[0], teams[1], "C"), (None, teams[2], teams[3], "C")]
    groups = {"A": teams[:4], "B": teams[4:8]}
    sec = scores.build_score_sections(model, fixtures, {}, groups, hosts=set())
    text = scores.format_scores_console(sec)
    assert "Predicted match scores" in text
    assert "Group A" in text
    assert teams[0] in text
