"""Free gap-closers — #1 availability, #2 odds movement, #3 rotation, #4 conditional calibration.

All network-free: the availability/odds tests hit the pure parsers and a tmp snapshot dir; the rotation
test drives the simulator's group method directly on synthetic ratings.
"""

from __future__ import annotations

import numpy as np
import pytest

from wc2026.collective import availability as av
from wc2026.collective import odds_movement as om
from wc2026.evaluate import conditional_calibration as cc
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel, DixonColesParams
from wc2026.simulate.tournament import TournamentSimulator


# --------------------------------------------------------------- #4 conditional calibration
def _rows(n=600, seed=0):
    """Two synthetic 'tournament' rows of well-calibrated 3-class forecasts + total goals."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        probs = rng.dirichlet([2.0, 1.5, 2.0], size=n // 2)
        outcomes = np.array([rng.choice(3, p=row) for row in probs])  # outcomes drawn from probs
        et = rng.uniform(1.8, 3.4, n // 2)
        at = rng.poisson(et).astype(float)  # actual totals ~ Poisson(expected) → unbiased
        out.append({"probs": probs, "outcomes": outcomes, "exp_total": et, "act_total": at})
    return out


def test_conditional_reliability_structure_and_calibration():
    cr = cc.conditional_reliability(_rows())
    assert set(cr) == {"overall", "by_confidence", "by_predicted_class"}
    o = cr["overall"]
    assert o["n"] == 600
    # Forecasts drawn from their own probs are calibrated → small confidence gap.
    assert abs(o["conf_gap"]) < 0.06
    assert 0.0 <= o["ece"] <= 1.0
    for d in cr["by_predicted_class"].values():
        assert 0.0 <= d["mean_conf"] <= 1.0


def test_goals_calibration_is_unbiased_on_poisson_data():
    gc = cc.goals_calibration(_rows())
    assert gc["n"] == 600
    assert abs(gc["bias"]) < 0.25  # Poisson(actual|expected) ⇒ expected ≈ actual on average
    assert gc["by_pred_total"]  # at least one populated bucket


def test_goals_recal_preserves_supremacy_and_recalibrates_total():
    dc = DixonColesModel()
    dc.params = DixonColesParams(
        mu0=0.1, home=0.0, rho=-0.05,
        attack={"A": 0.4, "B": -0.2}, defense={"A": 0.2, "B": -0.1},
    )
    base = MatchModel(dc)
    recal = MatchModel(dc, goals_recal=(1.66, 0.36))
    lam0, mu0 = base.rates("A", "B")
    lam1, mu1 = recal.rates("A", "B")
    # Supremacy (lam - mu) is preserved; total is mapped by the affine recalibration.
    assert (lam1 - mu1) == pytest.approx(lam0 - mu0, abs=1e-9)
    assert (lam1 + mu1) == pytest.approx(1.66 + 0.36 * (lam0 + mu0), abs=1e-9)
    # W/D/L stays a valid distribution after recalibration.
    assert sum(recal.wdl("A", "B")) == pytest.approx(1.0)


# --------------------------------------------------------------- #2 odds movement
def test_parse_and_movement(tmp_path):
    (tmp_path / "odds_20260601.csv").write_text("# hdr\nteam,odds\nSpain,6.00\nBrazil,10.00\n")
    (tmp_path / "odds_20260608.csv").write_text("# hdr\nteam,odds\nSpain,5.00\nBrazil,12.00\n")
    hist = om.load_odds_history(tmp_path)
    assert set(hist) == {"20260601", "20260608"}
    assert hist["20260601"]["Spain"] == 6.0
    mv = om.movement_features(hist, window=7)
    # Spain shortened (6.0→5.0 ⇒ implied up ⇒ delta>0); Brazil drifted (10→12 ⇒ delta<0).
    assert mv["Spain"]["delta"] > 0
    assert mv["Brazil"]["delta"] < 0
    assert mv["Spain"]["n_snapshots"] == 2


def test_movement_empty_until_two_snapshots(tmp_path):
    (tmp_path / "odds_20260601.csv").write_text("team,odds\nSpain,6.00\n")
    assert om.movement_features(om.load_odds_history(tmp_path)) == {}


# --------------------------------------------------------------- #1 availability (pure parsers)
def test_injury_headline_detection():
    assert av.is_injury_headline("Pedri ruled out of World Cup with hamstring injury")
    assert av.is_injury_headline("Neuer suspended after red card")
    assert not av.is_injury_headline("Pedri scores stunning winner in friendly")
    assert not av.is_injury_headline("Spain confident ahead of opener")


def test_players_in_text_and_scan():
    players = ["Pedri González", "Lamine Yamal", "Rodri"]
    assert av.players_in_text("Pedri ruled out with injury", players) == ["Pedri González"]
    flags = av.scan_headlines(
        ["Yamal injury doubt for opener", "Rodri scores", "weather update"], players
    )
    assert flags == [("Lamine Yamal", "Yamal injury doubt for opener")]


def test_parse_replacements():
    wt = "Foo [[Gavi]] withdrew through injury and was replaced by [[Aleix García]]. [[Ter Stegen]] ruled out."
    reps = av.parse_replacements(wt)
    assert "Gavi" in reps and "Ter Stegen" in reps
    assert reps == list(dict.fromkeys(reps))  # deduped, order preserved


# --------------------------------------------------------------- #3 dead-rubber rotation
def _sim_with_groups(seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(48)]
    strength = {t: float(rng.normal(0, 0.25)) for t in teams}
    dc = DixonColesModel()
    dc.params = DixonColesParams(
        mu0=0.2, home=0.0, rho=-0.05, attack=dict(strength),
        defense={t: -v for t, v in strength.items()},
    )
    groups = {chr(ord("A") + g): teams[g * 4:(g + 1) * 4] for g in range(12)}
    return MatchModel(dc), groups


def test_rotation_downweights_a_clinched_team():
    model, groups = _sim_with_groups()
    g = groups["A"]
    t0, t1, t2, t3 = g
    # Lock matchdays 1-2 so t0 has 6 pts (clinched) going into its matchday-3 game vs t3.
    known = {
        frozenset((t0, t1)): (t0, 1, 0),
        frozenset((t2, t3)): (t2, 0, 0),
        frozenset((t0, t2)): (t0, 1, 0),
        frozenset((t1, t3)): (t1, 0, 0),
    }
    base = TournamentSimulator([model], groups, known_group_results=known, rotation_penalty=0.0)
    rot = TournamentSimulator([model], groups, known_group_results=known, rotation_penalty=1.0)

    def mean_t0_gf(sim):
        rng = np.random.default_rng(7)
        tot = 0.0
        N = 600
        for _ in range(N):
            recs = {r.team: r for r in sim._simulate_group(model, g, rng)}
            tot += recs[t0].gf
        return tot / N

    # t0's only simulated goals are its matchday-3 game; rotation should lower them.
    assert mean_t0_gf(rot) < mean_t0_gf(base)


def test_rotation_keeps_probabilities_valid():
    model, groups = _sim_with_groups()
    res = TournamentSimulator([model], groups, rotation_penalty=0.5).run(n_sims=2000, seed=1)
    champ = res.reach_prob["champion"]
    assert all(0.0 <= p <= 1.0 for p in champ.values())
    assert sum(champ.values()) == pytest.approx(1.0, abs=1e-6)
