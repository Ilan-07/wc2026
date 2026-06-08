"""Lane 2 tests — #3 ET-from-DC-grid + shootout win-propensity, #4 xG measurement-error joint fit."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from wc2026.model.match_model import MatchModel
from wc2026.model.shootout import ShootoutModel, _sigmoid, running_psi_features
from wc2026.ratings.dixon_coles import DixonColesModel, DixonColesParams
from wc2026.ratings.xg_rating import JointXGParams, JointXGRating


def _equal_model(rho: float = -0.05, home: float = 0.0) -> MatchModel:
    """Two equal-strength teams A, B so any winner bias must come from the shootout model."""
    dc = DixonColesModel()
    dc.params = DixonColesParams(
        mu0=0.0, home=home, rho=rho, attack={"A": 0.0, "B": 0.0}, defense={"A": 0.0, "B": 0.0}
    )
    return MatchModel(dc)


# ------------------------------------------------------- #3 ET from the DC grid
def test_sample_at_rates_is_calibrated_to_the_rate():
    mm = _equal_model()
    rng = np.random.default_rng(0)
    draws = np.array([mm._sample_at_rates(1.6, 0.9, rng) for _ in range(20000)])
    assert draws[:, 0].mean() == pytest.approx(1.6, abs=0.08)  # home goals track lambda
    assert draws[:, 1].mean() == pytest.approx(0.9, abs=0.08)  # away goals track mu
    assert draws.min() >= 0


def test_sample_at_rates_keeps_low_score_dependence():
    # With rho<0 the DC grid puts MORE mass on 0-0 than independent Poissons would (tau(0,0)=1-lam*mu*rho>1).
    mm = _equal_model(rho=-0.1)
    rng = np.random.default_rng(1)
    draws = np.array([mm._sample_at_rates(0.7, 0.7, rng) for _ in range(40000)])
    p00_dc = float(np.mean((draws[:, 0] == 0) & (draws[:, 1] == 0)))
    p00_indep = float(np.exp(-0.7) ** 2)  # independent-Poisson P(0)·P(0)
    assert p00_dc > p00_indep  # the correlation the old independent-Poisson ET draw discarded


def test_sample_knockout_always_returns_a_competitor():
    mm = _equal_model()
    rng = np.random.default_rng(2)
    outs = {mm.sample_knockout("A", "B", rng) for _ in range(200)}
    assert outs <= {"A", "B"}


def test_shootout_model_shifts_knockout_winner():
    mm = _equal_model()

    class _Stub:
        def __init__(self, p):
            self.p = p

        def win_prob(self, home, away, psi):
            return self.p

    rng = np.random.default_rng(3)
    n = 6000
    a_with_win = np.mean([mm.sample_knockout("A", "B", rng, shootout_model=_Stub(1.0)) == "A"
                          for _ in range(n)])
    a_with_loss = np.mean([mm.sample_knockout("A", "B", rng, shootout_model=_Stub(0.0)) == "A"
                           for _ in range(n)])
    # Equal teams: only the shootouts differ, so always-win pushes A above 0.5 and always-lose below.
    assert a_with_win > 0.5 > a_with_loss


# ------------------------------------------------------- #3 shootout win-propensity model
def _shootouts(seq: list[tuple[str, str, str]], start=dt.date(2000, 1, 1)) -> list[dict]:
    return [
        {"date": str(start + dt.timedelta(days=i)), "home": h, "away": a, "winner": w}
        for i, (h, a, w) in enumerate(seq)
    ]


def test_running_psi_features_are_leakage_free():
    # First-ever shootout has no prior history → skipped (no feature can be built from zero games).
    recs = _shootouts([("A", "B", "A"), ("A", "C", "A"), ("B", "C", "B")])
    dpsi, y, skipped = running_psi_features(recs)
    assert skipped == 1                 # the very first game is cold-start
    assert len(dpsi) == len(y) == 2
    assert np.all(np.isfinite(dpsi))


def test_shootout_model_fit_and_monotone_win_prob():
    # A wins every shootout, Z loses every shootout → A should be favoured over Z.
    rng = np.random.default_rng(0)
    seq = []
    teams = ["A", "B", "C", "D", "Z"]
    for _ in range(120):
        h, a = rng.choice(teams, 2, replace=False)
        # A always wins; Z always loses; others coin flip
        if "A" in (h, a):
            w = "A"
        elif "Z" in (h, a):
            w = a if h == "Z" else h
        else:
            w = h if rng.random() < 0.5 else a
        seq.append((str(h), str(a), str(w)))
    params = ShootoutModel().fit(_shootouts(seq))
    assert np.isfinite(params.psi_scale) and np.isfinite(params.intercept)
    assert params.psi_scale > 0  # higher psi (better record) ⇒ higher win prob

    sm = ShootoutModel(); sm.fit(_shootouts(seq))
    psi = {"A": 0.9, "Z": 0.1}
    assert sm.win_prob("A", "Z", psi) > 0.5
    assert sm.win_prob("Z", "A", psi) < 0.5


def test_shootout_model_falls_back_without_data():
    params = ShootoutModel().fit([])  # no records
    assert params.psi_scale == pytest.approx(0.4)  # hand-set fallback
    sm = ShootoutModel()  # unfit
    assert sm.win_prob("A", "B", None) == pytest.approx(0.5)  # coin flip
    assert _sigmoid(0.0) == pytest.approx(0.5)


# ------------------------------------------------------- #4 xG measurement-error joint fit
def _xg_records(strengths, n, seed, goal_noise=True):
    """Matches where xG is a clean measurement of exp(eta) but goals are noisy Poisson draws."""
    rng = np.random.default_rng(seed)
    teams = list(strengths)
    recs = []
    for _ in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        eta_h = 0.1 + strengths[h] - strengths[a]
        eta_a = 0.1 + strengths[a] - strengths[h]
        lam, mu = np.exp(eta_h), np.exp(eta_a)
        recs.append({
            "home": str(h), "away": str(a),
            "home_goals": int(rng.poisson(lam)) if goal_noise else lam,
            "away_goals": int(rng.poisson(mu)) if goal_noise else mu,
            # xG = the true rate times mild lognormal measurement noise (lower variance than goals)
            "home_xg": float(lam * np.exp(rng.normal(0, 0.1))),
            "away_xg": float(mu * np.exp(rng.normal(0, 0.1))),
        })
    return recs


def test_joint_xg_weight_zero_is_goals_only():
    # With xg_weight=0 the xG channel is off → a pure goals Poisson MLE (the nested baseline).
    recs = _xg_records({"A": 0.6, "B": 0.0, "C": -0.6}, 300, seed=0)
    p = JointXGRating(xg_weight=0.0).fit(recs)
    assert isinstance(p, JointXGParams)
    assert sum(p.attack.values()) == pytest.approx(0.0, abs=1e-6)
    assert sum(p.defense.values()) == pytest.approx(0.0, abs=1e-6)
    assert p.attack["A"] > p.attack["B"] > p.attack["C"]


def test_joint_fit_recovers_strength_ordering_and_rates():
    recs = _xg_records({"A": 0.8, "B": 0.2, "C": -0.4, "D": -0.6}, 400, seed=1)
    p = JointXGRating(xg_weight=2.0).fit(recs)
    order = sorted(["A", "B", "C", "D"], key=lambda t: p.attack[t], reverse=True)
    assert order == ["A", "B", "C", "D"]
    lam, mu = p.rates("A", "D")
    assert lam > mu > 0          # strong home outscores weak away
    assert p.rates("A", "Atlantis") is None  # unrated opponent → no rate


def test_joint_xg_channel_sharpens_on_noisy_goals():
    # Few games/team: goals are noise-dominated, so adding the clean xG channel should track the
    # true strength gap more tightly than goals-only (the whole point of the measurement-error fit).
    strengths = {"A": 0.9, "B": 0.3, "C": -0.3, "D": -0.9}
    recs = _xg_records(strengths, 120, seed=7)
    goals_only = JointXGRating(xg_weight=0.0).fit(recs)
    joint = JointXGRating(xg_weight=3.0).fit(recs)

    def err(p):
        # compare attack spread (A minus D) to the true gap, both centred (sum-zero)
        true_gap = strengths["A"] - strengths["D"]
        return abs((p.attack["A"] - p.attack["D"]) - true_gap)

    assert err(joint) < err(goals_only)
