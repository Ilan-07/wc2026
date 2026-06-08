"""Lane 1 tests — #1 dynamic state-space rating, #2 Bayesian tau + pooled-home.

The state-space tests are pure/fast (synthetic filtered series). The Bayesian tau+pooled-home test
runs a tiny MCMC and is skipped if PyMC is unavailable, mirroring the existing bayesian-DC test.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from wc2026.ratings.state_space import StateSpaceRating, _norm_cdf


# --------------------------------------------------------------------- #1 state-space
def _series(strengths: dict[str, float], n: int, seed: int, step_days: int = 3) -> list[dict]:
    """A synthetic chronological history where goal diff ~ Normal(true supremacy, 1)."""
    rng = np.random.default_rng(seed)
    teams = list(strengths)
    base = dt.date(2023, 1, 1)
    ms = []
    for k in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        gd = int(round(rng.normal(strengths[h] - strengths[a], 1.0)))
        ms.append(
            dict(
                home_team=str(h),
                away_team=str(a),
                home_score=max(0, gd),
                away_score=max(0, -gd),
                date=base + dt.timedelta(days=k * step_days),
                neutral=True,
            )
        )
    return ms


def test_norm_cdf_matches_known_values():
    assert _norm_cdf(0.0) == pytest.approx(0.5)
    assert _norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert _norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_wdl_is_a_valid_distribution_and_antisymmetric():
    p = StateSpaceRating().fit(_series({"A": 1.0, "B": 0.0, "C": -1.0}, 300, seed=1))
    home, draw, away = p.wdl("A", "C")
    assert all(0.0 <= x <= 1.0 for x in (home, draw, away))
    assert home + draw + away == pytest.approx(1.0)
    # Swapping venue mirrors home/away and leaves the draw mass unchanged (neutral game).
    rev = p.wdl("C", "A")
    assert rev[0] == pytest.approx(away)
    assert rev[2] == pytest.approx(home)
    assert rev[1] == pytest.approx(draw)


def test_filter_recovers_strength_ordering():
    p = StateSpaceRating().fit(_series({"A": 1.2, "B": 0.0, "C": -1.2}, 400, seed=3))
    assert p.strength["A"] > p.strength["B"] > p.strength["C"]
    # Centred near the true values (the filter is consistent, not just monotone).
    assert p.strength["A"] == pytest.approx(1.2, abs=0.4)
    assert p.strength["C"] == pytest.approx(-1.2, abs=0.4)


def test_idle_variance_grows_with_time():
    # Variance of an idle team should inflate the further past its last match we predict.
    ms = _series({"A": 2.0, "B": 0.0}, 80, seed=5)
    ss = StateSpaceRating(sigma_proc=0.02)
    p = ss.fit(ms)
    last = p.last_played["A"]
    assert last is not None
    near = p._inflated_var("A", as_of=last + dt.timedelta(days=1))
    far = p._inflated_var("A", as_of=last + dt.timedelta(days=400))
    assert far > near >= p.variance["A"]
    # A clear favourite's win prob regresses toward 0.5 as idle uncertainty widens the band.
    home_near, _, _ = p.wdl("A", "B", as_of=last + dt.timedelta(days=1))
    home_far, _, _ = p.wdl("A", "B", as_of=last + dt.timedelta(days=400))
    assert home_near > 0.5
    assert home_far < home_near


def test_home_advantage_shifts_supremacy():
    p = StateSpaceRating(home=0.4).fit(_series({"A": 0.0, "B": 0.0}, 200, seed=7))
    m_neutral, _ = p.supremacy("A", "B", neutral=True)
    m_home, _ = p.supremacy("A", "B", neutral=False)
    assert m_home - m_neutral == pytest.approx(0.4)


def test_init_attack_maps_strength_to_dc_scale():
    ss = StateSpaceRating()
    ss.fit(_series({"A": 1.0, "B": -1.0}, 200, seed=9))
    init = ss.init_attack(scale=2.0)
    assert init["A"] == pytest.approx(ss.params.strength["A"] / 2.0)
    assert init["A"] > init["B"]


def test_unseen_team_falls_back_to_prior():
    p = StateSpaceRating(init_var=0.8).fit(_series({"A": 0.0, "B": 0.0}, 100, seed=11))
    # An unknown opponent uses prior mean 0 and prior variance, still a valid forecast.
    home, draw, away = p.wdl("A", "Atlantis")
    assert all(0.0 <= x <= 1.0 for x in (home, draw, away))
    assert home + draw + away == pytest.approx(1.0)
    assert p._inflated_var("Atlantis") == pytest.approx(0.8)


# --------------------------------------------------------- #2 Bayesian tau + pooled home
def _poisson_history(strengths, n, seed, home_adv=0.25):
    """Synthetic DC-ish history: half the games non-neutral so a home effect is learnable."""
    rng = np.random.default_rng(seed)
    teams = list(strengths)
    base = dt.date(2023, 1, 1)
    ms = []
    for k in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        non_neutral = k % 2 == 1
        lam = np.exp(0.1 + strengths[h] - strengths[a] + (home_adv if non_neutral else 0.0))
        mu = np.exp(0.1 + strengths[a] - strengths[h])
        ms.append(
            dict(
                home_team=str(h),
                away_team=str(a),
                home_score=int(rng.poisson(lam)),
                away_score=int(rng.poisson(mu)),
                neutral=not non_neutral,
                date=base + dt.timedelta(days=k),
            )
        )
    return ms


def test_bayesian_tau_home_fits_and_recovers():
    pytest.importorskip("pymc")
    from wc2026.ratings.bayesian_dc import BayesianDixonColesTauHome
    from wc2026.ratings.dixon_coles import DixonColesParams

    strengths = {"A": 0.5, "B": 0.2, "C": -0.2, "D": -0.5}
    ms = _poisson_history(strengths, 400, seed=0)
    b = BayesianDixonColesTauHome(draws=200, tune=300, chains=2, seed=0)
    p = b.fit(ms)

    assert isinstance(p, DixonColesParams)
    # Learned rho is finite and exercises the DC tau path (non-zero, unlike the Poisson model).
    assert np.isfinite(p.rho)
    # Attack ratings recover the true ordering.
    assert p.attack["A"] > p.attack["B"] > p.attack["C"] > p.attack["D"]
    # Sum-to-zero identifiability holds for attack and defense.
    assert sum(p.attack.values()) == pytest.approx(0.0, abs=1e-6)
    assert sum(p.defense.values()) == pytest.approx(0.0, abs=1e-6)

    diag = b.diagnostics()
    # A handful of divergences can appear on this deliberately tiny run; just guard it's not pervasive.
    assert diag["divergences"] <= 10
    assert {"rho_mean", "rho_sd", "home_mu_mean", "home_sigma_mean", "forecast_max_rhat"} <= set(diag)

    home = b.team_home_advantage()
    assert set(home) == set(strengths)
    # Pooled per-team home effects average to the global level (ZeroSumNormal offsets).
    assert np.mean(list(home.values())) == pytest.approx(diag["home_mu_mean"], abs=1e-6)


def test_bayesian_tau_home_rho_path_changes_wdl():
    """A non-zero learned rho must actually flow into the W/D/L via the DC tau correction."""
    pytest.importorskip("pymc")
    from wc2026.model.match_model import MatchModel
    from wc2026.ratings.bayesian_dc import BayesianDixonColesTauHome
    from wc2026.ratings.dixon_coles import DixonColesModel

    ms = _poisson_history({"A": 0.4, "B": 0.0, "C": -0.4}, 300, seed=2)
    p = BayesianDixonColesTauHome(draws=120, tune=120, chains=2, seed=1).fit(ms)
    dc = DixonColesModel(); dc.params = p
    with_rho = MatchModel(dc).wdl("A", "C", neutral=True)
    # Same params but rho forced to 0 → draw mass shifts, proving rho is wired through.
    p.rho = 0.0
    no_rho = MatchModel(dc).wdl("A", "C", neutral=True)
    assert with_rho[1] != pytest.approx(no_rho[1])
