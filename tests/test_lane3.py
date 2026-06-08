"""Lane 3 tests — #5 multi-tournament backtest, #6 stage reliability, #7 SBC, #8 learned blend weight.

The backtest/SBC tests fit real models and are kept to small subsets so the suite stays fast.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from wc2026.evaluate import stage_reliability as sr
from wc2026.evaluate import tournament_backtest as tb
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel, DixonColesParams


# --------------------------------------------------------------------- #5 multi-tournament backtest
def test_specs_are_wellformed():
    keys = [s.key for s in tb.SPECS]
    assert len(keys) == len(set(keys))           # unique ids
    assert {"wc2018", "wc2022"} <= set(keys)     # the two World Cups are present
    assert any(s.name == "UEFA Euro" for s in tb.SPECS)
    assert any(s.name == "Copa América" for s in tb.SPECS)


@pytest.fixture(scope="module")
def all_matches():
    from wc2026.data import loaders
    return loaders.load_results(since="2006-01-01", min_team_matches=15)


def test_backtest_one_wc2018_shows_skill(all_matches):
    spec = next(s for s in tb.SPECS if s.key == "wc2018")
    r = tb.backtest_one(spec, all_matches)
    assert r is not None
    assert r["n"] == 64
    # The fitted rating beats the uniform baseline out-of-sample.
    assert r["rps"] < r["uniform_rps"]
    assert 0.15 < r["rps"] < 0.25


def test_run_pools_multiple_tournaments(all_matches):
    # Two-spec subset keeps the test fast while exercising the pooling path.
    subset = [s for s in tb.SPECS if s.key in ("wc2022", "copa2024")]
    res = tb.run(specs=subset)
    assert res["n_tournaments"] == 2
    assert res["n_matches"] > 80
    assert res["pooled_rps"] < res["pooled_uniform_rps"]   # pooled skill
    assert res["skill_vs_uniform"] == pytest.approx(
        res["pooled_uniform_rps"] - res["pooled_rps"], abs=1e-9
    )
    assert np.isfinite(res["per_tournament_rps_std"])


# --------------------------------------------------------------------- #6 stage reliability
def _synthetic_8group_model(seed=0):
    """A DC model over 32 synthetic teams arranged in 8 groups of 4 (A..H)."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(32)]
    strength = {t: float(rng.normal(0, 0.3)) for t in teams}
    dc = DixonColesModel()
    dc.params = DixonColesParams(
        mu0=0.0, home=0.0, rho=-0.05,
        attack=dict(strength), defense={t: -v for t, v in strength.items()},
    )
    groups = {chr(ord("A") + g): teams[g * 4:(g + 1) * 4] for g in range(8)}
    return MatchModel(dc), groups


def test_actual_stages_2018_is_correct():
    a = sr.actual_stages(2018)
    assert a is not None
    assert a["champion"] == {"France"}
    assert "Croatia" in a["final"]          # 2018 finalists were France & Croatia
    assert len(a["qualify"]) == 16 and len(a["qf"]) == 8 and len(a["sf"]) == 4
    assert a["final"] <= a["sf"] <= a["qf"] <= a["qualify"]  # nested reach sets


def test_simulate_stages_obeys_reach_constraints():
    model, groups = _synthetic_8group_model()
    pred = sr.simulate_stages(model, groups, n_sims=1500, seed=1)
    teams = [t for ts in groups.values() for t in ts]
    # Slot-count identities: reach-probs sum to the number of slots at each stage.
    assert sum(pred["qualify"].values()) == pytest.approx(16, abs=0.5)
    assert sum(pred["champion"].values()) == pytest.approx(1.0, abs=1e-6)
    # Per-team monotonicity: deeper stages are never more likely than shallower ones.
    for t in teams:
        assert pred["qualify"][t] >= pred["qf"][t] >= pred["sf"][t] >= pred["final"][t] >= pred["champion"][t]
        assert 0.0 <= pred["champion"][t] <= 1.0


def test_stage_reliability_backtest_summarize():
    bt = sr.backtest(editions={2018: date(2018, 6, 14)}, n_sims=1500)
    s = sr.summarize(bt)
    assert s["editions"] == [2018]
    assert set(s["stages"]) == set(sr.STAGES)
    for d in s["stages"].values():
        assert d["n"] == 32                       # one 32-team edition
        assert 0.0 <= d["ece"] <= 1.0
        assert np.isfinite(d["brier"])
    # mean predicted equals the base rate per stage by the slot-count identity.
    assert s["stages"]["qualify"]["mean_pred"] == pytest.approx(0.5, abs=1e-6)


# --------------------------------------------------------------------- #7 SBC
def test_sbc_runs_and_ranks_are_valid():
    pytest.importorskip("pymc")
    from wc2026.evaluate.sbc import TRACKED, run_sbc

    res = run_sbc(n_sims=5, n_teams=4, n_matches=40, draws=80, tune=80, chains=2, seed=0)
    assert res["n_post"] == 160
    for k in TRACKED:
        ranks = res["ranks"][k]
        assert len(ranks) == 5
        assert np.all((ranks >= 0) & (ranks <= res["n_post"]))  # ranks live in [0, n_post]
        assert 0.0 <= res["uniformity"][k] <= 1.0


def test_rank_uniformity_pvalue_extremes():
    from wc2026.evaluate.sbc import rank_uniformity_pvalue

    rng = np.random.default_rng(0)
    n_post = 200
    # Genuinely uniform ranks → not flagged (high p).
    uniform = rng.integers(0, n_post + 1, 500)
    assert rank_uniformity_pvalue(uniform, n_post) > 0.01
    # All ranks piled at 0 (pathologically miscalibrated) → flagged (p ≈ 0).
    piled = np.zeros(500, dtype=int)
    assert rank_uniformity_pvalue(piled, n_post) < 1e-6


# --------------------------------------------------------------------- #8 learned blend weight
def _two_sources(n=600, seed=0):
    """A sharp, correct 'good' forecast and a uniform 'bad' one over n 3-class outcomes."""
    rng = np.random.default_rng(seed)
    outcomes = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.1)
    good[np.arange(n), outcomes] = 0.8           # concentrates mass on the truth
    bad = np.full((n, 3), 1 / 3)                  # uniform, uninformative
    return good, bad, outcomes


def test_fit_model_weight_picks_the_better_source():
    from wc2026.fusion.pool import fit_model_weight

    good, bad, outcomes = _two_sources()
    # If the *market* slot holds the good forecast, the model (bad) weight → 0.
    assert fit_model_weight(model_p=bad, market_p=good, outcomes=outcomes) < 0.05
    # If the *model* slot holds the good forecast, its weight → 1.
    assert fit_model_weight(model_p=good, market_p=bad, outcomes=outcomes) > 0.95


def test_fit_model_weight_logloss_and_prior_shrinkage():
    from wc2026.fusion.pool import fit_model_weight

    good, bad, outcomes = _two_sources()
    assert fit_model_weight(bad, good, outcomes, score="logloss") < 0.05
    # A strong ridge toward 0.5 pulls the estimate up off the RPS-optimal 0.
    pulled = fit_model_weight(bad, good, outcomes, prior_weight=0.5, prior_strength=5.0)
    assert pulled > 0.05


def test_cross_val_model_weight_is_stable_and_honest():
    from wc2026.fusion.pool import cross_val_model_weight

    good, bad, outcomes = _two_sources()
    res = cross_val_model_weight(model_p=bad, market_p=good, outcomes=outcomes, k=5)
    assert len(res["weights"]) == 5
    assert res["mean_weight"] < 0.05            # model adds nothing over the good market
    assert res["std_weight"] < 0.1              # and the weight is stable across folds
    # Fused never worse than the better (market) source out-of-sample.
    assert res["cv_fused"] <= res["cv_market"] + 1e-6
    assert res["cv_market"] < res["cv_model"]   # the good source genuinely scores better
