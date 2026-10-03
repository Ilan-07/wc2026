"""Regression gate for the headline validation numbers in README.md / FINDINGS.md.

Each published RPS / Brier / ECE number is re-computed on the pinned, sha256-verified evaluation
snapshot (``wc2026.data.snapshot``) and must match what the docs print. If code changes move a number,
this fails and the docs must be updated in the same commit — no silent drift. The snapshot hash check
catches the other failure mode (upstream data edits).

Skips locally when the snapshot is not downloaded; CI sets ``WC2026_REQUIRE_SNAPSHOT=1`` so a missing
or modified snapshot is a failure there, never a skip. Runtime ~3 min (one 9-tournament backtest, one
4-WC stage simulation, five league DC fits).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from wc2026.data import snapshot

_problems = snapshot.verify()
if _problems and os.environ.get("WC2026_REQUIRE_SNAPSHOT"):
    raise RuntimeError("evaluation snapshot required but not intact: " + "; ".join(_problems))
pytestmark = pytest.mark.skipif(bool(_problems), reason="evaluation snapshot not downloaded")

# Tight enough to catch the +0.0392 -> +0.0389 drift seen when upstream edited history; loose enough
# for cross-platform floating-point differences in the MLE optimizer.
TOL = 2e-4


def _check(name: str, observed: dict, expected: dict) -> None:
    """Compare every observed value to its (expected, tol) and fail listing ALL of them, so one CI log
    shows the full picture on each platform. Observed values are printed (shown with ``pytest -rP``)."""
    print(f"[repro] {name}: {json.dumps(observed, sort_keys=True)}")
    bad = {k: (observed[k], want, tol) for k, (want, tol) in expected.items()
           if abs(observed[k] - want) > tol}
    assert not bad, f"{name} drifted (observed, expected, tol): {bad}; all observed: {observed}"


@pytest.fixture(scope="module", autouse=True)
def _pinned_data():
    from wc2026.data import loaders

    old = loaders.DEFAULT_RESULTS
    snapshot.use_snapshot()
    yield
    loaders.DEFAULT_RESULTS = old


@pytest.fixture(scope="module")
def backtest():
    from wc2026.evaluate.tournament_backtest import run

    return run()


def test_nine_tournament_backtest_matches_readme(backtest):
    # README: "RPS 0.195 vs 0.234 uniform → skill +0.039, positive on 8 of 9" (FINDINGS: +0.0392)
    assert backtest["n_tournaments"] == 9
    assert backtest["n_matches"] == 399
    keys = ("pooled_rps", "pooled_uniform_rps", "skill_vs_uniform", "per_tournament_rps_mean")
    _check("backtest", {k: backtest[k] for k in keys}, {
        "pooled_rps": (0.19514, TOL), "pooled_uniform_rps": (0.23434, TOL),
        "skill_vs_uniform": (0.03920, TOL), "per_tournament_rps_mean": (0.18795, TOL),
    })
    skill = {r["key"]: r["uniform_rps"] - r["rps"] for r in backtest["rows"]}
    assert sum(v > 0 for v in skill.values()) == 8
    assert skill["euro2016"] < 0  # the documented lone miss


def test_match_level_ece_matches_readme(backtest):
    # README: "Match-level calibration ECE ~0.03" — the `cli.py probe` overall ECE on the same 399 matches.
    from wc2026.evaluate.conditional_calibration import conditional_reliability

    o = conditional_reliability(backtest["rows"])["overall"]
    assert o["n"] == 399
    _check("match_ece", {"ece": o["ece"]}, {"ece": (0.0311, 1e-3)})


def test_stage_reliability_matches_readme():
    # README: "Pooled Brier 0.104, ECE 0.020" over the 2010-2022 World Cups.
    from wc2026.evaluate import stage_reliability as sr

    s = sr.summarize(sr.backtest())
    assert s["editions"] == [2010, 2014, 2018, 2022]
    assert s["pooled"]["n"] == 640
    observed = {"brier": s["pooled"]["brier"], "ece": s["pooled"]["ece"],
                **{f"ece_{k}": d["ece"] for k, d in s["stages"].items()}}
    _check("stage_reliability", observed, {"brier": (0.1044, TOL), "ece": (0.0197, 1e-3)})


def test_model_vs_market_matches_readme():
    # README: "Market wins (RPS 0.190 vs 0.204)"; FINDINGS: CV blend weight 0.00 ± 0.00.
    import fusion_validate
    from wc2026.evaluate.metrics import ranked_probability_score as rps
    from wc2026.fusion.pool import cross_val_model_weight

    model_p, market_p, outcomes = fusion_validate.collect()
    n = len(outcomes)
    ev = np.random.default_rng(0).permutation(n)[n // 2:]  # same held-out half as fusion_validate.main
    assert len(ev) == 787
    cv = cross_val_model_weight(model_p, market_p, outcomes, k=5, score="rps")
    _check("model_vs_market", {
        "model_rps": rps(model_p[ev], outcomes[ev]), "market_rps": rps(market_p[ev], outcomes[ev]),
        "cv_weight": cv["mean_weight"],
    }, {"model_rps": (0.2043, TOL), "market_rps": (0.1897, TOL), "cv_weight": (0.0, 1e-3)})
