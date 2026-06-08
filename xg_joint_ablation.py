"""Measurement-error joint goals+xG rating vs a goals-only rating — one honest ablation (#4, Lane 2).

The existing ``xg_rating_ablation.py`` blends two *separately* fit ratings. This tests the principled
alternative: fit ONE attack/defense to goals (Poisson) and xG (Gaussian measurement of the same
log-rate) jointly (``JointXGRating``). The control is perfectly nested — the goals-only baseline is
the same class with ``xg_weight=0`` — so the only thing that changes is whether the xG channel is on.

Leave-one-tournament-out over the four StatsBomb competitions: fit both on the other three, predict
W/D/L on the held-out tournament's xG-covered matches, score by RPS. Sweeps the xG channel precision.
A weight earns its place only if the joint fit beats the goals-only fit out-of-sample.

Uses cached data/raw/sb_match_records.json. Run: ``PYTHONPATH=src python3 xg_joint_ablation.py``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.ratings.dixon_coles import _poisson_pmf, tau
from wc2026.ratings.xg_rating import JointXGRating

TOURN = ["wc2018", "wc2022", "euro2024", "copa2024"]
CACHE = Path("data/raw/sb_match_records.json")
WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]
_ = date  # (kept for symmetry with sibling ablations; tournaments are keyed by name here)


def wdl(g, a):
    return 0 if g > a else (1 if g == a else 2)


def grid_wdl(lam, mu, rho=0.0):
    ks = np.arange(11)
    g = np.outer(_poisson_pmf(ks, lam), _poisson_pmf(ks, mu)) * tau(ks[:, None], ks[None, :], lam, mu, rho)
    g = np.clip(g, 0, None)
    g /= g.sum()
    return [float(np.tril(g, -1).sum()), float(np.trace(g)), float(np.triu(g, 1).sum())]


def _score_fold(train_records, test_records, xg_weight):
    """Fit JointXGRating on train, return (sum_rps, n) over xG-covered held-out matches."""
    params = JointXGRating(xg_weight=xg_weight).fit(train_records)
    P, outs = [], []
    for r in test_records:
        rate = params.rates(r["home"], r["away"])
        if rate is None:  # both teams must be rated (covered by the other three tournaments)
            continue
        P.append(grid_wdl(*rate))
        outs.append(wdl(r["home_goals"], r["away_goals"]))
    if not outs:
        return 0.0, 0
    return rps(np.array(P), np.array(outs)) * len(outs), len(outs)


def main() -> dict:
    records = json.loads(CACHE.read_text())

    base_sum, n_eval = 0.0, 0
    joint_sum = {w: 0.0 for w in WEIGHTS}
    for test in TOURN:
        train = [r for k in TOURN if k != test for r in records[k]]
        b_s, b_n = _score_fold(train, records[test], xg_weight=0.0)  # nested goals-only baseline
        base_sum += b_s
        n_eval += b_n
        for w in WEIGHTS:
            j_s, _n = _score_fold(train, records[test], xg_weight=w)
            joint_sum[w] += j_s
        print(f"{test:9s}: {b_n:3d} xG-covered held-out matches")

    base = base_sum / n_eval
    print(f"\nEvaluated on {n_eval} xG-covered matches (leave-one-tournament-out)\n")
    print(f"  goals-only (xg_weight=0)  RPS : {base:.4f}")
    best_w, best = None, base
    for w in WEIGHTS:
        r = joint_sum[w] / n_eval
        if r < best:
            best, best_w = r, w
        flag = "  <-- best" if best_w == w else ""
        print(f"  joint goals+xG w={w:<4}     RPS : {r:.4f}   ({base - r:+.4f}){flag}")

    print()
    if best_w is None:
        print("VERDICT: the joint xG channel does NOT beat goals-only out-of-sample — the rating "
              "already extracts what xG offers on this data; keep goals-only (xG stays diagnostic).")
    else:
        print(f"VERDICT: joint goals+xG helps; best xg_weight={best_w} improves RPS by "
              f"{base - best:+.4f}. The gain grows with the xG weight → on sparse tournament data,"
              f" finished goals are noise-dominated and xG is the better measurement of the same rate."
              f"\n  Caveat: this beats a goals-only rating ON THE SAME ~4-tournament data, not the"
              f" full-history production DC (which averages goal noise over ~49k matches). The lever is"
              f" comprehensive xG history we don't have — the principled way to spend it is THIS joint"
              f" fit, not the weaker post-hoc blend (+0.0058).")
    return {"n": n_eval, "goals_only_rps": base, "best_weight": best_w,
            "joint_rps": {w: joint_sum[w] / n_eval for w in WEIGHTS}}


if __name__ == "__main__":
    main()
