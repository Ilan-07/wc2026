"""Conditional-calibration probing (Gap 4, free) — *where* is the model mis-calibrated?

Match-level ECE (~0.03) and stage reliability are pooled averages; they hide *which slices* the model
gets wrong. This slices the out-of-sample predictions (the per-match probs the 9-tournament backtest
already returns) by **confidence tier** and **predicted class**, and checks calibration of the top-pick
confidence in each slice (mean predicted top-prob vs how often the top pick actually wins — a positive
gap = overconfident). It also runs a **posterior-predictive goals check**: does the model's expected
total goals match reality, overall and across predicted-total buckets? These are the free diagnostics
that say where to spend effort, on the numbers the model uniquely produces.

Reuses ``tournament_backtest.run()`` (no extra fitting beyond that backtest). Run via ``probe.py``.
"""

from __future__ import annotations

import numpy as np

from .metrics import expected_calibration_error
from .tournament_backtest import run


def _slice_stats(p: np.ndarray, o: np.ndarray) -> dict:
    """Top-pick calibration for a slice of (probs, outcomes)."""
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    hit = (pred == o).astype(float)
    return {
        "n": int(len(o)),
        "mean_conf": float(conf.mean()),       # average stated confidence in the top pick
        "top_hit_rate": float(hit.mean()),     # how often the top pick actually won
        "conf_gap": float(conf.mean() - hit.mean()),  # +ve = overconfident, -ve = underconfident
        "ece": float(expected_calibration_error(p, o)),
    }


def conditional_reliability(rows: list[dict] | None = None) -> dict:
    """Top-pick calibration overall and sliced by confidence tier and predicted class."""
    rows = rows if rows is not None else run()["rows"]
    p = np.concatenate([r["probs"] for r in rows])
    o = np.concatenate([r["outcomes"] for r in rows])

    out = {"overall": _slice_stats(p, o), "by_confidence": {}, "by_predicted_class": {}}
    conf = p.max(axis=1)
    for name, lo, hi in [("toss-up [.33,.45)", 0.33, 0.45),
                         ("lean [.45,.60)", 0.45, 0.60),
                         ("strong [.60,1]", 0.60, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.any():
            out["by_confidence"][name] = _slice_stats(p[m], o[m])
    for c, nm in enumerate(["home", "draw", "away"]):
        m = p.argmax(axis=1) == c
        if m.any():
            out["by_predicted_class"][nm] = _slice_stats(p[m], o[m])
    return out


def goals_calibration(rows: list[dict] | None = None) -> dict:
    """Posterior-predictive check on total goals: model expectation vs reality, overall + bucketed."""
    rows = rows if rows is not None else run()["rows"]
    et = np.concatenate([r["exp_total"] for r in rows])
    at = np.concatenate([r["act_total"] for r in rows]).astype(float)
    out = {
        "n": int(len(at)),
        "mean_pred_total": float(et.mean()),
        "mean_act_total": float(at.mean()),
        "bias": float(et.mean() - at.mean()),  # +ve = model over-predicts goals
        "by_pred_total": {},
    }
    for lo, hi in [(0.0, 2.3), (2.3, 2.8), (2.8, 3.3), (3.3, 99.0)]:
        m = (et >= lo) & (et < hi)
        if m.any():
            out["by_pred_total"][f"[{lo:.1f},{hi:.1f})"] = {
                "n": int(m.sum()), "pred": float(et[m].mean()), "act": float(at[m].mean()),
                "bias": float(et[m].mean() - at[m].mean()),
            }
    return out
