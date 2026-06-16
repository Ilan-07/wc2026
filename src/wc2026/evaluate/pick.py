"""Draw-aware match pick + decisive-match accuracy — an honest "correct calls" hit-rate.

The naive "correct calls" number is the argmax of the W/D/L probabilities. But a draw is almost
never any side's single most-likely outcome (it tops out near 0.35), so argmax effectively *never*
predicts a draw — while ~26% of tournament matches actually are draws. Counting those structurally
unpickable matches as misses makes the hit-rate look far worse than the forecast is (WC2026 group
stage: 8 of the first 16 matches were draws → a 6/16 headline that is really 6/8 on decisive games).

Two honest tools live here:

* ``decisive_accuracy`` — the hit-rate on matches that actually had a winner (draws excluded). This
  is the number worth quoting: it grades the pick on the matches the pick can possibly get right.
* ``pick`` — an optional draw-aware pick that calls a draw when ``p_draw`` clears a calibrated
  threshold. Calibration (``draw_pick_calibration.py``, 9-tournament backtest) shows it is
  **accuracy-neutral at best out-of-sample** (leave-one-tournament-out −0.010): the threshold lands
  so high it rarely fires, because the model's calibrated draw mass seldom beats a decisive side.
  Shipped as a documented, conservative option (threshold 0.335) — *not* because it beats argmax.
  See ``FINDINGS.md``.
"""

from __future__ import annotations

import numpy as np

# Pooled accuracy-maximizing draw threshold on the 9-tournament backtest (399 matches, accuracy
# 0.5414 vs argmax 0.5388). At/above this value the draw-aware pick stays near plain argmax on
# history; lowering it trades decisive-match hits for draw hits at a net loss (LOTO −0.010).
# Reproduce with ``python3 draw_pick_calibration.py``.
DRAW_PICK_THRESHOLD = 0.335


def pick(probs, draw_threshold: float = DRAW_PICK_THRESHOLD) -> int:
    """Predicted outcome index (0 home / 1 draw / 2 away) for one match.

    Calls a draw when ``p_draw`` clears ``draw_threshold``, otherwise the more likely decisive side.
    With the default threshold this matches argmax on virtually every fixture (see module docstring).
    """
    p = np.asarray(probs, dtype=float)
    if p[1] >= draw_threshold:
        return 1
    return 0 if p[0] >= p[2] else 2


def decisive_accuracy(probs, outcomes) -> tuple[int, int]:
    """``(correct, n_decisive)`` — hit-rate on matches that were *not* draws.

    Draws are excluded because they are structurally unpickable by a single most-likely outcome;
    this grades the home/away call on the matches it can actually get right. The honest hit-rate.
    """
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=int)
    mask = o != 1
    if not mask.any():
        return 0, 0
    pred = np.where(p[mask, 0] >= p[mask, 2], 0, 2)
    return int((pred == o[mask]).sum()), int(mask.sum())


def _accuracy(P: np.ndarray, O: np.ndarray, threshold: float) -> float:
    """Draw-aware pick accuracy over an (n,3) prob matrix and (n,) outcomes (threshold≥1 ⇒ argmax)."""
    pred = np.where(P[:, 1] >= threshold, 1, np.where(P[:, 0] >= P[:, 2], 0, 2))
    return float((pred == O).mean())


def calibrate(rows: list[dict], grid: np.ndarray | None = None) -> dict:
    """Calibrate the draw threshold on backtest ``rows`` (each with ``probs``/``outcomes``).

    Returns the pooled accuracy-maximizing threshold plus a leave-one-tournament-out (LOTO) honest
    comparison of draw-aware vs plain-argmax accuracy — the test that decides whether the draw pick
    earns its place. ``rows`` is the ``run()["rows"]`` from :mod:`wc2026.evaluate.tournament_backtest`.
    """
    grid = np.round(np.arange(0.24, 0.42, 0.005), 3) if grid is None else grid
    P = np.vstack([r["probs"] for r in rows])
    O = np.concatenate([r["outcomes"] for r in rows])

    accs = {float(g): _accuracy(P, O, g) for g in grid}
    best_acc = max(accs.values())
    pooled_best = min(g for g, a in accs.items() if a == best_acc)

    loto_base = loto_draw = n = 0
    for i, r in enumerate(rows):
        others = [x for j, x in enumerate(rows) if j != i]
        Po = np.vstack([x["probs"] for x in others])
        Oo = np.concatenate([x["outcomes"] for x in others])
        thr = max(grid, key=lambda t: _accuracy(Po, Oo, t))
        Ph, Oh = r["probs"], r["outcomes"]
        loto_base += _accuracy(Ph, Oh, 2.0) * len(Oh)
        loto_draw += _accuracy(Ph, Oh, thr) * len(Oh)
        n += len(Oh)

    return {
        "n": n,
        "argmax_accuracy": _accuracy(P, O, 2.0),
        "pooled_best_threshold": pooled_best,
        "pooled_best_accuracy": best_acc,
        "loto_argmax_accuracy": loto_base / n,
        "loto_draw_aware_accuracy": loto_draw / n,
        "loto_gain": (loto_draw - loto_base) / n,
    }
