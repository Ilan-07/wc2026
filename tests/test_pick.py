"""Draw-aware pick + decisive-match accuracy (honest hit-rate reporting)."""

from __future__ import annotations

import numpy as np
import pytest

from wc2026.evaluate.pick import DRAW_PICK_THRESHOLD, calibrate, decisive_accuracy, pick


def test_pick_defaults_to_decisive_side():
    # Draw mass below the threshold → call the more likely decisive side, never the draw.
    assert pick([0.55, 0.30, 0.15]) == 0
    assert pick([0.15, 0.30, 0.55]) == 2
    assert DRAW_PICK_THRESHOLD > 0.30  # a real draw rarely clears it — the whole point


def test_pick_calls_draw_only_above_threshold():
    assert pick([0.30, 0.40, 0.30]) == 1               # p_draw clears default threshold
    assert pick([0.40, 0.20, 0.40], draw_threshold=0.30) == 0  # tie on decisive → home
    assert pick([0.10, 0.50, 0.40], draw_threshold=0.45) == 1


def test_decisive_accuracy_excludes_draws():
    probs = np.array([[0.6, 0.2, 0.2],   # home win, called right
                      [0.2, 0.2, 0.6],   # away win, called wrong (actual home)
                      [0.4, 0.4, 0.2]])  # actual draw → excluded entirely
    outcomes = np.array([0, 0, 1])
    correct, n = decisive_accuracy(probs, outcomes)
    assert (correct, n) == (1, 2)        # 2 decisive matches, 1 called right; the draw is dropped


def test_decisive_accuracy_all_draws():
    correct, n = decisive_accuracy(np.array([[0.4, 0.4, 0.2]]), np.array([1]))
    assert (correct, n) == (0, 0)


def test_calibrate_reports_loto_comparison():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(3):
        p = rng.dirichlet([3, 2, 3], size=40)
        o = np.array([rng.choice(3, p=row) for row in p])
        rows.append({"probs": p, "outcomes": o})
    res = calibrate(rows)
    assert {"argmax_accuracy", "pooled_best_threshold", "loto_gain"} <= res.keys()
    assert 0.0 <= res["argmax_accuracy"] <= 1.0
    assert res["loto_gain"] == pytest.approx(
        res["loto_draw_aware_accuracy"] - res["loto_argmax_accuracy"])
