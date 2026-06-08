"""Crowd-vs-Model divergence (plan Tier C output).

The most actionable product of fusion is not the blended number but the **disagreement** between
what the model believes and what the market/crowd believes. Large positive divergence = the crowd
rates a team more highly than the fundamentals do (possible overhype); large negative = the model
likes a team the crowd is sleeping on (possible value). This is both a headline deliverable and a
feature a downstream meta-learner can use.
"""

from __future__ import annotations

import numpy as np

_LABELS = ("home", "draw", "away")


def divergence(model_p: np.ndarray, market_p: np.ndarray) -> np.ndarray:
    """Per-outcome divergence = market probability − model probability (same shape as inputs)."""
    return np.asarray(market_p, dtype=float) - np.asarray(model_p, dtype=float)


def flag_match(model_p, market_p, threshold: float = 0.10) -> dict:
    """Summarise one match's divergence, flagging outcomes where crowd and model disagree a lot."""
    d = divergence(np.atleast_2d(model_p), np.atleast_2d(market_p))[0]
    flags = []
    for i, lab in enumerate(_LABELS):
        if d[i] >= threshold:
            flags.append(f"crowd higher on {lab} (+{d[i]:.0%})")
        elif d[i] <= -threshold:
            flags.append(f"model higher on {lab} ({d[i]:.0%})")
    return {"divergence": d, "flags": flags}


def team_divergence(team_probs: dict[str, tuple[float, float]]) -> list[tuple[str, float]]:
    """Rank teams by (market_winprob − model_winprob); +ve = crowd overrates vs model.

    ``team_probs`` maps team -> (model_win_prob, market_win_prob), e.g. tournament-win odds.
    """
    rows = [(t, mk - md) for t, (md, mk) in team_probs.items()]
    return sorted(rows, key=lambda x: x[1], reverse=True)
