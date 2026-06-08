"""Probability fusion via logarithmic opinion pooling (plan Tier C).

Given several independent-ish probability forecasts for the same match (the structural model,
the de-vigged market, later sentiment), combine them as a weighted **geometric** mean:

    P_fused(y) ∝ Π_s P_s(y) ** w_s ,   w_s ≥ 0,  Σ w_s = 1

The log opinion pool is "externally Bayesian" and tends to stay sharp (unlike a linear average,
which can wash out confident-but-correct forecasts). The weights ``w_s`` are learned to minimise
a proper score (RPS) on held-out matches — so if the model adds nothing over the market, its
weight is driven toward zero, and vice versa. That learned weight is itself the headline result:
it tells you how much the model is trusted relative to the crowd.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..evaluate.metrics import log_loss, ranked_probability_score

_EPS = 1e-12


def _scorer(score: str):
    """Proper-score function by name ('rps' or 'logloss'); both are minimised."""
    if score == "rps":
        return ranked_probability_score
    if score == "logloss":
        return log_loss
    raise ValueError(f"unknown score {score!r} (use 'rps' or 'logloss')")


def log_opinion_pool(probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted geometric mean of forecasts, renormalised. ``probs`` are (n, k) arrays."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    log_mix = np.zeros_like(probs[0], dtype=float)
    for p, w in zip(probs, weights):
        log_mix += w * np.log(np.clip(p, _EPS, 1.0))
    pooled = np.exp(log_mix)
    return pooled / pooled.sum(axis=1, keepdims=True)


def pool_two(model_p: np.ndarray, market_p: np.ndarray, w_model: float) -> np.ndarray:
    """Convenience two-source pool: weight ``w_model`` on the model, ``1-w_model`` on the market."""
    return log_opinion_pool([model_p, market_p], np.array([w_model, 1.0 - w_model]))


def fit_model_weight(
    model_p: np.ndarray,
    market_p: np.ndarray,
    outcomes: np.ndarray,
    score: str = "rps",
    prior_weight: float | None = None,
    prior_strength: float = 0.0,
) -> float:
    """Find the model weight w∈[0,1] minimising the fused ``score`` against the market (1-w).

    ``score`` is 'rps' or 'logloss'. ``prior_weight``/``prior_strength`` add an optional ridge
    ``prior_strength·(w − prior_weight)²`` that shrinks the estimate toward a prior (e.g. a small
    editorial weight) when the data is thin — set ``prior_strength=0`` (default) for the pure fit.
    """
    scorer = _scorer(score)

    def obj(w: float) -> float:
        s = scorer(pool_two(model_p, market_p, w), outcomes)
        if prior_weight is not None:
            s += prior_strength * (w - prior_weight) ** 2
        return s

    res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def cross_val_model_weight(
    model_p: np.ndarray,
    market_p: np.ndarray,
    outcomes: np.ndarray,
    k: int = 5,
    score: str = "rps",
    seed: int = 0,
) -> dict:
    """k-fold cross-validated model weight: the honest, out-of-sample blend weight + its stability.

    Each fold fits the weight on the other k−1 folds and scores the fused forecast on the held-out
    fold, so neither the weight nor the reported score sees its own evaluation data. Returns the mean
    and spread of the per-fold weights (the spread is the real signal — a weight that swings wildly
    across folds is not trustworthy) alongside the CV'd fused/market/model scores.
    """
    scorer = _scorer(score)
    n = len(outcomes)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    weights, fused_s, market_s, model_s = [], [], [], []
    for i in range(k):
        ev = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        w = fit_model_weight(model_p[tr], market_p[tr], outcomes[tr], score=score)
        weights.append(w)
        fused_s.append(scorer(pool_two(model_p[ev], market_p[ev], w), outcomes[ev]))
        market_s.append(scorer(market_p[ev], outcomes[ev]))
        model_s.append(scorer(model_p[ev], outcomes[ev]))
    return {
        "score": score,
        "k": k,
        "weights": [float(x) for x in weights],
        "mean_weight": float(np.mean(weights)),
        "std_weight": float(np.std(weights, ddof=1)),
        "cv_fused": float(np.mean(fused_s)),
        "cv_market": float(np.mean(market_s)),
        "cv_model": float(np.mean(model_s)),
    }
