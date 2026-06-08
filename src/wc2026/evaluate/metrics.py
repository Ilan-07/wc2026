"""Proper scoring rules and calibration diagnostics (plan validation section).

These are the metrics that *prove* the model is rigorous rather than just plausible. For
football's ordinal Home/Draw/Away outcome the headline metric is the Ranked Probability
Score (RPS); log-loss and the multiclass Brier score complement it, and a reliability
curve / Expected Calibration Error check that stated probabilities match observed frequencies.

Conventions
-----------
* W/D/L outcomes are ordered ``[home, draw, away]``.
* ``probs`` is an (n, k) array of forecast probabilities (rows sum to 1).
* ``outcomes`` is an (n,) integer array of realised class indices.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-15


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean RPS for ordered-categorical forecasts (lower is better).

    RPS = (1/(k-1)) * sum_{i=1}^{k-1} (CDF_pred_i - CDF_obs_i)^2, averaged over matches.
    It rewards forecasts whose *cumulative* mass sits near the true ordered outcome, so
    predicting a draw when the home team wins is penalised less than predicting an away win.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), outcomes] = 1.0
    cdf_pred = np.cumsum(probs, axis=1)
    cdf_obs = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cdf_pred - cdf_obs) ** 2, axis=1) / (k - 1)))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean multiclass cross-entropy (lower is better)."""
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0)
    outcomes = np.asarray(outcomes, dtype=int)
    return float(-np.mean(np.log(probs[np.arange(len(outcomes)), outcomes])))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean multiclass Brier score = mean squared error vs the one-hot outcome."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n, k = probs.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_curve(
    probs: np.ndarray, outcomes: np.ndarray, class_idx: int = 0, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability data for one class: (mean predicted prob, observed freq, bin count).

    A perfectly calibrated model lies on the diagonal (predicted == observed) in every bin.
    """
    p = np.asarray(probs, dtype=float)[:, class_idx]
    hit = (np.asarray(outcomes, dtype=int) == class_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = which == b
        counts[b] = int(mask.sum())
        if counts[b]:
            mean_pred[b] = p[mask].mean()
            obs_freq[b] = hit[mask].mean()
    return mean_pred, obs_freq, counts


def expected_calibration_error(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over all classes: count-weighted mean |predicted - observed| across bins."""
    probs = np.asarray(probs, dtype=float)
    n, k = probs.shape
    total = 0.0
    for c in range(k):
        mean_pred, obs_freq, counts = reliability_curve(probs, outcomes, c, n_bins)
        valid = counts > 0
        total += np.sum(counts[valid] * np.abs(mean_pred[valid] - obs_freq[valid]))
    return float(total / (n * k))


def summary(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """Convenience bundle of all headline metrics."""
    return {
        "rps": ranked_probability_score(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "brier": brier_score(probs, outcomes),
        "ece": expected_calibration_error(probs, outcomes),
        "n": float(len(outcomes)),
    }
