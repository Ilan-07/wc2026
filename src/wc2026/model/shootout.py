"""Penalty-shootout win-propensity model (Lane 2 #3).

When a knockout tie is still level after extra time it goes to penalties. The simulator used to
resolve that with ``sigmoid(0.4 * (psi_home - psi_away))`` — a *hand-set* scale on the shrunk
historical shootout win rate ``psi``. That 0.4 was never measured. This module fits the scale
**from real shootout history** instead, and in doing so answers the honest empirical question: does
a team's past shootout record predict its next shootout, or are penalties essentially a coin flip?

The fit is leakage-free by construction. ``psi`` is itself derived from shootout wins, so regressing
an outcome on the full-history psi would be circular. Instead each shootout's feature is the psi
*difference computed from strictly prior shootouts only* (a running, chronological record), and the
logistic coefficient is fit on those temporally-honest features. ``win_prob`` then maps a psi
difference through the learned ``intercept + psi_scale * dpsi``.

Forecasting note: the strongest real shootout predictor is who shoots **first** (~60% win rate, the
Apesteguia–Palacios-Huerta result), but the order is decided by a coin toss, so it is *unusable* for
predicting a future World Cup shootout. ``shootout_ablation.py`` reports it only as a pipeline
sanity-check (a known effect the data should reveal); the production model uses team skill (psi).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class ShootoutParams:
    """Learned logistic: P(home wins) = sigmoid(intercept + psi_scale * (psi_home - psi_away))."""

    intercept: float
    psi_scale: float


def running_psi_features(
    records: list[dict], prior: float = 4.0
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build temporally-honest (X, y) from chronological shootout ``records``.

    Each record needs ``home``, ``away``, ``winner`` and ``date``. For shootout *t* the feature is
    ``psi_home - psi_away`` where each psi is the shrunk win rate over shootouts *before* t only
    (``(wins + prior/2) / (games + prior)``), so no outcome leaks into its own feature. Records where
    *neither* team has prior history (both psi at the 0.5 prior → feature exactly 0) are dropped, since
    they carry no signal. Returns ``(dpsi, y, n_skipped)``.
    """
    ordered = sorted(records, key=lambda r: r["date"])
    wins: dict[str, float] = {}
    games: dict[str, float] = {}

    def psi(t: str) -> float:
        g = games.get(t, 0.0)
        return (wins.get(t, 0.0) + prior / 2) / (g + prior)

    dpsi, y = [], []
    skipped = 0
    for r in ordered:
        h, a, w = r["home"], r["away"], r["winner"]
        seen = games.get(h, 0.0) + games.get(a, 0.0)
        if seen > 0:  # at least one side has a prior record
            dpsi.append(psi(h) - psi(a))
            y.append(1.0 if w == h else 0.0)
        else:
            skipped += 1
        # update running record AFTER using priors
        games[h] = games.get(h, 0.0) + 1
        games[a] = games.get(a, 0.0) + 1
        wins[w] = wins.get(w, 0.0) + 1
    return np.array(dpsi), np.array(y), skipped


class ShootoutModel:
    """Logistic win-propensity model for penalty shootouts, fit from shootout history."""

    def __init__(self, prior: float = 4.0, reg: float = 1.0):
        self.prior = prior
        self.reg = reg  # L2 on the slope only — keeps the scale finite on a near-coin-flip target
        self.params: ShootoutParams | None = None

    def fit(self, records: list[dict]) -> ShootoutParams:
        """Fit the logistic on temporally-honest psi-difference features (see module docstring)."""
        x, y, _ = running_psi_features(records, prior=self.prior)
        if len(y) < 10:
            # Too little data to learn a scale — fall back to a weak, hand-set prior slope.
            self.params = ShootoutParams(intercept=0.0, psi_scale=0.4)
            return self.params

        def nll(p):
            z = p[0] + p[1] * x
            ll = y * np.log(_sigmoid(z) + 1e-12) + (1 - y) * np.log(1 - _sigmoid(z) + 1e-12)
            return -ll.sum() + self.reg * p[1] ** 2

        res = minimize(nll, np.array([0.0, 0.4]), method="L-BFGS-B")
        self.params = ShootoutParams(intercept=float(res.x[0]), psi_scale=float(res.x[1]))
        return self.params

    def win_prob(self, home: str, away: str, psi: dict[str, float] | None) -> float:
        """P(home wins the shootout). Uses the learned scale; coin flip if unfit and no psi."""
        if self.params is None:
            d = 0.0 if psi is None else psi.get(home, 0.5) - psi.get(away, 0.5)
            return float(_sigmoid(0.4 * d))
        d = 0.0 if psi is None else psi.get(home, 0.5) - psi.get(away, 0.5)
        return float(_sigmoid(self.params.intercept + self.params.psi_scale * d))
