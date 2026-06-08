"""Squad-feature covariates for the match model (plan P4 integration).

Turns the per-team squad features into a single learned log-rate adjustment that plugs into
``MatchModel(adjustment=...)``. The contribution of team *i* relative to *j* in a fixture is

    delta(i, j) = sum_k theta_k * (z_k[i] - z_k[j])

where ``z_k`` is the cross-team standardized value of feature *k* and ``theta_k`` is a weight
learned by maximum likelihood on historical World Cup matches (see ``evaluate.ablation``).
Standardizing means each weight is on a comparable scale and an unhelpful feature simply gets
a weight near zero under L2 regularization — the ablation gate from the plan.
"""

from __future__ import annotations

import numpy as np


def zscore_features(feats: dict[str, dict[str, float]]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Standardize each feature across the given team set.

    Returns ``(feature_names, {team: z_vector})``. Features with zero variance are dropped.
    """
    teams = list(feats)
    names = sorted({k for f in feats.values() for k in f})
    mat = np.array([[feats[t].get(n, 0.0) for n in names] for t in teams], dtype=float)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    keep = std > 1e-9
    names = [n for n, k in zip(names, keep) if k]
    z = (mat[:, keep] - mean[keep]) / std[keep]
    return names, {t: z[i] for i, t in enumerate(teams)}


class SquadAdjustment:
    """Callable log-rate adjustment from standardized squad features and learned weights."""

    def __init__(self, feature_names: list[str], z_by_team: dict[str, np.ndarray], theta: np.ndarray):
        self.feature_names = feature_names
        self.z = z_by_team
        self.theta = np.asarray(theta, dtype=float)
        self._zero = np.zeros(len(feature_names))

    def quality(self, team: str) -> float:
        """Scalar squad-quality contribution for a team (theta . z)."""
        return float(self.theta @ self.z.get(team, self._zero))

    def __call__(self, home: str, away: str) -> float:
        return self.quality(home) - self.quality(away)

    def contributions(self, team: str) -> dict[str, float]:
        """Per-feature contribution theta_k * z_k[team] — the explainability breakdown."""
        zt = self.z.get(team, self._zero)
        return {n: float(self.theta[i] * zt[i]) for i, n in enumerate(self.feature_names)}
