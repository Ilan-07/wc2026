"""From a fitted Dixon-Coles model to match outcomes (plan Tier A4).

Wraps :class:`DixonColesModel` to produce, for any fixture:
  * the joint scoreline distribution,
  * W/D/L probabilities,
  * an expected-goals pair,
  * a sampled scoreline (for Monte Carlo),
  * a knockout winner (regulation -> extra time -> shootout).

Contextual intelligence covariates (plan A2) will later adjust ``(lambda, mu)`` here via
an optional ``log_rate_adjustment`` hook before the scoreline grid is built; the base
model is fully functional without them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ratings.dixon_coles import DixonColesModel, DixonColesParams


@dataclass
class Outcome:
    p_home: float
    p_draw: float
    p_away: float
    exp_goals_home: float
    exp_goals_away: float


class MatchModel:
    """Outcome and sampling layer over a fitted Dixon-Coles model."""

    def __init__(self, dc: DixonColesModel, adjustment=None, xg_blend=None, goals_recal=None):
        if dc.params is None:
            raise ValueError("DixonColesModel must be fitted before use")
        self.dc = dc
        self.params: DixonColesParams = dc.params
        # Optional contextual log-rate adjustment (plan P4). A callable (home, away) -> delta:
        # log λ_home += delta and log μ_home -= delta, i.e. a positive delta makes `home` the
        # stronger side. This is the seam where squad/coach/chemistry covariates enter without
        # touching the base Dixon-Coles ratings (which are fit from results only).
        self.adjustment = adjustment
        # Optional xG-rating blend (gate-passed: +0.0058 RPS out-of-sample, see xg_rating_ablation).
        # A pair (XGRatingParams, weight): the goals rate is pulled geometrically toward the team's
        # xG rate for the ~50 teams with StatsBomb xG, leaving uncovered teams untouched. Applied to
        # the base rating *before* the antisymmetric injury/altitude deltas.
        self.xg_blend = xg_blend
        # Optional total-goals recalibration (alpha, beta): T' = alpha + beta*(lam+mu), supremacy
        # lam-mu preserved. The model's predicted totals are over-dispersed (probe finding); this affine
        # map nearly zeroes the pooled total-goals bias. It is RPS-NEUTRAL on W/D/L (totals are
        # orthogonal to the favourite) — it sharpens scorelines/draw rates, not the headline pick.
        self.goals_recal = goals_recal
        # Cache the (cumulative) scoreline grid per fixture: the simulator samples the
        # same fixtures millions of times, so recomputing the DC grid would dominate runtime.
        self._cdf_cache: dict[tuple[str, str, bool], tuple[np.ndarray, int]] = {}

    # ------------------------------------------------------------- rates/grid
    def rates(
        self, home: str, away: str, neutral: bool = True, extra_delta: float = 0.0
    ) -> tuple[float, float]:
        """Expected goals (lambda, mu). ``extra_delta`` is a per-match log-rate shift applied to
        the home side (e.g. venue altitude): log lambda += extra_delta, log mu -= extra_delta."""
        lam, mu = self.params.rates(home, away, neutral=neutral)
        if self.xg_blend is not None:
            from ..ratings.xg_rating import blend_rates

            xparams, w = self.xg_blend
            lam, mu = blend_rates((lam, mu), xparams, home, away, w)
        d = extra_delta + (self.adjustment(home, away) if self.adjustment is not None else 0.0)
        if d:
            lam, mu = lam * np.exp(d), mu * np.exp(-d)
        if self.goals_recal is not None:
            # Recalibrate the total (lam+mu) while preserving the supremacy (lam-mu).
            a, b = self.goals_recal
            t = max(0.2, a + b * (lam + mu))
            s = lam - mu
            lam, mu = max(1e-3, (t + s) / 2.0), max(1e-3, (t - s) / 2.0)
        return float(lam), float(mu)

    def scoreline_grid(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        lam, mu = self.rates(home, away, neutral=neutral)
        return self.dc.score_matrix(lam, mu)

    # ----------------------------------------------------------- probabilities
    def outcome(self, home: str, away: str, neutral: bool = True) -> Outcome:
        grid = self.scoreline_grid(home, away, neutral=neutral)
        p_home = float(np.tril(grid, -1).sum())  # rows (home goals) > cols (away goals)
        p_away = float(np.triu(grid, 1).sum())
        p_draw = float(np.trace(grid))
        lam, mu = self.rates(home, away, neutral=neutral)
        return Outcome(p_home, p_draw, p_away, lam, mu)

    def wdl(self, home: str, away: str, neutral: bool = True) -> tuple[float, float, float]:
        o = self.outcome(home, away, neutral=neutral)
        return o.p_home, o.p_draw, o.p_away

    # ----------------------------------------------------------------- sampling
    def sample_score(
        self, home: str, away: str, rng: np.random.Generator, neutral: bool = True,
        extra_delta: float = 0.0,
    ) -> tuple[int, int]:
        """Draw a scoreline from the exact DC-corrected joint distribution.

        Uses inverse-CDF sampling against a cached flattened grid so repeated fixtures
        (the common case inside the Monte Carlo loop) avoid recomputing the grid. A non-zero
        ``extra_delta`` (e.g. venue altitude) bypasses the cache and builds the grid fresh.
        """
        if extra_delta:
            lam, mu = self.rates(home, away, neutral=neutral, extra_delta=extra_delta)
            grid = self.dc.score_matrix(lam, mu)
            cdf = np.cumsum(grid.ravel())
            cdf[-1] = 1.0
            k = int(np.searchsorted(cdf, rng.random()))
            return k // grid.shape[1], k % grid.shape[1]
        key = (home, away, neutral)
        cached = self._cdf_cache.get(key)
        if cached is None:
            grid = self.scoreline_grid(home, away, neutral=neutral)
            cdf = np.cumsum(grid.ravel())
            cdf[-1] = 1.0  # guard against floating-point drift
            cached = (cdf, grid.shape[1])
            self._cdf_cache[key] = cached
        cdf, ncols = cached
        k = int(np.searchsorted(cdf, rng.random()))
        return k // ncols, k % ncols

    def _sample_at_rates(self, lam: float, mu: float, rng: np.random.Generator) -> tuple[int, int]:
        """Inverse-CDF draw from the DC-corrected score grid at arbitrary (lam, mu).

        Unlike two independent ``rng.poisson`` draws, this keeps the Dixon-Coles low-score
        dependence (tau) — which is exactly what dominates a low-scoring window like extra time,
        where 0-0/1-0/1-1 carry most of the mass.
        """
        grid = self.dc.score_matrix(lam, mu)
        cdf = np.cumsum(grid.ravel())
        cdf[-1] = 1.0
        k = int(np.searchsorted(cdf, rng.random()))
        return k // grid.shape[1], k % grid.shape[1]

    def sample_knockout(
        self,
        home: str,
        away: str,
        rng: np.random.Generator,
        neutral: bool = True,
        psi: dict[str, float] | None = None,
        shootout_scale: float = 0.4,
        et_fraction: float = 30.0 / 90.0,
        shootout_model=None,
    ) -> str:
        """Return the winner of a knockout fixture.

        Regulation is sampled from the scoreline grid; a level game goes to extra time, whose
        scoreline is drawn from the **same Dixon-Coles grid at ``et_fraction`` of the regulation
        rate** (preserving the low-score correlation, not two independent Poissons); a still-level
        game goes to penalties. The shootout uses ``shootout_model`` (a fitted
        :class:`~wc2026.model.shootout.ShootoutModel`, the learned win-propensity) when supplied,
        otherwise the legacy ``sigmoid(shootout_scale * (psi_home - psi_away))`` bias; absent both a
        model and ``psi`` it is a coin flip.
        """
        gh, ga = self.sample_score(home, away, rng, neutral=neutral)
        if gh > ga:
            return home
        if ga > gh:
            return away

        # Extra time: drawn from the DC grid at a fraction of the regulation rate (keeps tau).
        lam, mu = self.rates(home, away, neutral=neutral)
        eh, ea = self._sample_at_rates(lam * et_fraction, mu * et_fraction, rng)
        if eh > ea:
            return home
        if ea > eh:
            return away

        # Penalty shootout.
        if shootout_model is not None:
            p_home = shootout_model.win_prob(home, away, psi)
        elif psi is None:
            p_home = 0.5
        else:
            d = psi.get(home, 0.0) - psi.get(away, 0.0)
            p_home = 1.0 / (1.0 + np.exp(-shootout_scale * d))
        return home if rng.random() < p_home else away
