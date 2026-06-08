"""Dixon-Coles correlated-Poisson match model and its time-weighted MLE.

This is the statistical spine (plan Tier A1-A3). Goals in a match between team *i* (home)
and team *j* (away) are modelled as

    X ~ Poisson(lambda),   Y ~ Poisson(mu)
    lambda = exp(mu0 + atk_i - def_j + home)
    mu     = exp(mu0 + atk_j - def_i)

with the Dixon-Coles low-score dependence correction tau(x, y; lambda, mu, rho):

    tau(0,0) = 1 - lambda*mu*rho
    tau(0,1) = 1 + lambda*rho
    tau(1,0) = 1 + mu*rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1   otherwise

Parameters {mu0, home, rho, atk_t, def_t} are estimated by maximising a time-weighted,
identifiability-constrained log-likelihood (recent matches weighted more via an
exponential half-life). The contextual "intelligence" covariates from the plan (positional,
coach, chemistry, ...) plug into the same log-rate later as additive terms; this module
provides the base on which they sit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

# A small floor keeps log() finite and the tau correction from driving probabilities < 0.
_EPS = 1e-12

# Relative weight of a match in the likelihood by competition importance.
# NOTE (ablation finding): turning this ON *worsened* out-of-sample RPS on both the 2018 and 2022
# World Cups (-0.006, -0.0045) — friendlies carry real signal and the recency half-life already
# down-weights old matches. So ``fit(use_importance=...)`` defaults to False; this is kept only
# for experimentation, not shipped in the production forecast.
IMPORTANCE_WEIGHTS: dict[str, float] = {
    "friendly": 0.5,
    "nations_league": 0.9,
    "qualifier": 1.0,
    "continental": 1.1,
    "confederations": 1.1,
    "world_cup": 1.2,
}


def tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction. Accepts scalars or broadcastable arrays."""
    x = np.asarray(x)
    y = np.asarray(y)
    out = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


def _poisson_pmf(k, rate):
    """Poisson pmf without scipy.stats (keeps the hot path light)."""
    from scipy.special import gammaln

    k = np.asarray(k, dtype=float)
    rate = np.asarray(rate, dtype=float)
    return np.exp(k * np.log(rate + _EPS) - rate - gammaln(k + 1.0))


@dataclass
class DixonColesParams:
    """Fitted parameters. ``attack``/``defense`` are per-team dicts (sum-to-zero constrained)."""

    mu0: float
    home: float
    rho: float
    attack: dict[str, float]
    defense: dict[str, float]

    def teams(self) -> list[str]:
        return sorted(self.attack)

    def rates(self, home_team: str, away_team: str, neutral: bool = True) -> tuple[float, float]:
        """Expected goals (lambda, mu) for a fixture under the base model."""
        h = 0.0 if neutral else self.home
        lam = np.exp(self.mu0 + self.attack[home_team] - self.defense[away_team] + h)
        mu = np.exp(self.mu0 + self.attack[away_team] - self.defense[home_team])
        return float(lam), float(mu)


class DixonColesModel:
    """Fit and hold a Dixon-Coles model over a set of matches."""

    def __init__(self, max_goals: int = 10, half_life_days: float | None = 540.0):
        self.max_goals = max_goals
        self.half_life_days = half_life_days
        self.params: DixonColesParams | None = None

    # ------------------------------------------------------------------ weights
    def _time_weights(self, days_ago: np.ndarray) -> np.ndarray:
        if self.half_life_days is None:
            return np.ones_like(days_ago, dtype=float)
        xi = np.log(2.0) / self.half_life_days
        return np.exp(-xi * days_ago)

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        matches: list[dict],
        ref_date=None,
        reg: float = 0.01,
        init_attack: dict[str, float] | None = None,
        use_importance: bool = False,
        sample_weights: np.ndarray | None = None,
    ) -> DixonColesParams:
        """Estimate parameters by time-weighted, ridge-regularised maximum likelihood.

        ``matches`` items need ``home_team``, ``away_team``, ``home_score``,
        ``away_score`` and optionally ``date`` (datetime), ``neutral`` (bool) and
        ``importance`` (str). ``init_attack`` warm-starts attack ratings from Elo.
        ``reg`` is an L2 penalty pulling attack/defense toward zero.
        ``use_importance`` multiplies each match weight by :data:`IMPORTANCE_WEIGHTS`
        (so friendlies count less). ``sample_weights`` (length == len(matches)) is an extra
        multiplier used by the bootstrap to up/down-weight resampled matches.
        """
        import pandas as pd

        teams = sorted({m["home_team"] for m in matches} | {m["away_team"] for m in matches})
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)

        home_i = np.array([idx[m["home_team"]] for m in matches])
        away_i = np.array([idx[m["away_team"]] for m in matches])
        gh = np.array([int(m["home_score"]) for m in matches], dtype=float)
        ga = np.array([int(m["away_score"]) for m in matches], dtype=float)
        neutral = np.array([bool(m.get("neutral", True)) for m in matches], dtype=float)

        if any("date" in m and m["date"] is not None for m in matches):
            dates = pd.to_datetime([m.get("date") for m in matches])
            ref = pd.to_datetime(ref_date) if ref_date is not None else dates.max()
            delta = np.asarray(ref - dates, dtype="timedelta64[D]")
            days_ago = np.clip(delta.astype(float), 0.0, None)
        else:
            days_ago = np.zeros(len(matches))
        w = self._time_weights(days_ago)
        if use_importance:
            w = w * np.array(
                [IMPORTANCE_WEIGHTS.get(m.get("importance", "friendly"), 1.0) for m in matches]
            )
        if sample_weights is not None:
            w = w * np.asarray(sample_weights, dtype=float)

        # Parameter vector: [atk(0..n-2 free), def(0..n-2 free), mu0, home, rho_raw].
        # Last team's attack/defense are fixed by the sum-to-zero constraint.
        def unpack(p):
            atk = np.empty(n)
            def_ = np.empty(n)
            atk[:-1] = p[: n - 1]
            atk[-1] = -atk[:-1].sum()
            def_[:-1] = p[n - 1 : 2 * (n - 1)]
            def_[-1] = -def_[:-1].sum()
            mu0 = p[2 * (n - 1)]
            home = p[2 * (n - 1) + 1]
            rho = np.tanh(p[2 * (n - 1) + 2])  # keep rho in (-1, 1)
            return atk, def_, mu0, home, rho

        def neg_log_lik(p):
            atk, def_, mu0, home, rho = unpack(p)
            lam = np.exp(mu0 + atk[home_i] - def_[away_i] + home * (1.0 - neutral))
            mu = np.exp(mu0 + atk[away_i] - def_[home_i])
            ll_goals = (
                gh * np.log(lam + _EPS) - lam
                + ga * np.log(mu + _EPS) - mu
            )
            corr = tau(gh, ga, lam, mu, rho)
            ll = ll_goals + np.log(np.clip(corr, _EPS, None))
            penalty = reg * (np.sum(atk**2) + np.sum(def_**2))
            return -np.sum(w * ll) + penalty

        p0 = np.zeros(2 * (n - 1) + 3)
        if init_attack:
            for t, k in idx.items():
                if k < n - 1:
                    p0[k] = init_attack.get(t, 0.0)
        p0[2 * (n - 1)] = np.log(max(gh.mean(), 0.1))  # mu0 ~ log(mean goals)
        p0[2 * (n - 1) + 1] = 0.25  # home advantage
        p0[2 * (n - 1) + 2] = -0.05  # rho_raw

        res = minimize(neg_log_lik, p0, method="L-BFGS-B")
        atk, def_, mu0, home, rho = unpack(res.x)

        self.params = DixonColesParams(
            mu0=float(mu0),
            home=float(home),
            rho=float(rho),
            attack={t: float(atk[idx[t]]) for t in teams},
            defense={t: float(def_[idx[t]]) for t in teams},
        )
        return self.params

    # --------------------------------------------------------- uncertainty
    def bootstrap(
        self, matches: list[dict], n_boot: int = 30, seed: int = 0, **fit_kwargs
    ) -> list[DixonColesParams]:
        """Bayesian-bootstrap ensemble of parameter sets for uncertainty propagation.

        Each replicate re-fits the model under Dirichlet(1,...,1) match weights (mean 1), which
        perturbs the data the way a posterior would while keeping every team present (unlike
        resampling with replacement). Feeding these into the simulator — one draw per run —
        turns point-estimate champion odds into odds that carry genuine parameter uncertainty.
        """
        rng = np.random.default_rng(seed)
        nm = len(matches)
        out: list[DixonColesParams] = []
        for _ in range(n_boot):
            wts = rng.dirichlet(np.ones(nm)) * nm
            m = DixonColesModel(self.max_goals, self.half_life_days)
            out.append(m.fit(matches, sample_weights=wts, **fit_kwargs))
        return out

    # --------------------------------------------------------- scoreline grid
    def score_matrix(self, lam: float, mu: float) -> np.ndarray:
        """Joint P(X=x, Y=y) over a (max_goals+1) x (max_goals+1) grid, DC-corrected
        and renormalised so it is a proper distribution."""
        ks = np.arange(self.max_goals + 1)
        px = _poisson_pmf(ks, lam)
        py = _poisson_pmf(ks, mu)
        grid = np.outer(px, py)
        x = ks[:, None]
        y = ks[None, :]
        grid = grid * tau(x, y, lam, mu, self.rho_or_zero())
        grid = np.clip(grid, 0.0, None)
        grid /= grid.sum()
        return grid

    def rho_or_zero(self) -> float:
        return 0.0 if self.params is None else self.params.rho
