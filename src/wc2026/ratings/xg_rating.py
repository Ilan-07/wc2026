"""xG-based attack/defense rating (the StatsBomb-data lever).

The goals Dixon-Coles rating sees *finished* goals, which are a noisy realisation of the chances a
team actually created. **Expected goals (xG)** is that same quantity with the finishing variance
removed, so an xG rating is a lower-noise read of team strength over the same matches — the premise
behind every "supercomputer" that licenses Opta/StatsBomb xG.

Model (per match, treated as neutral-venue tournament games):

    log(xG_for_i_vs_j) = mu0 + atk_i - def_j

Two rows per match (each side's xG). Because xG is **continuous**, this is fit by time-weighted
**ridge least squares** in log-space (closed form, robust) rather than Poisson MLE — the ridge term
is essential: StatsBomb open data covers only 4 tournaments (~211 matches), so most teams have just
3-6 games and their ratings must shrink toward the average (0) or they overfit a hot/cold run.

Honest scope (kept front and centre): this is **not** a full-history rating — there is no free xG
for qualifiers/friendlies. It rates only the ~50 teams that played in WC2018/WC2022/Euro2024/
Copa2024, from few games each. Its role is a *gated blend* into the goals rating for those teams
(see ``blend_rates``), and it only ships if it beats the goals-only baseline out-of-sample
(``xg_rating_ablation.py``). Coverage is always reported, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-6  # xG can be exactly 0 (a game with no real chances); floor keeps log finite.


@dataclass
class XGRatingParams:
    """Fitted xG attack/defense (log-rate space), comparable in scale to Dixon-Coles atk/def."""

    mu0: float
    attack: dict[str, float]
    defense: dict[str, float]

    def has(self, team: str) -> bool:
        return team in self.attack

    def teams(self) -> list[str]:
        return sorted(self.attack)

    def rates(self, home_team: str, away_team: str) -> tuple[float, float] | None:
        """Expected (xG_home, xG_away); None if either team has no xG rating."""
        if home_team not in self.attack or away_team not in self.attack:
            return None
        lam = np.exp(self.mu0 + self.attack[home_team] - self.defense[away_team])
        mu = np.exp(self.mu0 + self.attack[away_team] - self.defense[home_team])
        return float(lam), float(mu)


class XGRating:
    """Fit and hold an xG attack/defense rating from per-match xG records."""

    def __init__(self, half_life_days: float | None = 900.0, reg: float = 0.35):
        self.half_life_days = half_life_days
        self.reg = reg  # ridge strength; high because data is sparse (few games/team)
        self.params: XGRatingParams | None = None

    def _time_weights(self, days_ago: np.ndarray) -> np.ndarray:
        if self.half_life_days is None:
            return np.ones_like(days_ago, dtype=float)
        xi = np.log(2.0) / self.half_life_days
        return np.exp(-xi * days_ago)

    def fit(self, records: list[dict], ref_date=None) -> XGRatingParams:
        """Ridge WLS on ``records`` with home/away, home_xg/away_xg and optional date.

        Returns attack/defense per team (ridge-shrunk toward 0) plus the intercept ``mu0``.
        """
        import pandas as pd

        teams = sorted({r["home"] for r in records} | {r["away"] for r in records})
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        # parameter layout: [atk_0..atk_{n-1}, def_0..def_{n-1}, mu0]
        p_atk, p_def, p_mu0 = 0, n, 2 * n
        ncol = 2 * n + 1

        rows, ys, ws = [], [], []
        if any(r.get("date") for r in records):
            dates = pd.to_datetime([r.get("date") for r in records])
            ref = pd.to_datetime(ref_date) if ref_date is not None else dates.max()
            days = np.clip(np.asarray(ref - dates, dtype="timedelta64[D]").astype(float), 0, None)
        else:
            days = np.zeros(len(records))
        wt = self._time_weights(days)

        for r, w in zip(records, wt):
            h, a = idx[r["home"]], idx[r["away"]]
            for atk_i, def_j, xg in ((h, a, r["home_xg"]), (a, h, r["away_xg"])):
                row = np.zeros(ncol)
                row[p_atk + atk_i] = 1.0
                row[p_def + def_j] = -1.0
                row[p_mu0] = 1.0
                rows.append(row)
                ys.append(np.log(max(float(xg), 0.0) + _EPS))
                ws.append(w)

        X = np.array(rows)
        y = np.array(ys)
        sw = np.array(ws)
        # ridge: penalise attack/defense toward 0 (identifiability + shrinkage), not the intercept.
        R = np.eye(ncol)
        R[p_mu0, p_mu0] = 0.0
        XtW = X.T * sw
        beta = np.linalg.solve(XtW @ X + self.reg * R, XtW @ y)

        self.params = XGRatingParams(
            mu0=float(beta[p_mu0]),
            attack={t: float(beta[p_atk + idx[t]]) for t in teams},
            defense={t: float(beta[p_def + idx[t]]) for t in teams},
        )
        return self.params


@dataclass
class JointXGParams:
    """Shared attack/defense from a joint goals+xG fit (log-rate space, sum-to-zero)."""

    mu0: float
    attack: dict[str, float]
    defense: dict[str, float]

    def has(self, team: str) -> bool:
        return team in self.attack

    def teams(self) -> list[str]:
        return sorted(self.attack)

    def rates(self, home_team: str, away_team: str) -> tuple[float, float] | None:
        """Expected (goals_home, goals_away) under the jointly-fit rating; None if a team is unrated."""
        if home_team not in self.attack or away_team not in self.attack:
            return None
        lam = np.exp(self.mu0 + self.attack[home_team] - self.defense[away_team])
        mu = np.exp(self.mu0 + self.attack[away_team] - self.defense[home_team])
        return float(lam), float(mu)


class JointXGRating:
    """One latent attack/defense fit to goals **and** xG as two channels (measurement-error model).

    The geometric ``blend_rates`` blends two *separately* fitted ratings after the fact. This is the
    principled alternative: a single shared log-rate eta_ij = mu0 + atk_i - def_j is estimated from
    both observation channels at once —

        goals:   G_ij  ~ Poisson(exp(eta_ij))                 (every match; the anchor)
        xG:      log(xG_ij + eps) ~ Normal(eta_ij, sigma^2)    (a *noisy measurement* of the same rate)

    so xG sharpens the rating wherever it exists without overriding the goals it must also explain.
    ``xg_weight`` is the relative precision of the xG channel (1/sigma^2): at ``xg_weight=0`` the fit
    collapses to a pure goals Poisson MLE — the exact goals-only baseline the ablation compares against,
    nested in the same code path. Ridge ``reg`` shrinks attack/defense toward 0 (xG data is sparse).
    """

    def __init__(self, xg_weight: float = 1.0, reg: float = 0.1, eps: float = 1e-3):
        self.xg_weight = xg_weight
        self.reg = reg
        self.eps = eps
        self.params: JointXGParams | None = None

    def fit(self, records: list[dict]) -> JointXGParams:
        """Joint MLE on ``records`` with home/away, home_goals/away_goals, home_xg/away_xg."""
        from scipy.optimize import minimize

        teams = sorted({r["home"] for r in records} | {r["away"] for r in records})
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([idx[r["home"]] for r in records])
        ai = np.array([idx[r["away"]] for r in records])
        gh = np.array([float(r["home_goals"]) for r in records])
        ga = np.array([float(r["away_goals"]) for r in records])
        lxh = np.log(np.array([float(r["home_xg"]) for r in records]) + self.eps)
        lxa = np.log(np.array([float(r["away_xg"]) for r in records]) + self.eps)

        def unpack(p):
            atk = np.empty(n)
            deff = np.empty(n)
            atk[:-1] = p[: n - 1]
            atk[-1] = -atk[:-1].sum()
            deff[:-1] = p[n - 1 : 2 * (n - 1)]
            deff[-1] = -deff[:-1].sum()
            return atk, deff, p[2 * (n - 1)]

        def neg_ll(p):
            atk, deff, mu0 = unpack(p)
            eta_h = mu0 + atk[hi] - deff[ai]
            eta_a = mu0 + atk[ai] - deff[hi]
            pois = np.sum(gh * eta_h - np.exp(eta_h) + ga * eta_a - np.exp(eta_a))
            gauss = np.sum((lxh - eta_h) ** 2 + (lxa - eta_a) ** 2)
            penalty = self.reg * (np.sum(atk**2) + np.sum(deff**2))
            return -pois + self.xg_weight * 0.5 * gauss + penalty

        p0 = np.zeros(2 * (n - 1) + 1)
        p0[2 * (n - 1)] = np.log(max(gh.mean(), 0.1))
        res = minimize(neg_ll, p0, method="L-BFGS-B")
        atk, deff, mu0 = unpack(res.x)
        self.params = JointXGParams(
            mu0=float(mu0),
            attack={t: float(atk[idx[t]]) for t in teams},
            defense={t: float(deff[idx[t]]) for t in teams},
        )
        return self.params


def blend_rates(
    goals_rate: tuple[float, float],
    xg_params: XGRatingParams,
    home_team: str,
    away_team: str,
    weight: float,
) -> tuple[float, float]:
    """Geometric blend of goals-model rates toward xG-model rates, by ``weight`` in [0, 1].

    ``lambda_blend = lambda_goals^(1-w) * lambda_xg^w`` (a log-space convex combination, so the
    blended rate stays a positive rate). If either team lacks an xG rating, the goals rate is
    returned unchanged — the blend only acts where xG evidence exists.
    """
    xg = xg_params.rates(home_team, away_team)
    if xg is None or weight <= 0.0:
        return goals_rate
    lam = goals_rate[0] ** (1 - weight) * xg[0] ** weight
    mu = goals_rate[1] ** (1 - weight) * xg[1] ** weight
    return float(lam), float(mu)
