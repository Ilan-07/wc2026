"""Dynamic (time-varying) state-space team rating — a Kalman filter on goal supremacy.

The Dixon-Coles spine produces *one* attack/defense snapshot per team from a time-weighted
window: an exponential half-life is a crude, hand-set assumption about how fast strength decays.
The principled alternative is to let each team's strength be a **latent state that evolves**, and
to let the data set its drift rate. That is a local-level (random-walk) state-space model, filtered
exactly by a Kalman filter:

    state    theta_k(t) = theta_k(t-1) + w,      w ~ Normal(0, sigma_proc^2 * dt)   (per team, indep.)
    obs      d = theta_i - theta_j + home*h + e, e ~ Normal(0, sigma_obs^2)

where ``d`` is the observed goal difference of a match between home team *i* and away team *j*,
``h`` is 1 at a non-neutral venue (0 otherwise), and ``home`` is the home-advantage coefficient.
Because the model is linear-Gaussian the filter is exact: every match shrinks the two teams'
posterior variance, and the gap between a team's matches inflates it (uncertainty grows while a
team is idle, unlike fixed-K Elo). The single signal-to-noise knob ``sigma_proc / sigma_obs``
plays the role the half-life played for Dixon-Coles, but it is **tuned by out-of-sample RPS**
(see ``state_space_sweep.py``) instead of assumed.

Predictive W/D/L comes from the supremacy predictive distribution. The latent supremacy of a
fixture is Normal with

    mean  m = theta_i - theta_j + home*h
    var   V = P_ii + P_jj - 2*P_ij + sigma_obs^2      (parameter + observation uncertainty)

and an integer goal difference is a *win* when it exceeds a draw half-band ``draw_band`` (a
continuity threshold around 0), giving

    P(away) = Phi((-band - m)/sqrt(V))
    P(draw) = Phi(( band - m)/sqrt(V)) - P(away)
    P(home) = 1 - P(away) - P(draw)

Dependency-light: numpy plus ``math.erf`` for the normal CDF. The filtered means are exported as a
plain rating dict and can warm-start the Dixon-Coles attack ratings exactly like Elo does.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field

import numpy as np

# Probabilities are clipped off the exact 0/1 rail so log-loss/RPS stay finite.
_PCLIP = 1e-6


def _norm_cdf(z: float) -> float:
    """Standard-normal CDF via the error function (no scipy needed on the hot path)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class StateSpaceParams:
    """Filtered dynamic rating: per-team strength means and their posterior variances.

    ``strength`` is on the goal-difference scale (a team's standalone contribution to supremacy);
    ``variance`` is the diagonal of the posterior covariance at the freeze date. ``home`` is the
    learned home-advantage coefficient (goals) and ``sigma_obs``/``draw_band`` carry through to the
    predictive W/D/L. ``as_of`` records the date the state was frozen so idle-time variance growth
    can be applied at prediction time.
    """

    strength: dict[str, float]
    variance: dict[str, float]
    home: float
    sigma_obs: float
    draw_band: float
    sigma_proc: float
    as_of: _dt.date | None = None
    last_played: dict[str, _dt.date | None] = field(default_factory=dict)

    def teams(self) -> list[str]:
        return sorted(self.strength)

    def _inflated_var(self, team: str, as_of=None) -> float:
        """Posterior variance of a team, grown for idle days between its last match and ``as_of``."""
        v = self.variance.get(team)
        base = v if v is not None else self._prior_var
        if as_of is None or self.last_played.get(team) is None or v is None:
            return base
        try:
            dt = (as_of - self.last_played[team]).days
        except (TypeError, AttributeError):
            return base
        return base + self.sigma_proc**2 * max(dt, 0)

    # _prior_var is attached by the model after fitting (variance for an unseen team).
    _prior_var: float = 1.0

    def supremacy(self, home_team: str, away_team: str, neutral: bool = True, as_of=None):
        """Predictive supremacy (mean, variance) for a fixture, idle-inflated to ``as_of``."""
        si = self.strength.get(home_team, 0.0)
        sj = self.strength.get(away_team, 0.0)
        vi = self._inflated_var(home_team, as_of)
        vj = self._inflated_var(away_team, as_of)
        h = 0.0 if neutral else self.home
        mean = si - sj + h
        var = vi + vj + self.sigma_obs**2  # cross-cov P_ij ~ 0 between distinct teams' walks
        return mean, var

    def wdl(self, home_team: str, away_team: str, neutral: bool = True, as_of=None):
        """Predictive (P_home, P_draw, P_away) from the supremacy distribution."""
        mean, var = self.supremacy(home_team, away_team, neutral=neutral, as_of=as_of)
        sd = math.sqrt(max(var, 1e-9))
        b = self.draw_band
        p_away = _norm_cdf((-b - mean) / sd)
        p_draw = _norm_cdf((b - mean) / sd) - p_away
        p_home = 1.0 - p_away - p_draw
        p = np.clip(np.array([p_home, p_draw, p_away]), _PCLIP, None)
        return tuple(p / p.sum())


class StateSpaceRating:
    """Kalman-filter a dynamic team rating from a chronological match history.

    The full posterior covariance over teams is maintained densely; each match is a rank-1 update
    touching the two teams involved. Process noise is applied lazily per team (exact for the diagonal
    random-walk model, since independent per-team walks never create off-diagonal cross-terms), so
    idle teams correctly accrue uncertainty without an O(n^2) sweep every match.
    """

    def __init__(
        self,
        sigma_proc: float = 0.0035,
        sigma_obs: float = 1.30,
        home: float = 0.30,
        draw_band: float = 0.60,
        init_var: float = 1.0,
        max_margin: float | None = 5.0,
    ):
        self.sigma_proc = sigma_proc      # random-walk SD per day (goal-diff scale)
        self.sigma_obs = sigma_obs        # observation SD of a single match goal difference
        self.home = home                  # home-advantage coefficient (goals)
        self.draw_band = draw_band        # half-width of the draw band around 0
        self.init_var = init_var          # prior variance for a team's strength at first sight
        self.max_margin = max_margin      # clip blowout margins to limit single-match leverage
        self.params: StateSpaceParams | None = None

    def fit(self, matches: list[dict], ref_date=None) -> StateSpaceParams:
        """Run the forward Kalman filter over ``matches`` (sorted by date) and freeze the state.

        Each match needs ``home_team``, ``away_team``, ``home_score``, ``away_score`` and ideally a
        ``date`` (datetime/date); ``neutral`` (default True) toggles the home term. ``ref_date`` is
        the freeze date used to grow idle-team variance for later prediction (defaults to the last
        match date).
        """
        ms = sorted(matches, key=lambda m: m.get("date") or 0)
        teams = sorted({m["home_team"] for m in ms} | {m["away_team"] for m in ms})
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)

        mean = np.zeros(n)
        cov = np.eye(n) * self.init_var
        last_day: list[_dt.date | None] = [None] * n
        q = self.sigma_proc**2
        r = self.sigma_obs**2

        for m in ms:
            i, j = idx[m["home_team"]], idx[m["away_team"]]
            date = m.get("date")
            # predict step: inflate each involved team's variance for days idle since its last match
            for k in (i, j):
                if last_day[k] is not None and date is not None:
                    dt = (date - last_day[k]).days
                    if dt > 0:
                        cov[k, k] += q * dt
                last_day[k] = date

            neutral = bool(m.get("neutral", True))
            h = 0.0 if neutral else self.home
            d = float(m["home_score"]) - float(m["away_score"])
            if self.max_margin is not None:
                d = float(np.clip(d, -self.max_margin, self.max_margin))

            pred = mean[i] - mean[j] + h
            innov = d - pred
            # H = e_i - e_j ; S = H P H^T + R ; K = P H^T / S
            hp = cov[i, :] - cov[j, :]          # H P  (row vector)
            s = hp[i] - hp[j] + r               # = P_ii + P_jj - 2 P_ij + R
            k_gain = (cov[:, i] - cov[:, j]) / s  # P H^T / S  (column)
            mean = mean + k_gain * innov
            cov = cov - np.outer(k_gain, hp)    # Joseph-equivalent for the symmetric rank-1 update

        as_of = ref_date if ref_date is not None else (ms[-1].get("date") if ms else None)
        params = StateSpaceParams(
            strength={t: float(mean[idx[t]]) for t in teams},
            variance={t: float(cov[idx[t], idx[t]]) for t in teams},
            home=self.home,
            sigma_obs=self.sigma_obs,
            draw_band=self.draw_band,
            sigma_proc=self.sigma_proc,
            as_of=as_of,
            last_played={t: last_day[idx[t]] for t in teams},
        )
        params._prior_var = self.init_var
        self.params = params
        return params

    def init_attack(self, scale: float = 2.0) -> dict[str, float]:
        """Map filtered strengths to Dixon-Coles attack warm-starts (like Elo's init_attack).

        Supremacy is split symmetrically into attack/defense, so a team's attack prior is roughly
        half its goal-scale strength; ``scale`` damps it to the log-rate magnitude DC expects.
        """
        if self.params is None:
            raise RuntimeError("fit() must be called before init_attack()")
        return {t: s / scale for t, s in self.params.strength.items()}
