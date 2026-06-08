"""Hierarchical Bayesian rating (gap #2) — partial-pooling Poisson goals model (PyMC).

The frequentist Dixon-Coles uses a fixed L2 ridge (crude pooling toward the mean) and a bootstrap
for uncertainty. This is the principled version (Baio & Blangiardo 2010): team attack/defense are
drawn from learned Normal populations, so the data *itself* decides how much to shrink — minnows
with few matches are pulled strongly toward the mean (sane ratings), strong/ well-observed teams
less so. Sampling yields a genuine posterior, which feeds the simulator directly as the uncertainty
ensemble (replacing the bootstrap).

Model:
    mu0 ~ Normal;  home ~ Normal
    sigma_att, sigma_def ~ HalfNormal                  (learned pooling strength)
    att ~ ZeroSumNormal(sigma_att);  def ~ ZeroSumNormal(sigma_def)   (sum-to-zero, identified)
    goals_home ~ Poisson(exp(mu0 + att_i - def_j + home·home_i))
    goals_away ~ Poisson(exp(mu0 + att_j - def_i))

Note: this drops the Dixon-Coles low-score (tau) correction — it's a hierarchical *Poisson* model
(rho=0), the standard Bayesian football model. The hierarchy, not tau, is the point. Output is a
``DixonColesParams`` (rho=0) so it plugs into MatchModel exactly like the MLE fit.
"""

from __future__ import annotations

import numpy as np

from .dixon_coles import DixonColesParams


class BayesianDixonColes:
    def __init__(self, draws: int = 500, tune: int = 500, chains: int = 2,
                 target_accept: float = 0.9, seed: int = 0):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.seed = seed
        self.idata = None
        self.teams: list[str] = []
        self.params: DixonColesParams | None = None

    def fit(self, matches: list[dict]) -> DixonColesParams:
        import pymc as pm

        teams = sorted({m["home_team"] for m in matches} | {m["away_team"] for m in matches})
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([idx[m["home_team"]] for m in matches])
        ai = np.array([idx[m["away_team"]] for m in matches])
        gh = np.array([int(m["home_score"]) for m in matches], dtype="int64")
        ga = np.array([int(m["away_score"]) for m in matches], dtype="int64")
        home_adv = np.array([0.0 if m.get("neutral", True) else 1.0 for m in matches])

        with pm.Model():
            mu0 = pm.Normal("mu0", 0.0, 1.0)
            home = pm.Normal("home", 0.25, 0.5)
            sigma_att = pm.HalfNormal("sigma_att", 0.5)
            sigma_def = pm.HalfNormal("sigma_def", 0.5)
            att = pm.ZeroSumNormal("att", sigma=sigma_att, shape=n)
            deff = pm.ZeroSumNormal("def", sigma=sigma_def, shape=n)
            log_lam = mu0 + att[hi] - deff[ai] + home * home_adv
            log_mu = mu0 + att[ai] - deff[hi]
            pm.Poisson("gh", mu=pm.math.exp(log_lam), observed=gh)
            pm.Poisson("ga", mu=pm.math.exp(log_mu), observed=ga)
            self.idata = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains, cores=self.chains,
                target_accept=self.target_accept, random_seed=self.seed, progressbar=False,
            )

        self.teams = teams
        post = self.idata.posterior
        att_m = post["att"].mean(("chain", "draw")).values
        def_m = post["def"].mean(("chain", "draw")).values
        self.params = DixonColesParams(
            mu0=float(post["mu0"].mean()), home=float(post["home"].mean()), rho=0.0,
            attack={t: float(att_m[idx[t]]) for t in teams},
            defense={t: float(def_m[idx[t]]) for t in teams},
        )
        return self.params

    def diagnostics(self) -> dict:
        """R-hat (max) and divergence count — confirm the sampler converged cleanly."""
        import arviz as az

        rhat = az.rhat(self.idata)
        max_rhat = float(max(float(rhat[v].max()) for v in rhat.data_vars))
        div = int(self.idata.sample_stats["diverging"].sum())
        return {"max_rhat": max_rhat, "divergences": div}

    def posterior_params(self, n: int = 30) -> list[DixonColesParams]:
        """Draw ``n`` parameter sets from the posterior → uncertainty ensemble for the simulator."""
        import arviz as az

        ext = az.extract(self.idata, group="posterior", num_samples=n)  # dataset with a 'sample' dim
        idx = {t: k for k, t in enumerate(self.teams)}
        ns = ext.sizes["sample"]
        out = []
        for s in range(ns):
            att = ext["att"].isel(sample=s).values
            deff = ext["def"].isel(sample=s).values
            out.append(DixonColesParams(
                mu0=float(ext["mu0"].isel(sample=s)), home=float(ext["home"].isel(sample=s)), rho=0.0,
                attack={t: float(att[idx[t]]) for t in self.teams},
                defense={t: float(deff[idx[t]]) for t in self.teams},
            ))
        return out


class BayesianDixonColesTauHome:
    """Hierarchical Bayesian DC with a *learned* tau correction and *partially-pooled* home effect (#2).

    Two principled additions over :class:`BayesianDixonColes`, the two pieces the Poisson version
    drops:

    * **Learned Dixon-Coles rho** — the low-score dependence correction tau(x, y; lam, mu, rho) is
      re-attached as a ``pm.Potential`` on top of the two Poisson likelihoods, with rho given a weak
      prior and *inferred from the data* (the MLE fixes it per-fit; here it carries posterior
      uncertainty). This restores the 0-0/1-0/0-1/1-1 dependence the bivariate-Poisson approximation
      misses.
    * **Partially-pooled per-team home advantage** — instead of one global ``home`` scalar, each team
      gets ``home_mu + home_off_t`` with ``home_off ~ ZeroSumNormal(home_sigma)``. Teams that play
      many home qualifiers no longer have that home form misattributed to attack strength; teams with
      few home games shrink to the global mean. This *deconfounds* attack/defense from venue, which is
      the point even though WC predictions are at neutral venues (the cleaner att/def is what ships).

    Returns a :class:`DixonColesParams` with the posterior-mean rho and the *global* home level, so it
    plugs into ``MatchModel`` exactly like the other fits (and now actually exercises the tau path).
    """

    def __init__(self, draws: int = 500, tune: int = 500, chains: int = 2,
                 target_accept: float = 0.92, seed: int = 0):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.seed = seed
        self.idata = None
        self.teams: list[str] = []
        self.params: DixonColesParams | None = None

    def fit(self, matches: list[dict]) -> DixonColesParams:
        import pymc as pm

        teams = sorted({m["home_team"] for m in matches} | {m["away_team"] for m in matches})
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([idx[m["home_team"]] for m in matches])
        ai = np.array([idx[m["away_team"]] for m in matches])
        gh = np.array([int(m["home_score"]) for m in matches], dtype="int64")
        ga = np.array([int(m["away_score"]) for m in matches], dtype="int64")
        home_flag = np.array([0.0 if m.get("neutral", True) else 1.0 for m in matches])
        # tau acts only on the four low-score cells; precompute the selector masks as constants.
        m00 = ((gh == 0) & (ga == 0)).astype(float)
        m01 = ((gh == 0) & (ga == 1)).astype(float)
        m10 = ((gh == 1) & (ga == 0)).astype(float)
        m11 = ((gh == 1) & (ga == 1)).astype(float)

        with pm.Model():
            mu0 = pm.Normal("mu0", 0.0, 1.0)
            home_mu = pm.Normal("home_mu", 0.25, 0.5)            # global home level
            home_sigma = pm.HalfNormal("home_sigma", 0.25)       # pooling strength
            home_off = pm.ZeroSumNormal("home_off", sigma=home_sigma, shape=n)
            home_team = home_mu + home_off                       # per-team home advantage
            sigma_att = pm.HalfNormal("sigma_att", 0.5)
            sigma_def = pm.HalfNormal("sigma_def", 0.5)
            att = pm.ZeroSumNormal("att", sigma=sigma_att, shape=n)
            deff = pm.ZeroSumNormal("def", sigma=sigma_def, shape=n)
            rho = pm.Normal("rho", -0.05, 0.08)                  # learned DC low-score correction

            lam = pm.math.exp(mu0 + att[hi] - deff[ai] + home_team[hi] * home_flag)
            mu = pm.math.exp(mu0 + att[ai] - deff[hi])
            pm.Poisson("gh", mu=lam, observed=gh)
            pm.Poisson("ga", mu=mu, observed=ga)

            # Dixon-Coles tau correction as a log-likelihood Potential (masks zero out other cells).
            eps = 1e-9
            log_corr = (
                m00 * pm.math.log(pm.math.clip(1.0 - lam * mu * rho, eps, np.inf))
                + m01 * pm.math.log(pm.math.clip(1.0 + lam * rho, eps, np.inf))
                + m10 * pm.math.log(pm.math.clip(1.0 + mu * rho, eps, np.inf))
                + m11 * pm.math.log(pm.math.clip(1.0 - rho, eps, np.inf))
            )
            pm.Potential("dc_tau", pm.math.sum(log_corr))

            self.idata = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains, cores=self.chains,
                target_accept=self.target_accept, random_seed=self.seed, progressbar=False,
            )

        self.teams = teams
        post = self.idata.posterior
        att_m = post["att"].mean(("chain", "draw")).values
        def_m = post["def"].mean(("chain", "draw")).values
        self.params = DixonColesParams(
            mu0=float(post["mu0"].mean()),
            home=float(post["home_mu"].mean()),
            rho=float(post["rho"].mean()),
            attack={t: float(att_m[idx[t]]) for t in teams},
            defense={t: float(def_m[idx[t]]) for t in teams},
        )
        return self.params

    # Parameters that actually feed the forecast (the per-team home_off/home_sigma are nuisances).
    _FORECAST_VARS = ("mu0", "home_mu", "rho", "sigma_att", "sigma_def", "att", "def")

    def diagnostics(self) -> dict:
        """R-hat / divergences plus the posteriors of the two new parameters.

        Reports both ``max_rhat`` (over *all* parameters) and ``forecast_max_rhat`` (over only the
        attack/defense/rho/home_mu parameters that drive predictions). When between-team home
        variance is near zero the ``home_sigma``/``home_off`` hierarchy funnels and inflates the
        global max R-hat, even though the forecast-relevant parameters have mixed — so the two
        numbers are reported separately rather than letting a nuisance funnel hide a clean fit.
        """
        import arviz as az

        rhat = az.rhat(self.idata)
        max_rhat = float(max(float(rhat[v].max()) for v in rhat.data_vars))
        fc_vars = [v for v in self._FORECAST_VARS if v in rhat.data_vars]
        forecast_max_rhat = float(max(float(rhat[v].max()) for v in fc_vars))
        div = int(self.idata.sample_stats["diverging"].sum())
        post = self.idata.posterior
        return {
            "max_rhat": max_rhat,
            "forecast_max_rhat": forecast_max_rhat,
            "divergences": div,
            "rho_mean": float(post["rho"].mean()),
            "rho_sd": float(post["rho"].std()),
            "home_mu_mean": float(post["home_mu"].mean()),
            "home_sigma_mean": float(post["home_sigma"].mean()),
        }

    def team_home_advantage(self) -> dict[str, float]:
        """Posterior-mean home advantage per team (home_mu + home_off_t) — the pooled estimates."""
        post = self.idata.posterior
        hm = float(post["home_mu"].mean())
        off = post["home_off"].mean(("chain", "draw")).values
        return {t: hm + float(off[k]) for k, t in enumerate(self.teams)}
