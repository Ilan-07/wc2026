"""Simulation-based calibration of the hierarchical Bayesian rating (Lane 3 #7).

The backtests check the model against *reality*. SBC (Talts et al. 2018) checks the inference against
*itself*: if the sampler recovers the posterior correctly, then drawing a ground truth from the prior,
simulating data from it, and re-fitting should leave the truth at a **uniformly random rank** within
its own posterior. Any systematic shape in the rank histogram is a fingerprint of broken inference —
∪ = posterior too narrow (overconfident), ∩ = too wide, a slope = biased.

The loop is the textbook one, kept exact by construction:

    1. draw (theta*, y*) jointly from the model's prior predictive  (generation == the model prior);
    2. condition on y* and sample the posterior theta | y*;
    3. record rank(theta*) = #{posterior draws < theta*} for each tracked scalar.

Done on a deliberately small Baio–Blangiardo Poisson model (few teams, few matches, short chains) so
the hundreds of refits are tractable. A chi-square test on the rank histogram flags non-uniformity.
This validates the *machinery* behind ``predict --bayesian``; it is a property of the sampler+prior,
independent of the football data.
"""

from __future__ import annotations

import numpy as np

# Scalars tracked for uniformity (well-identified globals + one example team-attack component).
TRACKED = ["mu0", "home", "sigma_att", "att_0"]


def run_sbc(
    n_sims: int = 100,
    n_teams: int = 5,
    n_matches: int = 60,
    draws: int = 200,
    tune: int = 200,
    chains: int = 2,
    seed: int = 0,
) -> dict:
    """Run SBC and return per-parameter rank arrays, the number of posterior draws, and chi-square p."""
    import pymc as pm

    rng = np.random.default_rng(seed)
    # Fixed random design: who plays whom (every match neutral; the home term is still identifiable
    # via a random subset flagged as non-neutral so the `home` parameter is exercised).
    hi = rng.integers(0, n_teams, n_matches)
    ai = (hi + rng.integers(1, n_teams, n_matches)) % n_teams  # away != home
    home_flag = rng.integers(0, 2, n_matches).astype(float)

    with pm.Model() as model:
        gh_data = pm.Data("gh_data", np.zeros(n_matches, dtype="int64"))
        ga_data = pm.Data("ga_data", np.zeros(n_matches, dtype="int64"))
        mu0 = pm.Normal("mu0", 0.0, 1.0)
        home = pm.Normal("home", 0.25, 0.5)
        sigma_att = pm.HalfNormal("sigma_att", 0.5)
        sigma_def = pm.HalfNormal("sigma_def", 0.5)
        att = pm.ZeroSumNormal("att", sigma=sigma_att, shape=n_teams)
        deff = pm.ZeroSumNormal("def", sigma=sigma_def, shape=n_teams)
        lam = pm.math.exp(mu0 + att[hi] - deff[ai] + home * home_flag)
        mu = pm.math.exp(mu0 + att[ai] - deff[hi])
        pm.Poisson("gh", lam, observed=gh_data, shape=gh_data.shape)
        pm.Poisson("ga", mu, observed=ga_data, shape=ga_data.shape)

        # 1. joint prior draws of parameters and simulated data.
        prior = pm.sample_prior_predictive(draws=n_sims, random_seed=seed)

    pri = prior.prior
    prip = prior.prior_predictive
    ranks: dict[str, list[int]] = {k: [] for k in TRACKED}
    n_post = draws * chains

    for i in range(n_sims):
        gh_i = np.asarray(prip["gh"].values[0, i]).astype("int64")
        ga_i = np.asarray(prip["ga"].values[0, i]).astype("int64")
        with model:
            pm.set_data({"gh_data": gh_i, "ga_data": ga_i})
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=chains,
                progressbar=False, random_seed=seed + i + 1,
                compute_convergence_checks=False,
            )
        post = idata.posterior
        truth = {
            "mu0": float(pri["mu0"].values[0, i]),
            "home": float(pri["home"].values[0, i]),
            "sigma_att": float(pri["sigma_att"].values[0, i]),
            "att_0": float(pri["att"].values[0, i, 0]),
        }
        draws_of = {
            "mu0": post["mu0"].values.ravel(),
            "home": post["home"].values.ravel(),
            "sigma_att": post["sigma_att"].values.ravel(),
            "att_0": post["att"].values[:, :, 0].ravel(),
        }
        for k in TRACKED:
            ranks[k].append(int((draws_of[k] < truth[k]).sum()))

    return {
        "ranks": {k: np.array(v) for k, v in ranks.items()},
        "n_post": n_post,
        "n_sims": n_sims,
        "uniformity": {k: rank_uniformity_pvalue(np.array(ranks[k]), n_post) for k in TRACKED},
    }


def rank_uniformity_pvalue(ranks: np.ndarray, n_post: int, n_bins: int = 10) -> float:
    """Chi-square p-value that the SBC ranks are uniform on {0..n_post} (low p ⇒ miscalibrated)."""
    from scipy.stats import chi2

    edges = np.linspace(-0.5, n_post + 0.5, n_bins + 1)
    counts, _ = np.histogram(ranks, bins=edges)
    expected = len(ranks) / n_bins
    if expected <= 0:
        return float("nan")
    stat = float(np.sum((counts - expected) ** 2 / expected))
    return float(chi2.sf(stat, df=n_bins - 1))
