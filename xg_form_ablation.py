"""Does an xG-based form covariate improve match RPS? (the real integration test)

Having shown xG is a better leading indicator than goals, this tests the *integration*: for each
of the 4 xG tournaments, fit the base Dixon-Coles model on pre-tournament data, then for every match
where both teams have prior matches, predict W/D/L with:
  * base        — pre-tournament ratings only (what we do now)
  * + xG-form    — base shifted by theta * (running xG-diff_home − xG-diff_away)
  * + goal-form  — same but using goal-diff (the control)
theta is fit leave-one-tournament-out, so the comparison is out-of-sample. If +xG-form beats both
base and +goal-form, the signal earns integration into the live forecast.

Uses the cached data/raw/sb_match_records.json (no re-fetch).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from wc2026.data import loaders
from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.ratings.dixon_coles import DixonColesModel, _poisson_pmf, tau
from wc2026.ratings.elo import EloModel

TOURN = {"wc2018": date(2018, 6, 14), "wc2022": date(2022, 11, 20),
         "euro2024": date(2024, 6, 14), "copa2024": date(2024, 6, 20)}
CACHE = Path("data/raw/sb_match_records.json")


def wdl(g, a):
    return 0 if g > a else (1 if g == a else 2)


def grid_wdl(lam, mu, rho, delta):
    lam, mu = lam * np.exp(delta), mu * np.exp(-delta)
    ks = np.arange(11)
    g = np.outer(_poisson_pmf(ks, lam), _poisson_pmf(ks, mu))
    g = g * tau(ks[:, None], ks[None, :], lam, mu, rho)
    g = np.clip(g, 0, None)
    g /= g.sum()
    return [float(np.tril(g, -1).sum()), float(np.trace(g)), float(np.triu(g, 1).sum())]


def build(tkey, allm, records):
    start = TOURN[tkey]
    train = [m for m in allm if m["date"] < start]
    elo = EloModel().fit(train)
    mm = float(np.mean(list(elo.ratings.values())))
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(train, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})

    form: dict[str, list] = {}
    obs = []
    for r in sorted(records[tkey], key=lambda r: r["date"]):
        h, a = r["home"], r["away"]
        if h in dc.params.attack and a in dc.params.attack and form.get(h) and form.get(a):
            xg_delta = np.mean([x for _, x in form[h]]) - np.mean([x for _, x in form[a]])
            goal_delta = np.mean([g for g, _ in form[h]]) - np.mean([g for g, _ in form[a]])
            lam, mu = dc.params.rates(h, a, neutral=True)
            obs.append({"lam": lam, "mu": mu, "rho": dc.params.rho,
                        "xg_delta": float(xg_delta), "goal_delta": float(goal_delta),
                        "out": wdl(r["home_goals"], r["away_goals"])})
        form.setdefault(h, []).append((r["home_goals"] - r["away_goals"], r["home_xg"] - r["away_xg"]))
        form.setdefault(a, []).append((r["away_goals"] - r["home_goals"], r["away_xg"] - r["home_xg"]))
    return obs


def eval_set(obs, theta, feature):
    P = [grid_wdl(o["lam"], o["mu"], o["rho"], 0.0 if feature is None else theta * o[feature]) for o in obs]
    return rps(np.array(P), np.array([o["out"] for o in obs]))


def main():
    records = json.loads(CACHE.read_text())
    allm = loaders.load_results(since="2006-01-01", min_team_matches=15)
    obs_by = {k: build(k, allm, records) for k in TOURN}
    n_total = sum(len(v) for v in obs_by.values())
    print(f"Usable matches (both teams have prior form): {n_total}\n")

    keys = list(TOURN)
    for feature in ("xg_delta", "goal_delta"):
        base_w = feat_w = n = 0
        thetas = []
        for test in keys:
            train_obs = [o for k in keys if k != test for o in obs_by[k]]
            th = minimize_scalar(lambda t, _obs=train_obs, _f=feature: eval_set(_obs, t, _f),
                                 bounds=(-3, 3), method="bounded").x
            thetas.append(th)
            te = obs_by[test]
            base_w += eval_set(te, 0, None) * len(te)
            feat_w += eval_set(te, th, feature) * len(te)
            n += len(te)
        label = "xG-form" if feature == "xg_delta" else "goal-form"
        print(f"{label:9s}: base RPS {base_w/n:.4f} | +{label} RPS {feat_w/n:.4f} "
              f"| improvement {(base_w-feat_w)/n:+.4f}  (theta~{np.mean(thetas):+.2f})")


if __name__ == "__main__":
    main()
