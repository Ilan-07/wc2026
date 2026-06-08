"""XGBoost vs Dixon-Coles head-to-head — is an ML classifier worth it as the core model?

Trains an XGBoost W/D/L classifier on running-Elo + recent-form features over the full
international history, and compares its out-of-sample RPS / log-loss to the Dixon-Coles model on
the 2018 and 2022 World Cups. Tests the claim that a structural Poisson model beats a tree
classifier in this small-data, calibration-critical regime.

    PYTHONPATH=src python xgboost_vs_dc.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.metrics import log_loss
from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.ratings.dixon_coles import DixonColesModel, _poisson_pmf, tau
from wc2026.ratings.elo import EloModel

WCS = {2018: date(2018, 6, 14), 2022: date(2022, 11, 20)}


def wdl(g, a):
    return 0 if g > a else (1 if g == a else 2)


def dc_probs(dc, h, a):
    lam, mu = dc.params.rates(h, a, neutral=True)
    ks = np.arange(11)
    g = np.outer(_poisson_pmf(ks, lam), _poisson_pmf(ks, mu))
    g = g * tau(ks[:, None], ks[None, :], lam, mu, dc.params.rho)
    g = np.clip(g, 0, None); g /= g.sum()
    return [float(np.tril(g, -1).sum()), float(np.trace(g)), float(np.triu(g, 1).sum())]


def featurize(matches):
    """Running-Elo + recent goal-form features computed *before* each match (no leakage)."""
    elo = EloModel()
    form = defaultdict(list)
    X, y, meta = [], [], []
    for m in matches:
        h, a = m["home_team"], m["away_team"]
        rh, ra = elo.rating(h), elo.rating(a)
        fh = np.mean(form[h][-5:]) if form[h] else 0.0
        fa = np.mean(form[a][-5:]) if form[a] else 0.0
        X.append([rh, ra, rh - ra, fh, fa, fh - fa, float(m.get("neutral", True))])
        y.append(wdl(m["home_score"], m["away_score"]))
        meta.append((h, a, m["date"]))
        elo.update(h, a, int(m["home_score"]), int(m["away_score"]),
                   importance=m.get("importance", "friendly"), neutral=bool(m.get("neutral", True)))
        form[h].append(m["home_score"] - m["away_score"])
        form[a].append(m["away_score"] - m["home_score"])
    return np.array(X), np.array(y), meta


def main():
    from xgboost import XGBClassifier

    allm = loaders.load_results(since="2008-01-01", min_team_matches=15)
    X, y, meta = featurize(allm)

    pooled = {"xgb": ([], []), "dc": ([], [])}
    for yr, start in WCS.items():
        tr = [i for i, (_, _, d) in enumerate(meta) if d < start]
        te = [i for i, (h, a, d) in enumerate(meta) if d >= start and d.year == yr]
        # only the World Cup matches in the test window
        te = [i for i in te if allm[i].get("importance") == "world_cup"]
        if not te:
            continue
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss")
        clf.fit(X[tr], y[tr])
        p_xgb = clf.predict_proba(X[te])

        # Dixon-Coles on the same pre-tournament window
        train_matches = [allm[i] for i in tr]
        elo = EloModel().fit(train_matches)
        mm = float(np.mean(list(elo.ratings.values())))
        dc = DixonColesModel(half_life_days=1100.0)
        dc.fit(train_matches, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})
        p_dc = []
        for i in te:
            h, a, _ = meta[i]
            if h in dc.params.attack and a in dc.params.attack:
                p_dc.append(dc_probs(dc, h, a))
                pooled["xgb"][0].append(p_xgb[te.index(i)]); pooled["xgb"][1].append(y[i])
                pooled["dc"][0].append(dc_probs(dc, h, a)); pooled["dc"][1].append(y[i])

    for name, (P, O) in pooled.items():
        P, O = np.array(P), np.array(O)
        print(f"{name:4s}: RPS {rps(P, O):.4f} | log-loss {log_loss(P, O):.4f}  (n={len(O)})")


if __name__ == "__main__":
    main()
