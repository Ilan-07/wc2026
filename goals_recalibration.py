"""Probe follow-up: does recalibrating the model's total-goals expectation help? (free)

The conditional-calibration probe found the model's predicted *totals* are over-dispersed (bias −0.17
overall, −0.32 in low-scoring games). This fits an affine map ``T' = alpha + beta*(lam+mu)`` on the
training tournaments — preserving the supremacy ``lam-mu`` so the favourite/W-D-L tilt is unchanged,
only the total (hence draw sharpness + scoreline spread) is corrected — applies it **leave-one-
tournament-out**, and reports both the residual goals bias and the W/D/L RPS. It ships only if it
beats the baseline out-of-sample. Run: ``PYTHONPATH=src python3 goals_recalibration.py``.
"""

from __future__ import annotations

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.metrics import ranked_probability_score as rps
from wc2026.evaluate.tournament_backtest import SPECS, fit_rating
from wc2026.ratings.dixon_coles import _poisson_pmf, tau


def _wdl(g, a):
    return 0 if g > a else (1 if g == a else 2)


def grid_wdl(lam, mu, rho):
    ks = np.arange(11)
    g = np.outer(_poisson_pmf(ks, lam), _poisson_pmf(ks, mu)) * tau(ks[:, None], ks[None, :], lam, mu, rho)
    g = np.clip(g, 0, None)
    g /= g.sum()
    return [float(np.tril(g, -1).sum()), float(np.trace(g)), float(np.triu(g, 1).sum())]


def collect() -> dict[str, list[tuple]]:
    """Per-tournament rows of (lam, mu, rho, outcome, actual_total) from the production rating."""
    allm = loaders.load_results(since="2006-01-01", min_team_matches=15)
    folds: dict[str, list[tuple]] = {}
    for s in SPECS:
        train = [m for m in allm if m["date"] < s.start]
        model = fit_rating(train)
        rho = model.params.rho
        rated = set(model.params.attack)
        rows = []
        for m in allm:
            if (m["tournament"] == s.name and m["date"].year == s.year and m["date"] >= s.start
                    and m["home_team"] in rated and m["away_team"] in rated):
                lam, mu = model.rates(m["home_team"], m["away_team"], neutral=True)
                rows.append((lam, mu, rho, _wdl(m["home_score"], m["away_score"]),
                             int(m["home_score"]) + int(m["away_score"])))
        if len(rows) >= 10:
            folds[s.key] = rows
    return folds


def fit_affine(rows: list[tuple]) -> tuple[float, float]:
    pred = np.array([lam + mu for lam, mu, *_ in rows])
    act = np.array([t for *_, t in rows])
    beta, alpha = np.polyfit(pred, act, 1)  # actual ~ beta*pred + alpha
    return float(alpha), float(beta)


def _score(rows, alpha=None, beta=None):
    P, O, pred_t, act_t = [], [], [], []
    for lam, mu, rho, o, at in rows:
        if alpha is None:
            la, mb = lam, mu
        else:
            t = max(0.2, alpha + beta * (lam + mu))
            s = lam - mu
            la, mb = max(1e-3, (t + s) / 2), max(1e-3, (t - s) / 2)
        P.append(grid_wdl(la, mb, rho))
        O.append(o)
        pred_t.append(la + mb)
        act_t.append(at)
    return rps(np.array(P), np.array(O)), float(np.mean(pred_t) - np.mean(act_t))


def main() -> dict:
    folds = collect()
    keys = list(folds)
    nb = base_w = cal_w = bias_b = bias_c = 0.0
    for test in keys:
        train = [r for k in keys if k != test for r in folds[k]]
        alpha, beta = fit_affine(train)
        b_rps, b_bias = _score(folds[test])
        c_rps, c_bias = _score(folds[test], alpha, beta)
        n = len(folds[test])
        nb += n
        base_w += b_rps * n; cal_w += c_rps * n; bias_b += b_bias * n; bias_c += c_bias * n
        print(f"{test:9s} n={n:3d}  RPS {b_rps:.4f} -> {c_rps:.4f} ({b_rps - c_rps:+.4f})  "
              f"goalbias {b_bias:+.2f} -> {c_bias:+.2f}   (alpha={alpha:.2f} beta={beta:.2f})")
    base, cal = base_w / nb, cal_w / nb
    print(f"\nPOOLED ({int(nb)} matches): RPS base {base:.4f} -> recal {cal:.4f}  (delta {base - cal:+.4f})")
    print(f"  goals bias {bias_b / nb:+.3f} -> {bias_c / nb:+.3f}  (closer to 0 = better-calibrated totals)")
    verdict = ("IMPROVES W/D/L RPS → wire it" if base - cal > 1e-4
               else "neutral on W/D/L (expected — totals are ~orthogonal to the favourite); "
                    "keep as a scoreline/Over-Under calibration, not a W/D/L change")
    print(f"VERDICT: recalibration {verdict}.")
    return {"base_rps": base, "recal_rps": cal, "bias_before": bias_b / nb, "bias_after": bias_c / nb}


if __name__ == "__main__":
    main()
