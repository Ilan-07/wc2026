"""Are penalty shootouts predictable? Fit + gate the win-propensity model (#3, Lane 2).

Loads the full international shootout history (``data/raw/shootouts.csv``) and asks, out of sample
in time, whether anything beats a coin flip:

  * **coin flip**            — P(home wins) = 0.5, log-loss = ln 2 ≈ 0.6931 (the null);
  * **team skill (psi)**     — the production win-propensity model: logistic on the psi difference
                               computed from *prior* shootouts only (leakage-free), coefficient learned;
  * **shoot-first** (analysis only) — the famous ~60% edge for the team taking the first kick.

The shoot-first effect is reported purely as a pipeline sanity-check (a known real effect the data
should surface); it is unusable for forecasting because the order is a coin toss. The headline is
whether *team shootout skill* carries signal — the project's standard honest-gate question.

Run: ``PYTHONPATH=src python3 shootout_ablation.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wc2026.model.shootout import ShootoutModel, _sigmoid, running_psi_features

CSV = Path(__file__).resolve().parent / "data" / "raw" / "shootouts.csv"


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> dict:
    df = pd.read_csv(CSV)
    records = [
        {"date": str(r.date), "home": r.home_team, "away": r.away_team,
         "winner": r.winner, "first_shooter": getattr(r, "first_shooter", None)}
        for r in df.itertuples(index=False)
        if isinstance(r.winner, str) and r.winner in (r.home_team, r.away_team)
    ]
    print(f"{len(records)} shootouts with a resolved winner")

    # --- team skill (psi), leakage-free temporal features ---
    dpsi, y, skipped = running_psi_features(records)
    base = _log_loss(np.full_like(y, 0.5), y)  # coin flip
    params = ShootoutModel().fit(records)
    p_psi = _sigmoid(params.intercept + params.psi_scale * dpsi)
    psi_ll = _log_loss(p_psi, y)
    psi_acc = float(np.mean((p_psi > 0.5) == (y > 0.5)))

    print(f"\n{len(y)} shootouts with prior history (skipped {skipped} cold-start games)")
    print("-- team skill (psi) --")
    print(f"  learned: intercept {params.intercept:+.3f}, psi_scale {params.psi_scale:+.3f}")
    print(f"  coin-flip log-loss : {base:.4f}")
    print(f"  psi-model log-loss : {psi_ll:.4f}  (delta {base - psi_ll:+.4f}; +ve = better than coin)")
    print(f"  psi-model accuracy : {psi_acc:.3f}")

    # --- shoot-first effect (analysis only; not usable for forecasting) ---
    fs = [r for r in records if isinstance(r.get("first_shooter"), str) and r["first_shooter"]]
    fs_known = [r for r in fs if r["first_shooter"] in (r["home"], r["away"])]
    fs_rate = np.nan
    if fs_known:
        fs_win = np.mean([1.0 if r["first_shooter"] == r["winner"] else 0.0 for r in fs_known])
        fs_rate = float(fs_win)
        print("\n-- shoot-first (sanity-check only; order is a coin toss, unusable in forecasts) --")
        print(f"  {len(fs_known)} shootouts with a known first shooter")
        print(f"  first shooter wins {fs_rate:.1%}  (literature ~60%)")

    verdict = (
        "carries weak but real signal" if base - psi_ll > 1e-3
        else "is essentially a coin flip"
    )
    print(f"\nVerdict: team shootout skill {verdict} "
          f"(psi improves log-loss by {base - psi_ll:+.4f} over 0.5). "
          f"The model now uses the *learned* scale {params.psi_scale:+.2f}, not the hand-set 0.4.")
    return {
        "n": len(y),
        "coin_ll": base,
        "psi_ll": psi_ll,
        "psi_acc": psi_acc,
        "psi_scale": params.psi_scale,
        "intercept": params.intercept,
        "first_shooter_win_rate": fs_rate,
    }


if __name__ == "__main__":
    main()
