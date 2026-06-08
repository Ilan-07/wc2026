"""Learn the model/market blend weight by cross-validation (#8, Lane 3).

The WC2026 report blends the model's title probabilities with the market at a hand-set
MODEL_WEIGHT=0.35. This learns the weight properly: k-fold CV of the log-opinion-pool weight on the
real fixture-level data where we actually have both a model forecast and de-vigged bookmaker odds
(five leagues × three seasons, ~5k matches — the same data as ``fusion_validate.py``). The CV weight
and, crucially, its spread across folds are the honest read on how much the model should be trusted
next to the crowd.

Run: ``PYTHONPATH=src python3 blend_weight_fit.py``.
"""

from __future__ import annotations

from fusion_validate import collect
from wc2026.fusion.pool import cross_val_model_weight

EDITORIAL_WEIGHT = 0.35  # the hand-set value currently in report.py / config


def main() -> dict:
    print("Loading league odds + fitting Dixon-Coles per league...")
    model_p, market_p, outcomes = collect()
    n = len(outcomes)

    rps = cross_val_model_weight(model_p, market_p, outcomes, k=5, score="rps")
    ll = cross_val_model_weight(model_p, market_p, outcomes, k=5, score="logloss")

    print(f"\n5-fold CV blend weight on {n} held-out matches (weight on the model; 1−w on the market):\n")
    for tag, res in (("RPS", rps), ("log-loss", ll)):
        print(f"  by {tag:<8}: w = {res['mean_weight']:.3f} ± {res['std_weight']:.3f}   "
              f"folds {[round(x, 2) for x in res['weights']]}")
        print(f"            fused {res['cv_fused']:.4f}  vs market {res['cv_market']:.4f}  "
              f"vs model {res['cv_model']:.4f}")

    w = rps["mean_weight"]
    print(f"\nLearned (RPS-optimal) model weight ≈ {w:.2f}; the report ships {EDITORIAL_WEIGHT:.2f}.")
    print("Reconciliation: on liquid markets the RPS-optimal model weight is small — the de-vigged")
    print("market already dominates (the project's documented 'can't beat the market' finding). The")
    print(f"report's {EDITORIAL_WEIGHT:.2f} is a deliberate editorial choice to keep the model's voice")
    print("audible and to surface the stage probabilities the market never quotes — NOT an RPS-optimal")
    print(f"number. This script makes that trade-off explicit and gives the evidence-based anchor (~{w:.2f}).")
    return {"rps": rps, "logloss": ll, "editorial_weight": EDITORIAL_WEIGHT}


if __name__ == "__main__":
    main()
