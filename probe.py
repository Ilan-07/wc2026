"""Conditional-calibration probe (#Gap 4, free): where is the model mis-calibrated?

Run: ``PYTHONPATH=src python3 probe.py``  (reuses the 9-tournament backtest).
"""

from __future__ import annotations

from wc2026.evaluate.conditional_calibration import conditional_reliability, goals_calibration
from wc2026.evaluate.tournament_backtest import run


def main() -> dict:
    rows = run()["rows"]  # fit once, reuse for both probes
    cr = conditional_reliability(rows)
    gc = goals_calibration(rows)

    print("TOP-PICK CONFIDENCE CALIBRATION (conf_gap +ve = overconfident):\n")
    o = cr["overall"]
    print(f"  overall            n={o['n']:>3}  conf {o['mean_conf']:.3f}  hit {o['top_hit_rate']:.3f}"
          f"  gap {o['conf_gap']:+.3f}  ECE {o['ece']:.3f}")
    print("  by confidence tier:")
    for name, d in cr["by_confidence"].items():
        print(f"    {name:<18} n={d['n']:>3}  conf {d['mean_conf']:.3f}  hit {d['top_hit_rate']:.3f}"
              f"  gap {d['conf_gap']:+.3f}")
    print("  by predicted class:")
    for name, d in cr["by_predicted_class"].items():
        print(f"    {name:<18} n={d['n']:>3}  conf {d['mean_conf']:.3f}  hit {d['top_hit_rate']:.3f}"
              f"  gap {d['conf_gap']:+.3f}")

    print("\nEXPECTED vs ACTUAL TOTAL GOALS (bias +ve = model over-predicts):\n")
    print(f"  overall: pred {gc['mean_pred_total']:.2f}  act {gc['mean_act_total']:.2f}"
          f"  bias {gc['bias']:+.2f}  (n={gc['n']})")
    for name, d in gc["by_pred_total"].items():
        print(f"    pred-total {name:<12} n={d['n']:>3}  pred {d['pred']:.2f}  act {d['act']:.2f}"
              f"  bias {d['bias']:+.2f}")
    print("\nLargest |conf_gap| / |bias| slices are where to spend modelling effort "
          "(these are the numbers the market doesn't quote).")
    return {"conditional_reliability": cr, "goals_calibration": gc}


if __name__ == "__main__":
    main()
