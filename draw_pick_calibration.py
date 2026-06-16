"""Calibrate (and honestly validate) the draw-aware match pick.

The live track record's "correct calls" is the most-likely outcome. A draw is almost never any
side's single most-likely outcome, so argmax never calls a draw — yet ~26% of tournament matches
are draws. This script asks whether calling a draw once ``p_draw`` clears a learned threshold buys
back any of those, scored leave-one-tournament-out on the 9-tournament backtest so the threshold is
never tuned on the edition it's graded on.

    PYTHONPATH=src python3 draw_pick_calibration.py

Verdict (reproducible): the accuracy-maximizing threshold sits high (~0.335) and barely fires, and
LOTO accuracy does **not** beat argmax — confirming the honest hit-rate to quote is decisive-match
accuracy (draws excluded), not a draw-aware pick. The threshold ships as a documented, conservative
option in ``wc2026.evaluate.pick``; it is not an accuracy lever. See ``FINDINGS.md``.
"""

from __future__ import annotations

from wc2026.evaluate.pick import calibrate
from wc2026.evaluate.tournament_backtest import run


def main() -> dict:
    res = calibrate(run()["rows"])
    print(f"Draw-aware pick calibration over {res['n']} backtest matches:\n")
    print(f"  pooled argmax accuracy        : {res['argmax_accuracy']:.4f}")
    print(f"  pooled best draw threshold    : {res['pooled_best_threshold']:.3f}  "
          f"(accuracy {res['pooled_best_accuracy']:.4f})")
    print("  leave-one-tournament-out (the honest test):")
    print(f"    argmax accuracy             : {res['loto_argmax_accuracy']:.4f}")
    print(f"    draw-aware accuracy         : {res['loto_draw_aware_accuracy']:.4f}")
    print(f"    gain from the draw pick     : {res['loto_gain']:+.4f}")
    verdict = "helps" if res["loto_gain"] > 0.001 else ("neutral" if res["loto_gain"] >= -0.001 else "hurts")
    print(f"\n  → draw-aware pick {verdict} out-of-sample. Quote decisive-match accuracy instead.")
    return res


if __name__ == "__main__":
    main()
