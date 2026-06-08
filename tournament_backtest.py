"""Backtest the rating across every reconstructable major tournament (#5, Lane 3).

Widens the out-of-sample skill evidence from 2 World Cups to ~9 tournaments (WC + Euro + Copa).
Run: ``PYTHONPATH=src python3 tournament_backtest.py``.
"""

from __future__ import annotations

from wc2026.evaluate.tournament_backtest import run


def main() -> dict:
    res = run()
    print(f"Out-of-sample W/D/L RPS across {res['n_tournaments']} tournaments "
          f"({res['n_matches']} matches):\n")
    print(f"  {'tournament':<20} {'n':>3}   {'RPS':>7}  {'uniform':>7}   skill")
    for r in res["rows"]:
        print(f"  {r['label']:<20} {r['n']:>3}   {r['rps']:.4f}   {r['uniform_rps']:.4f}   "
              f"{r['uniform_rps'] - r['rps']:+.4f}")
    print(f"\n  pooled RPS         : {res['pooled_rps']:.4f}")
    print(f"  pooled uniform RPS : {res['pooled_uniform_rps']:.4f}")
    print(f"  skill vs uniform   : {res['skill_vs_uniform']:+.4f}  (pooled over matches)")
    print(f"  per-tournament RPS : {res['per_tournament_rps_mean']:.4f} "
          f"± {res['per_tournament_rps_std']:.4f} (mean ± sd across tournaments)")
    print(f"  pooled log-loss    : {res['pooled_log_loss']:.4f}")
    print("\nThe per-tournament spread is the honest measure of how much a single-WC number can move.")
    return res


if __name__ == "__main__":
    main()
