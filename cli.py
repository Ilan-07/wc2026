"""wc2026 — one entry point for the whole system (replaces the scattered scripts).

    PYTHONPATH=src python cli.py <command> [options]

Commands:
    predict     live winner forecast + dashboard + archive   (--refresh, --refresh-odds)
    intel       per-team traceable intelligence report       [teams...]
    scenario    injury / availability what-if
    score       grade the model against past World Cups
    validate    market-fusion validation on club odds
    odds        refresh live outright odds from The Odds API

  Validation / research (the ablation gates behind the modelling choices):
    backtest          out-of-sample RPS across 9 tournaments (WC + Euro + Copa)   [#5]
    stage-reliability deep-run (QF/SF/final/title) calibration over WCs           [#6]
    sbc               simulation-based calibration of the Bayesian sampler        [#7]
    blend-weight      cross-validated model/market blend weight                   [#8]
    state-space       dynamic state-space rating sweep vs Dixon-Coles             [#1]
    bayes-tau         Bayesian tau + pooled-home ablation                         [#2]
    shootout          penalty-shootout win-propensity ablation                    [#3]
    xg-joint          xG measurement-error joint-fit ablation                     [#4]

Run after each matchday during the tournament:  python cli.py predict --refresh --refresh-odds
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="wc2026", description="WC2026 forecasting system")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("predict", help="live winner forecast + dashboard")
    # refresh is ON by default so a stale run can't happen (gap #18); opt out for speed
    sp.add_argument("--no-refresh", action="store_true", help="skip re-downloading results")
    sp.add_argument("--no-odds", action="store_true", help="skip re-fetching live odds")
    sp.add_argument("--no-bayesian", action="store_true",
                    help="use the faster MLE rating instead of the default hierarchical Bayesian one")
    sp.add_argument("--sims", type=int, default=30_000)
    # Live-dashboard / frequent-tick options.
    sp.add_argument("--if-changed", action="store_true",
                    help="re-fit + regenerate only when a tracked input changed (cheap no-op otherwise)")
    sp.add_argument("--reload-secs", type=int, default=300,
                    help="auto-reload interval baked into the dashboard HTML (default 300s)")
    sp.add_argument("--odds-min-interval", type=float, default=6.0,
                    help="with --if-changed: min hours between paid Odds-API fetches (default 6)")

    si = sub.add_parser("intel", help="per-team intelligence report")
    si.add_argument("teams", nargs="*", help="teams to explain (default: top pick, France, Mexico)")

    sub.add_parser("scenario", help="injury what-if (edit SCENARIO in injury_scenario.py)")
    sub.add_parser("score", help="grade the model on past World Cups")
    sub.add_parser("calibrate", help="tournament-stage calibration backtest + reliability plot")
    sub.add_parser("validate", help="market-fusion validation on club odds")
    sub.add_parser("odds", help="refresh live outright odds")
    # Validation / research entry points (Lane 1-3 ablation gates).
    sub.add_parser("backtest", help="out-of-sample RPS across 9 tournaments (#5)")
    sub.add_parser("stage-reliability", help="deep-run stage calibration over WCs (#6)")
    sub.add_parser("sbc", help="simulation-based calibration of the Bayesian sampler (#7)")
    sub.add_parser("blend-weight", help="cross-validated model/market blend weight (#8)")
    sub.add_parser("state-space", help="dynamic state-space rating sweep (#1)")
    sub.add_parser("bayes-tau", help="Bayesian tau + pooled-home ablation (#2)")
    sub.add_parser("shootout", help="penalty-shootout win-propensity ablation (#3)")
    sub.add_parser("xg-joint", help="xG measurement-error joint-fit ablation (#4)")
    sub.add_parser("draw-pick", help="calibrate + LOTO-validate the draw-aware match pick")
    # Free gap-closers.
    sub.add_parser("injuries", help="suggest availability from Wikipedia + news (writes *.suggested.txt)")
    sub.add_parser("probe", help="conditional-calibration probe — where is the model mis-calibrated?")
    sub.add_parser("track", help="live tournament track record — score the forecast vs WC2026 results")

    args = p.parse_args(argv)

    if args.cmd == "predict":
        import predict
        predict.main(n_sims=args.sims, refresh=not args.no_refresh, refresh_odds=not args.no_odds,
                     bayesian=not args.no_bayesian, if_changed=args.if_changed,
                     reload_secs=args.reload_secs, odds_min_interval_h=args.odds_min_interval)
    elif args.cmd == "intel":
        import intelligence_report
        intelligence_report.main(args.teams)
    elif args.cmd == "scenario":
        import injury_scenario
        injury_scenario.main()
    elif args.cmd == "score":
        import score_backtest
        score_backtest.main()
    elif args.cmd == "calibrate":
        import numpy as np

        from wc2026.evaluate.calibration import backtest, reliability_plot
        preds, actual = backtest(n_sims=15000)
        print(f"{len(preds)} teams (2018+2022) | Brier {float(np.mean((preds-actual)**2)):.3f}")
        print("plot ->", reliability_plot(preds, actual))
    elif args.cmd == "validate":
        import fusion_validate
        fusion_validate.main()
    elif args.cmd == "odds":
        from wc2026.collective.odds_api import refresh
        print("wrote", refresh())
        from wc2026.collective.odds_movement import movement_features
        mv = {t: d for t, d in movement_features().items() if d["delta"] is not None}
        if mv:
            movers = sorted(mv.items(), key=lambda kv: kv[1]["delta"], reverse=True)
            print("market movers (implied-prob Δ over window):")
            for t, d in movers[:3]:
                print(f"  shortening  {t:<16} {d['delta']:+.4f}")
            for t, d in movers[-3:][::-1]:
                print(f"  drifting    {t:<16} {d['delta']:+.4f}")
        else:
            print("(odds movement: need ≥2 daily snapshots; banking starts now)")
    elif args.cmd == "injuries":
        from wc2026.collective.availability import suggest_injuries
        print("wrote suggestions ->", suggest_injuries())
    elif args.cmd == "probe":
        import probe
        probe.main()
    elif args.cmd == "track":
        import score_live
        score_live.main()
    elif args.cmd == "backtest":
        import tournament_backtest
        tournament_backtest.main()
    elif args.cmd == "stage-reliability":
        import stage_reliability
        stage_reliability.main()
    elif args.cmd == "sbc":
        import sbc_validate
        sbc_validate.main()
    elif args.cmd == "blend-weight":
        import blend_weight_fit
        blend_weight_fit.main()
    elif args.cmd == "state-space":
        import state_space_sweep
        state_space_sweep.main()
    elif args.cmd == "bayes-tau":
        import bayesian_tau_ablation
        bayesian_tau_ablation.main()
    elif args.cmd == "shootout":
        import shootout_ablation
        shootout_ablation.main()
    elif args.cmd == "xg-joint":
        import xg_joint_ablation
        xg_joint_ablation.main()
    elif args.cmd == "draw-pick":
        import draw_pick_calibration
        draw_pick_calibration.main()


if __name__ == "__main__":
    main(sys.argv[1:])
