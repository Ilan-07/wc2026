"""Generate the WC2026 forecast dashboard (plan P6).

Blends the structural simulator with the betting market and renders a static HTML dashboard:
title odds, stage probabilities, Crowd-vs-Model divergence, and a display-only news pulse feed.

    PYTHONPATH=src python report.py            # writes data/processed/wc2026_dashboard.html

Notes
-----
* The blend is a logarithmic opinion pool with a modest model weight: our match-level validation
  showed the market is hard to beat, so we lean on it for the title number while keeping the
  simulator's unique stage probabilities (reach SF/Final).
* News pulse is informational only and never enters the forecast.
"""

from __future__ import annotations

import datetime as dt
import webbrowser
from pathlib import Path

import numpy as np

from wc2026.collective import market, sentiment
from wc2026.data import loaders
from wc2026.fusion.pool import pool_two
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel
from wc2026.reports.dashboard import ForecastRow, build_dashboard
from wc2026.simulate.tournament import TournamentSimulator

MODEL_WEIGHT = 0.35  # title-odds blend: 35% model / 65% market (market is the stronger benchmark)
NEWS_TEAMS = 8       # fetch live headlines for the top-N blended teams


def _member(params) -> MatchModel:
    """Wrap a bootstrap parameter set in a MatchModel for the simulator."""
    m = DixonColesModel(half_life_days=1100.0)
    m.params = params
    return MatchModel(m)


def fit_full_model(matches):
    elo = EloModel().fit(matches)
    m = float(np.mean(list(elo.ratings.values())))
    init = {t: (r - m) / 400.0 for t, r in elo.ratings.items()}
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(matches, init_attack=init)
    return dc, init


def main(n_sims: int = 50_000, open_browser: bool = True, n_boot: int = 25) -> None:
    print("Loading data + draw...")
    groups = loaders.load_wc2026_groups()
    teams = [t for g in groups.values() for t in g]
    matches = loaders.load_results(since="2014-01-01", min_team_matches=20, keep_teams=set(teams))
    psi = loaders.load_shootout_psi()
    # Fatigue (rest-days + travel) covariate — gate-passed on 2018/2022 (rest-days −0.0006 RPS,
    # fatigue_ablation.py). The rest-days term is validated out-of-sample; the travel term is a
    # coordinate-based prior extension applied only here (WC2026 has venue coords).
    fatigue = loaders.load_wc2026_group_fixture_fatigue()

    print("Fitting model + bootstrap ensemble (uncertainty) + simulating...")
    dc, init = fit_full_model(matches)
    ensemble = [_member(p) for p in dc.bootstrap(matches, n_boot=n_boot, seed=1, init_attack=init)]

    # Central forecast over the ensemble (host advantage on, shootout skill on, fatigue on).
    result = TournamentSimulator(ensemble, groups, psi=psi, group_fixture_fatigue=fatigue).run(
        n_sims=n_sims, seed=0)
    model_champ = result.reach_prob["champion"]

    # Per-member champion% → between-member s.d. = real parameter uncertainty band.
    per_member = []
    for mdl in ensemble:
        r = TournamentSimulator([mdl], groups, psi=psi, group_fixture_fatigue=fatigue).run(
            n_sims=max(3000, n_sims // 8), seed=3)
        per_member.append(r.reach_prob["champion"])
    model_sd = {t: float(np.std([pm[t] for pm in per_member])) for t in teams}

    # Decompose the headline uncertainty into its two distinct sources:
    #   * parameter SE  = between-ensemble-member s.d. (model_sd) — genuine "we don't know the
    #     ratings exactly" uncertainty; only more DATA shrinks it.
    #   * Monte-Carlo SE = sqrt(p(1-p)/N) — pure sampling noise from running finitely many sims;
    #     only more SIMS shrink it. With N=50k this is tiny, confirming the band is parameter-driven.
    mc_se = {t: result.standard_error(model_champ[t]) for t in teams}

    print("Loading market odds + blending...")
    odds = market.load_outright_odds()
    market_champ = market.devig_outright(odds, teams)

    order = teams
    model_row = np.array([[model_champ[t] for t in order]])
    market_row = np.array([[market_champ[t] for t in order]])
    blended_row = pool_two(model_row, market_row, MODEL_WEIGHT)[0]
    blended = dict(zip(order, blended_row))

    rows = [
        ForecastRow(
            team=t,
            model=model_champ[t],
            market=market_champ[t],
            blended=blended[t],
            reach_sf=result.reach_prob["sf"][t],
            reach_final=result.reach_prob["final"][t],
            model_sd=model_sd[t],
        )
        for t in order
    ]

    top = [r.team for r in sorted(rows, key=lambda r: r.blended, reverse=True)[:NEWS_TEAMS]]
    print(f"Fetching live news pulse for: {', '.join(top)}")
    news = sentiment.fetch_many(top, limit=5)

    html = build_dashboard(
        rows,
        news,
        n_sims=n_sims,
        model_weight=MODEL_WEIGHT,
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        odds_note="editable snapshot",
    )
    out = Path("data/processed/wc2026_dashboard.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nDashboard written to {out.resolve()}")

    # Console summary
    print("\nTop 12 (blended):")
    for i, r in enumerate(sorted(rows, key=lambda r: r.blended, reverse=True)[:12], 1):
        print(f"  {i:2d}. {r.team:<13} blended {r.blended:5.1%}  (model {r.model:4.1%} / market {r.market:4.1%})")

    # Uncertainty decomposition: how much of the model champion band is "need more data"
    # (parameter SE) vs "need more sims" (Monte-Carlo SE). The former dominates by design.
    print("\nModel champion uncertainty — parameter SE (data-limited) vs Monte-Carlo SE (sim-limited):")
    for i, r in enumerate(sorted(rows, key=lambda r: r.model, reverse=True)[:12], 1):
        print(f"  {i:2d}. {r.team:<13} {r.model:5.1%}  ± {model_sd[r.team]:4.1%} param  "
              f"± {mc_se[r.team]:.2%} MC")

    if open_browser:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
