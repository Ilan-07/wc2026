"""Real FIFA World Cup 2026 forecast from the statistical spine (plan P0-P3 on real data).

Pipeline:
  1. Load real international results (martj42 dataset) and the actual WC2026 group draw.
  2. Warm-start attack ratings from World-Football Elo.
  3. Fit the Dixon-Coles match model (time-decay weighting) and validate it on a held-out
     recent split with proper scoring rules (RPS / log-loss / Brier / ECE).
  4. Monte-Carlo-simulate the tournament and print champion + stage probabilities.

    python forecast.py            # uses data/raw/results.csv (download via data/loaders docstring)

This is the fundamentals-only (Tier A) forecast. Contextual layers (P4) and market/sentiment
fusion (P5) are added on top later without changing this spine.
"""

from __future__ import annotations

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate import metrics
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel
from wc2026.simulate.tournament import TournamentSimulator


def wdl_index(gh: int, ga: int) -> int:
    return 0 if gh > ga else (1 if gh == ga else 2)


def _with_params(params) -> DixonColesModel:
    """Wrap a fitted DixonColesParams in a fresh model for the simulator's MatchModel."""
    m = DixonColesModel(half_life_days=1100.0)
    m.params = params
    return m


def main(n_sims: int = 50_000, n_boot: int = 20) -> None:
    print("Loading real international results + WC2026 draw...")
    groups = loaders.load_wc2026_groups()
    wc_teams = {t for g in groups.values() for t in g}
    matches = loaders.load_results(since="2014-01-01", min_team_matches=20, keep_teams=wc_teams)

    # Chronological hold-out for honest validation (last ~12 months).
    cutoff = matches[int(len(matches) * 0.9)]["date"]
    train = [m for m in matches if m["date"] < cutoff]
    test = [m for m in matches if m["date"] >= cutoff]
    print(f"  {len(train)} train / {len(test)} test matches (cutoff {cutoff})")

    # --- Elo warm-start ------------------------------------------------------
    elo = EloModel().fit(train)
    elo_mean = float(np.mean(list(elo.ratings.values())))
    init_attack = {t: (r - elo_mean) / 400.0 for t, r in elo.ratings.items()}

    # --- Fit Dixon-Coles (recency half-life ~3 years) ------------------------
    print("Fitting Dixon-Coles match model...")
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(train, init_attack=init_attack)
    model = MatchModel(dc)

    # --- Validate ------------------------------------------------------------
    probs, outcomes = [], []
    for m in test:
        if m["home_team"] not in dc.params.attack or m["away_team"] not in dc.params.attack:
            continue
        p = model.wdl(m["home_team"], m["away_team"], neutral=bool(m["neutral"]))
        probs.append(list(p))
        outcomes.append(wdl_index(m["home_score"], m["away_score"]))
    probs, outcomes = np.array(probs), np.array(outcomes)
    base = np.tile([1 / 3, 1 / 3, 1 / 3], (len(outcomes), 1))

    print("\nHeld-out validation (lower is better):")
    for k, v in metrics.summary(probs, outcomes).items():
        print(f"  {k:>9}: {v:.4f}")
    print(f"  {'rps(base)':>9}: {metrics.ranked_probability_score(base, outcomes):.4f}  (uniform)")

    # --- Refit on ALL data, then simulate ------------------------------------
    print("\nRefitting on full history and simulating the World Cup...")
    elo_full = EloModel().fit(matches)
    m_full = float(np.mean(list(elo_full.ratings.values())))
    init_full = {t: (r - m_full) / 400.0 for t, r in elo_full.ratings.items()}
    dc_full = DixonColesModel(half_life_days=1100.0)
    dc_full.fit(matches, init_attack=init_full)

    missing = [t for g in groups.values() for t in g if t not in dc_full.params.attack]
    if missing:
        print(f"  WARNING: no rating for {missing} (too few matches) — using model average.")

    # Phase A: bootstrap ensemble (uncertainty propagation), host advantage (default-on in the
    # simulator), and a data-driven penalty-shootout skill psi.
    psi = loaders.load_shootout_psi()
    if n_boot > 0:
        print(f"  bootstrapping {n_boot} parameter sets for uncertainty propagation...")
        ensemble = [MatchModel(_with_params(pp)) for pp in
                    dc_full.bootstrap(matches, n_boot=n_boot, seed=1, init_attack=init_full)]
        sim = TournamentSimulator(ensemble, groups, psi=psi)
    else:
        sim = TournamentSimulator(MatchModel(dc_full), groups, psi=psi)
    result = sim.run(n_sims=n_sims, seed=0)

    print(f"\n=== FIFA World Cup 2026 — fundamentals-only forecast ({n_sims:,} sims) ===\n")
    print(result.table(top=20))
    print("\nGroups:")
    for g, teams in groups.items():
        print(f"  {g}: {', '.join(teams)}")


if __name__ == "__main__":
    main()
