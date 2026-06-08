"""End-to-end demo of the WC2026 statistical spine (plan phases P0-P3).

Runs with zero external data: it generates a synthetic international history, fits the
Dixon-Coles match model (warm-started from Elo), validates it with proper scoring rules on a
held-out split, then Monte-Carlo-simulates the 48-team tournament and prints champion odds.

    python demo.py

Swap ``data.synthetic`` for ``data/loaders.py`` (real results + draw) to forecast for real.
"""

from __future__ import annotations

import numpy as np

from wc2026.data import synthetic
from wc2026.evaluate import metrics
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel
from wc2026.simulate.tournament import TournamentSimulator


def main() -> None:
    print("Generating synthetic international history (48 teams)...")
    teams = synthetic.make_teams(48, seed=0)
    history = synthetic.generate_history(teams, n_matches=5000, seed=1)

    # Train / test split (chronological) for honest validation.
    split = int(len(history) * 0.85)
    train, test = history[:split], history[split:]

    # --- Elo warm-start (plan A2 prior anchor) -------------------------------
    elo = EloModel().fit(train)
    elo_mean = np.mean(list(elo.ratings.values()))
    init_attack = {t: (elo.ratings.get(t, elo_mean) - elo_mean) / 400.0 for t in teams}

    # --- Fit Dixon-Coles ------------------------------------------------------
    print("Fitting Dixon-Coles match model...")
    dc = DixonColesModel(half_life_days=None)
    dc.fit(train, init_attack=init_attack)
    model = MatchModel(dc)

    # --- Validate with proper scoring rules ----------------------------------
    probs, outcomes = [], []
    for m in test:
        p_h, p_d, p_a = model.wdl(m["home_team"], m["away_team"], neutral=True)
        probs.append([p_h, p_d, p_a])
        gh, ga = m["home_score"], m["away_score"]
        outcomes.append(0 if gh > ga else (1 if gh == ga else 2))
    probs, outcomes = np.array(probs), np.array(outcomes)

    print("\nHeld-out validation (lower RPS/log-loss/Brier is better):")
    for k, v in metrics.summary(probs, outcomes).items():
        print(f"  {k:>9}: {v:.4f}")

    # Uniform baseline for context.
    base = np.tile([1 / 3, 1 / 3, 1 / 3], (len(outcomes), 1))
    print(f"  {'rps(base)':>9}: {metrics.ranked_probability_score(base, outcomes):.4f}")

    # --- Simulate the tournament ---------------------------------------------
    print("\nSimulating the World Cup (Monte Carlo)...")
    groups = synthetic.make_group_draw(teams, seed=2)
    sim = TournamentSimulator(model, groups)
    result = sim.run(n_sims=20_000, seed=0)

    print("\nChampion odds (top 16):\n")
    print(result.table(top=16))


if __name__ == "__main__":
    main()
