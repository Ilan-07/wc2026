"""Central configuration (gap #27) — one place for the parameters scattered across the code.

Plain Python (no YAML dependency). Import ``CONFIG`` and read fields; override per-run by mutating
the instance. Values are the backtest-tuned / documented defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # rating / model
    half_life_days: float = 1100.0      # Dixon-Coles recency half-life (tuned via tune_halflife.py)
    min_team_matches: int = 20          # established-team filter for training
    history_since: str = "2014-01-01"   # training window start

    # simulation
    n_sims: int = 30_000
    n_boot: int = 15                    # bootstrap ensemble size (uncertainty propagation)
    altitude_per_1000m: float = 0.12    # altitude penalty scale (unvalidated prior)

    # fusion
    model_weight: float = 0.35          # log-opinion-pool weight on the model (market gets the rest)

    # data
    odds_regions: str = "us,uk,eu"
    news_teams: int = 8


CONFIG = Config()
