"""AI Football Intelligence & Collective Tournament Reasoning System (WC2026).

Tier A (this package's statistical spine) is built first:
  - ratings.elo          : World Football Elo
  - ratings.dixon_coles  : correlated-Poisson (Dixon-Coles) match model + MLE
  - model.match_model    : scoreline distribution, W/D/L, knockout resolution
  - simulate             : WC2026 format + Monte Carlo tournament simulator
  - evaluate.metrics     : RPS / log-loss / Brier / calibration
"""

__version__ = "0.1.0"
