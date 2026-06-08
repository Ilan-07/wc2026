"""Synthetic international-football data for tests and the runnable demo.

This lets the statistical spine run end-to-end with zero external downloads. It draws each
team a latent attack/defense strength, generates a plausible chronological match history from
the *true* Dixon-Coles generative process, and lays out a 48-team, 12-group draw. Because the
data is generated from the same model family the spine fits, tests can check that the MLE
recovers the planted strengths (parameter-recovery test).

For real forecasts, replace this with ``data/loaders.py`` reading the Kaggle
"International football results 1872-present" set plus an Elo/FIFA ratings table.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from ..simulate.format import GROUP_NAMES


def make_teams(n: int = 48, seed: int = 0) -> dict[str, dict[str, float]]:
    """Return {team: {attack, defense}} with sum-to-zero latent strengths."""
    rng = np.random.default_rng(seed)
    names = [f"T{i:02d}" for i in range(n)]
    atk = rng.normal(0.0, 0.35, n)
    dfn = rng.normal(0.0, 0.30, n)
    atk -= atk.mean()
    dfn -= dfn.mean()
    return {names[i]: {"attack": float(atk[i]), "defense": float(dfn[i])} for i in range(n)}


def generate_history(
    teams: dict[str, dict[str, float]],
    n_matches: int = 4000,
    mu0: float = 0.1,
    rho: float = -0.08,
    start: date = date(2022, 1, 1),
    seed: int = 1,
) -> list[dict]:
    """Generate a chronological list of neutral-venue match dicts from the true model."""
    rng = np.random.default_rng(seed)
    names = list(teams)
    matches: list[dict] = []
    day = start
    for _ in range(n_matches):
        i, j = rng.choice(len(names), size=2, replace=False)
        ti, tj = names[i], names[j]
        lam = np.exp(mu0 + teams[ti]["attack"] - teams[tj]["defense"])
        mu = np.exp(mu0 + teams[tj]["attack"] - teams[ti]["defense"])
        gh, ga = int(rng.poisson(lam)), int(rng.poisson(mu))
        # light Dixon-Coles low-score nudge so rho is identifiable
        if gh <= 1 and ga <= 1 and rng.random() < abs(rho):
            if gh == ga:
                gh, ga = (gh, ga) if rng.random() < 0.5 else (ga, gh)
        matches.append(
            {
                "home_team": ti,
                "away_team": tj,
                "home_score": gh,
                "away_score": ga,
                "neutral": True,
                "date": day,
                "importance": "friendly",
            }
        )
        day += timedelta(days=1)
    return matches


def make_group_draw(teams: dict[str, dict[str, float]], seed: int = 2) -> dict[str, list[str]]:
    """Assign 48 teams to 12 groups of 4 (random draw; real draw is pot-based)."""
    names = list(teams)
    if len(names) != 48:
        raise ValueError("WC2026 draw needs exactly 48 teams")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(names))
    return {GROUP_NAMES[g]: order[g * 4 : g * 4 + 4] for g in range(12)}
