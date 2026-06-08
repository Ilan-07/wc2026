"""Betting-market signal (plan Tier B).

Bookmaker odds are the single strongest *aggregator* of football information — they already
embed statistics, news, and sentiment — which is exactly why the plan keeps them out of the
Dixon-Coles goal model and brings them in only at the fusion stage, as a benchmark forecast.

This module (a) converts decimal odds into a proper probability distribution by removing the
bookmaker's overround ("de-vigging"), and (b) loads the football-data.co.uk league CSVs used to
*validate* the fusion at match level (the plan's "train fusion weights at match level" rule).

    # download league odds, e.g. EPL/Bundesliga/La Liga/Serie A/Ligue 1, recent seasons:
    curl -sSL -o data/raw/odds/E0_2324.csv https://www.football-data.co.uk/mmz4281/2324/E0.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ODDS_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "odds"


def devig(odds_home: float, odds_draw: float, odds_away: float) -> np.ndarray:
    """Decimal odds -> de-vigged [P(home), P(draw), P(away)] (proportional normalization).

    Raw implied probability is 1/odds; these sum to >1 by the bookmaker's margin (overround),
    so we divide by their sum to recover a proper distribution.
    """
    raw = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away], dtype=float)
    return raw / raw.sum()


def overround(odds_home: float, odds_draw: float, odds_away: float) -> float:
    """Bookmaker margin = (sum of raw implied probs) - 1. Typically ~0.03-0.07."""
    return float(1.0 / odds_home + 1.0 / odds_draw + 1.0 / odds_away - 1.0)


# football-data.co.uk uses 'H'/'D'/'A' for full-time result; map to our 0/1/2 ordering.
_FTR = {"H": 0, "D": 1, "A": 2}


def load_odds_csv(path: str | Path, prefer: str = "Avg") -> list[dict]:
    """Load one football-data.co.uk CSV into match dicts with market probabilities.

    ``prefer`` selects the odds columns: 'Avg' (market average AvgH/D/A) or 'B365' (Bet365).
    Falls back to Bet365 if the preferred columns are missing.
    """
    df = pd.read_csv(path, encoding="latin-1")
    out: list[dict] = []
    for d in df.to_dict("records"):
        try:
            oh = d.get(f"{prefer}H") or d.get("B365H")
            od = d.get(f"{prefer}D") or d.get("B365D")
            oa = d.get(f"{prefer}A") or d.get("B365A")
            gh, ga = d["FTHG"], d["FTAG"]
            if any(pd.isna(x) for x in (oh, od, oa, gh, ga)):
                continue
            out.append(
                {
                    "home_team": d["HomeTeam"],
                    "away_team": d["AwayTeam"],
                    "home_score": int(gh),
                    "away_score": int(ga),
                    "date": pd.to_datetime(d["Date"], dayfirst=True).date(),
                    "neutral": False,  # club league matches are home/away
                    "market_prob": devig(float(oh), float(od), float(oa)),
                    "outcome": _FTR.get(str(d.get("FTR")), None),
                }
            )
        except (KeyError, ValueError, TypeError):
            continue
    return [m for m in out if m["outcome"] is not None]


def load_outright_odds(path: str | Path | None = None) -> dict[str, float]:
    """Load WC2026 outright (to-win) decimal odds from the snapshot CSV (skips '#' comments)."""
    p = Path(path) if path else (_ODDS_DIR.parent / "wc2026_outright_odds.csv")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    odds: dict[str, float] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("team,"):
            continue
        team, _, val = line.rpartition(",")
        odds[team.strip()] = float(val)
    return odds


def devig_outright(
    odds: dict[str, float], all_teams: list[str], default_odds: float = 500.0
) -> dict[str, float]:
    """Convert outright odds into a champion-probability distribution over ``all_teams``.

    Teams missing from the odds snapshot get ``default_odds`` (a long price). Raw implied
    probabilities (1/odds) are normalised across the field to remove the (large) futures
    overround, yielding a proper distribution that sums to 1.
    """
    raw = {t: 1.0 / odds.get(t, default_odds) for t in all_teams}
    total = sum(raw.values())
    return {t: v / total for t, v in raw.items()}


def load_league(league: str, odds_dir: str | Path | None = None) -> list[dict]:
    """Load all downloaded seasons for one league code (e.g. 'E0'), sorted by date."""
    d = Path(odds_dir) if odds_dir else _ODDS_DIR
    matches: list[dict] = []
    for p in sorted(d.glob(f"{league}_*.csv")):
        matches.extend(load_odds_csv(p))
    matches.sort(key=lambda m: m["date"])
    return matches
