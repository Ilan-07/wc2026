"""Outright-odds movement signal (Gap 2, free) — bank daily snapshots, read off the drift.

The daily job already fetches WC winner odds; ``odds_api.refresh()`` now also drops a dated snapshot
into ``data/processed/odds_history/``. Over time those snapshots are a free record of how the market
*moved* — shortening (money coming in) vs drifting (money leaving) — which is exactly the "sharp money"
signal the results-only model lacks and the market has. This module reads that history and computes a
per-team movement feature.

Honest scope: this is *plumbing that starts banking data now*. The movement feature only becomes
usable once a few days of snapshots exist, and a divergence-gating model on top of it must be gate-
tested (on the free club-odds data) before it touches the forecast. ``implied`` here is the raw 1/odds
(an un-normalised win propensity); for *movement* (a difference over time) the overround cancels to
first order, so it is used directionally, not as a calibrated probability.
"""

from __future__ import annotations

from pathlib import Path

_HIST = Path(__file__).resolve().parents[3] / "data" / "processed" / "odds_history"


def snapshot_path(day: str, hist_dir: str | Path | None = None) -> Path:
    return (Path(hist_dir) if hist_dir else _HIST) / f"odds_{day}.csv"


def _parse_csv(text: str) -> dict[str, float]:
    odds: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("team,"):
            continue
        team, _, val = line.rpartition(",")
        try:
            odds[team] = float(val)
        except ValueError:
            continue
    return odds


def load_odds_history(hist_dir: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Return {YYYYMMDD: {team: decimal_odds}} from the banked daily snapshots (chronological)."""
    d = Path(hist_dir) if hist_dir else _HIST
    if not d.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for p in sorted(d.glob("odds_*.csv")):
        out[p.stem.replace("odds_", "")] = _parse_csv(p.read_text())
    return out


def movement_features(
    history: dict[str, dict[str, float]] | None = None, window: int = 7
) -> dict[str, dict]:
    """Per-team movement over the last ``window`` snapshots.

    ``delta`` = implied(now) − implied(then) on the raw 1/odds scale: **positive = shortening**
    (market backing the team more), negative = drifting. ``None`` until ≥2 snapshots exist.
    """
    history = history if history is not None else load_odds_history()
    days = sorted(history)
    if len(days) < 2:
        return {}
    now = history[days[-1]]
    then = history[days[max(0, len(days) - 1 - window)]]
    out: dict[str, dict] = {}
    for team, odds in now.items():
        p_now = 1.0 / odds if odds else None
        p_then = (1.0 / then[team]) if (team in then and then[team]) else None
        out[team] = {
            "odds_now": odds,
            "implied_now": p_now,
            "implied_then": p_then,
            "delta": (p_now - p_then) if (p_now is not None and p_then is not None) else None,
            "n_snapshots": len(days),
        }
    return out
