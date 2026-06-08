"""Scoring harness — grade forecasts against reality (what makes it a *working* model).

A forecaster is only credible if it scores itself. This grades:
  * a champion forecast vs the actual winner (log-score, Brier, the winner's rank, did the top pick win),
  * match forecasts vs actual results (RPS / log-loss, reusing evaluate.metrics),
  * an archive of dated forecasts over a tournament, to show the pick sharpening toward the truth.

Use it to backtest (score a 2018/2022 forecast vs the known winner) now, and to grade the live
WC2026 forecasts after the tournament.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import log_loss, ranked_probability_score


@dataclass
class ChampionScore:
    winner: str
    p_assigned: float        # probability the forecast gave the actual winner
    rank: int                # where the winner ranked in the forecast
    top_pick: str
    top_pick_correct: bool
    log_score: float         # -log(p_winner); lower is better
    brier: float             # multiclass Brier over the champion distribution

    def __str__(self) -> str:
        ok = "✓" if self.top_pick_correct else "✗"
        return (f"winner {self.winner}: assigned {self.p_assigned:.1%} (rank #{self.rank}), "
                f"top pick {self.top_pick} {ok}, log-score {self.log_score:.3f}, brier {self.brier:.3f}")


def score_champion(champion_probs: dict[str, float], actual_winner: str) -> ChampionScore:
    """Grade a champion-probability distribution against the team that actually won."""
    ranked = sorted(champion_probs, key=lambda t: champion_probs[t], reverse=True)
    p = champion_probs.get(actual_winner, 0.0)
    brier = sum((champion_probs.get(t, 0.0) - (1.0 if t == actual_winner else 0.0)) ** 2
                for t in set(champion_probs) | {actual_winner})
    return ChampionScore(
        winner=actual_winner,
        p_assigned=p,
        rank=(ranked.index(actual_winner) + 1) if actual_winner in ranked else len(ranked) + 1,
        top_pick=ranked[0],
        top_pick_correct=(ranked[0] == actual_winner),
        log_score=-math.log(max(p, 1e-9)),
        brier=brier,
    )


def score_matches(probs, outcomes) -> dict[str, float]:
    """RPS / log-loss for a set of W/D/L match forecasts vs realised outcomes."""
    probs, outcomes = np.asarray(probs), np.asarray(outcomes)
    return {"rps": ranked_probability_score(probs, outcomes),
            "log_loss": log_loss(probs, outcomes), "n": float(len(outcomes))}


def score_archive(archive_dir: str | Path, actual_winner: str) -> list[dict]:
    """Grade every archived forecast JSON (from predict.py) — shows the pick sharpening over time."""
    rows = []
    for f in sorted(Path(archive_dir).glob("forecast_*.json")):
        rec = json.loads(f.read_text())
        cs = score_champion(rec.get("champion_odds", {}), actual_winner)
        rows.append({
            "vintage": rec.get("data_vintage"),
            "played": rec.get("group_matches_played"),
            "pick": rec.get("pick"),
            "p_winner": round(cs.p_assigned, 4),
            "log_score": round(cs.log_score, 3),
            "top_pick_correct": cs.top_pick_correct,
        })
    return rows
