"""Tournament DNA / mentality engine (plan Phase C).

Tests the popular claim that "certain nations (Germany, Argentina, ...) over-perform their raw
ratings in major tournaments." We operationalise it as a measurable residual: in historical
World Cup / continental-final matches, how much better (or worse) did a team do than its Elo
rating expected, averaged and shrunk toward zero for small samples.

    dna_i = shrink( mean over tournament matches of [ actual_result_i − elo_expected_i ] )

It enters the match model as a covariate Δ = dna_home − dna_away (via the adjustment hook). Like
every contextual layer it is **ablation-gated**: it only ships if it improves out-of-sample RPS on
the 2018/2022 World Cups. (Expectation, given our prior findings: small or no effect, because a
team's tournament results are already in its rating. The point is to measure it, not assume it.)
"""

from __future__ import annotations

from ..ratings.elo import EloModel, expected_score

# Competitions that count as "tournament" matches for DNA.
TOURNAMENT_IMPORTANCE = {"world_cup", "continental", "confederations"}


def compute_dna(matches: list[dict], shrink: float = 10.0) -> dict[str, float]:
    """Per-team tournament over/under-performance vs Elo expectation (shrunk mean residual).

    Expected results are recomputed from *running* Elo ratings as we stream the chronological
    matches, so there is no look-ahead. ``shrink`` is the pseudo-count pulling low-sample teams
    toward 0 (no DNA signal).
    """
    run = EloModel()
    resid_sum: dict[str, float] = {}
    count: dict[str, float] = {}
    for m in matches:
        h, a = m["home_team"], m["away_team"]
        if m.get("importance") in TOURNAMENT_IMPORTANCE:
            eh = expected_score(run.rating(h), run.rating(a), 0.0)
            sh = 1.0 if m["home_score"] > m["away_score"] else (0.5 if m["home_score"] == m["away_score"] else 0.0)
            resid_sum[h] = resid_sum.get(h, 0.0) + (sh - eh)
            resid_sum[a] = resid_sum.get(a, 0.0) + ((1 - sh) - (1 - eh))
            count[h] = count.get(h, 0.0) + 1
            count[a] = count.get(a, 0.0) + 1
        run.update(h, a, int(m["home_score"]), int(m["away_score"]),
                   importance=m.get("importance", "friendly"), neutral=bool(m.get("neutral", True)))
    return {t: resid_sum[t] / (count[t] + shrink) for t in resid_sum}


class DnaAdjustment:
    """Log-rate adjustment Δ(home, away) = scale · (dna_home − dna_away)."""

    def __init__(self, dna: dict[str, float], scale: float = 1.0):
        self.dna = dna
        self.scale = scale

    def __call__(self, home: str, away: str) -> float:
        return self.scale * (self.dna.get(home, 0.0) - self.dna.get(away, 0.0))


def top_dna(dna: dict[str, float], n: int = 10) -> list[tuple[str, float]]:
    """Most over-performing nations (highest tournament residual)."""
    return sorted(dna.items(), key=lambda x: x[1], reverse=True)[:n]
