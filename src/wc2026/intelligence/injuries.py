"""Injury & availability engine (plan Phase B / Tier 2).

Why this layer is different from the ones that failed the ablation gate: squad *reputation* and
*friendly weighting* re-encoded information the results-based rating already had, so they didn't
help. A key player suddenly being **unavailable** is information the past results do *not* contain
yet — it's genuinely orthogonal. That makes it the most promising contextual signal.

Honest caveat (kept front and centre): this **cannot be RPS-validated before WC2026 is played** —
there is no clean labelled injury/availability dataset for the 2018/2022 squads to backtest against.
So it does NOT enter the forecast as a gate-passed accuracy improver. It is a **principled prior
adjustment + scenario ("what-if") tool**: it lets you ask "how do Argentina's odds move if Messi is
out?" and adjusts ratings in the obvious, monotone direction. Magnitude is a modelling choice (``k``),
not a fitted truth.

Mechanism: a player's *importance share* within their squad (caps-based proxy; replace with market
value later) sets how much the team's effective strength drops when they're unavailable. The total
per-team penalty feeds the ``MatchModel`` adjustment hook so an injured team scores less / concedes
more — exactly the existing seam used by other covariates.
"""

from __future__ import annotations

from ..data.squads import Squad


def player_importance(
    squad: Squad, values: dict[str, float] | None = None
) -> dict[str, float]:
    """Each player's share of squad importance, summing to 1.

    With ``values`` (Transfermarkt market value €, from ``data.transfermarkt_values``) this prices in
    form/quality/scarcity — Yamal outweighs a 100-cap veteran, which caps got backwards. Players the
    value snapshot doesn't cover are imputed at the squad's 25th-percentile value (a plausible
    squad-filler), so they keep a small, non-zero share. Without ``values`` it falls back to the caps
    proxy (caps + ½·goals): a transparent experience signal when no values are available.
    """
    if values:
        present = sorted(values[p.name] for p in squad.players if p.name in values)
        floor = present[len(present) // 4] if present else 1.0  # ~25th-pct impute for the unmatched
        weights = {p.name: float(values.get(p.name, floor)) for p in squad.players}
    else:
        weights = {p.name: float(p.caps) + 0.5 * float(p.goals) for p in squad.players}
    total = sum(weights.values()) or 1.0
    return {name: w / total for name, w in weights.items()}


def squad_contributions(
    squad: Squad, values: dict[str, float] | None = None, top: int = 5
) -> dict:
    """Decompose a team's strength into per-player contribution shares (the player-contribution view).

    This is the defensible, data-honest contribution model: each player's share is their market-value
    weight within the squad (``player_importance``), which is a *prior* decomposition — a fitted
    marginal-goal-impact model would need player on/off tracking data that has no free source for
    internationals (the same moat as the xG breadth gap). Returns the ranked contributions plus the
    ``star_reliance`` = combined share of the ``top`` players (how fragile the team is to absences:
    a high value means one or two stars carry the side).
    """
    shares = player_importance(squad, values)
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    return {"contributions": ranked,
            "star_reliance": round(sum(s for _, s in ranked[:top]), 3)}


def availability_penalty(
    squad: Squad, unavailable: list[str], k: float = 1.0,
    values: dict[str, float] | None = None,
) -> float:
    """Team strength penalty (in log-goal units) from a list of unavailable player names.

    penalty = k · Σ importance_share(unavailable). ``k`` scales how much losing your single most
    important player hurts (default 1.0 ≈ a talisman worth ~10–16% of squad importance costs the
    team ~exp(-0.1..0.16) ≈ a 10–15% drop in expected goals). ``values`` switches the importance
    share from caps to market value.
    """
    shares = player_importance(squad, values)
    names = set(unavailable)
    return k * sum(shares.get(n, 0.0) for n in shares if n in names)


class InjuryAdjustment:
    """Log-rate adjustment from per-team availability penalties (plugs into MatchModel).

    ``penalties`` maps team -> penalty δ ≥ 0. For a fixture, the home log scoring rate is shifted by
    ``δ_away − δ_home`` (and the away side by the negative), so a depleted team scores less and
    concedes more. Teams not listed are treated as fully available.
    """

    def __init__(self, penalties: dict[str, float]):
        self.penalties = penalties

    def __call__(self, home: str, away: str) -> float:
        return self.penalties.get(away, 0.0) - self.penalties.get(home, 0.0)


def penalties_from_scenario(
    squads: dict[str, Squad], scenario: dict[str, list[str]], k: float = 1.0,
    values: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Build per-team penalties from a {team: [unavailable players]} scenario dict.

    ``values`` is the {team: {player: market_value}} table — when given, each team's importance
    shares are value-weighted instead of caps-weighted.
    """
    out: dict[str, float] = {}
    for team, players in scenario.items():
        if team in squads:
            out[team] = availability_penalty(
                squads[team], players, k=k, values=(values or {}).get(team)
            )
    return out


def load_manual_availability(path) -> dict[str, list[str]]:
    """Parse a free, hand-maintained availability file -> {team: [unavailable players]}.

    Format (one team per line, '#': comment):  ``Team: Player One, Player Two``
    The fastest, free, accurate way to feed current injuries (read off any team-news page) — the
    API-Football free tier is season-locked and can't serve live data.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, list[str]] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        team, _, players = line.partition(":")
        names = [n.strip() for n in players.split(",") if n.strip()]
        if names:
            out[team.strip()] = names
    return out
